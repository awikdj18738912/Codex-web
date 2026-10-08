"""Dedicated supervised verifier; audio train speakers and text train only.

Synthetic corruptions supplement real ASR errors, and are never reported as
real-audio improvements. Duplicate reference sentences stay in one partition.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time

from evaluate_entity_library import ROOT,read_jsonl,write_json,append_jsonl,normalized,parse_ner
from evaluate_entity_ranking import OUT,GENERAL,MED,CORPUS,entity_targets,sets
from system.entity_store import EntityStore
from system.entity_retrieval import EntityRetriever
from system.entity_matcher import _pinyin_tokens
from system.entity_boundaries import _tokens
from system.entity_ranker import candidate_input,FEATURE_NAMES,build_network
from system.protection import EntityProtector


def fold(reference):
    return int(hashlib.sha256(normalized(reference).encode()).hexdigest()[:8],16)%10


def build_pairs():
    rng=random.Random(20261006)
    general=EntityStore(GENERAL/'typed.db').list_entities(domain='general')
    medical=EntityStore(OUT/'expanded_medical.db').list_entities(domain='medical')
    retrievers={'general':EntityRetriever(general),'medical':EntityRetriever(medical)}
    defs={'general':general,'medical':medical};docs=[]
    train_ids={p.stem for p in (CORPUS/'wav/train').rglob('*.wav')}
    seen=set();byword={d.canonical_text:d for d in general}
    for line in (GENERAL/'source/aishell_ner_transcript.txt').read_text().splitlines():
        key,text=line.split(maxsplit=1)
        if key not in train_ids:continue
        text,entities=parse_ner(text)
        if text in seen:continue
        seen.add(text)
        relevant=[e for e in entities if e['text'] in byword and re.fullmatch(r'[\u3400-\u9fff]{2,24}',e['text'])]
        if relevant:docs.append(dict(id=key,reference=text,entities=relevant,domain='general'))
    rng.shuffle(docs);docs=docs[:2000]
    med_docs=[];med_byword={d.canonical_text:d for d in medical}
    for i,r in enumerate(json.loads((MED/'source/CMeEE_train.json').read_text())):
        entities=[dict(text=e['entity'],start=e['start_idx'],end=e['end_idx']+1,entity_type='TERM')
                  for e in r['entities'] if e['entity'] in med_byword and re.fullmatch(r'[\u3400-\u9fff]{2,24}',e['entity'])]
        if entities:med_docs.append(dict(id=f'CMeEE_train_{i}',reference=r['text'],entities=entities,domain='medical'))
    rng.shuffle(med_docs);docs+=med_docs[:1600]
    char_counts=Counter(c for d in docs for c in d['reference'] if '\u3400'<=c<='\u9fff')
    homophones=defaultdict(list)
    for c,_ in char_counts.most_common():homophones[_pinyin_tokens(c)[0]].append(c)
    for filename in ['pairs_train.jsonl','pairs_calibration.jsonl']:
        if (OUT/filename).exists():raise ValueError('Refusing to overwrite frozen pair data')
    counts=Counter();group_keys={'train':set(),'calibration':set()}
    def save(text,candidate,label,reference,record_id,kind,domain):
        partition='calibration' if fold(reference)<2 else 'train'
        original,pair,features=candidate_input(text,candidate)
        item=dict(id=record_id,kind=kind,domain=domain,reference_group=hashlib.sha256(normalized(reference).encode()).hexdigest(),
                  source=text,candidate=candidate,label=label,original_input=original,candidate_input=pair,features=features)
        append_jsonl(OUT/f'pairs_{partition}.jsonl',item);counts[partition+':'+kind+':'+str(label)]+=1;group_keys[partition].add(item['reference_group'])
    # Real ASR train errors and competing wrong candidates are supervised first.
    for row in read_jsonl(OUT/'training_audio/asr.jsonl'):
        domain='general';targets=entity_targets(row,general)
        for target in targets:
            if target['category']!='text_correctable_candidate':continue
            for c in retrievers[domain].alternatives(row['raw_text'],target['start'],target['end']):
                save(row['raw_text'],c.public_dict(),int(c.canonical==target['canonical']),row['reference'],row['id']+':'+row['variant'],'real_asr',domain)
        negative=[]
        for word,a,b in _tokens(row['raw_text']):
            if len(word)<2 or any(a<t['end'] and t['start']<b for t in targets):continue
            negative.extend(retrievers[domain].alternatives(row['raw_text'],a,b))
        for c in sorted(negative,key=lambda c:-c.retrieval_score)[:5]:
            save(row['raw_text'],c.public_dict(),0,row['reference'],row['id']+':'+row['variant'],'real_keep',domain)
    for index,doc in enumerate(docs):
        domain=doc['domain'];retriever=retrievers[domain];text=doc['reference']
        # Correct ordinary expressions and already-correct entity values are
        # essential hard negatives, including all-candidates-wrong examples.
        negative=[]
        for word,a,b in _tokens(text):
            if len(word)<2:continue
            negative.extend(retriever.alternatives(text,a,b))
        for c in sorted(negative,key=lambda c:-c.retrieval_score)[:4]:save(text,c.public_dict(),0,text,doc['id'],'text_keep',domain)
        for e in rng.sample(doc['entities'],min(2,len(doc['entities']))):
            canonical=e['text'];a,b=e['start'],e['end']
            if text[a:b]!=canonical:continue
            for variant in range(2):
                observed=list(canonical)
                chosen=rng.sample(range(len(observed)),min(len(observed),1+int(rng.random()<0.25)))
                changed=False
                for position in chosen:
                    options=[c for c in homophones[_pinyin_tokens(observed[position])[0]][:24] if c!=observed[position]]
                    if options:observed[position]=rng.choice(options);changed=True
                if not changed:continue
                observed=''.join(observed);source=text[:a]+observed+text[b:]
                candidates=retriever.alternatives(source,a,a+len(observed))
                for c in candidates:
                    save(source,c.public_dict(),int(c.canonical==canonical),text,doc['id']+f':corrupt{variant}','synthetic_spelling',domain)
        if index%200==0:print(index+1,'/',len(docs),dict(counts),flush=True)
    assert not group_keys['train']&group_keys['calibration']
    write_json(OUT/'pair_protocol.json',dict(seed=20261006,counts=counts,training_documents=len(docs),
        source='AISHELL train real ASR + AISHELL train NER text + CMeEE train text',
        split='Normalized reference sentence hash; 80% train, 20% calibration; no test audio or captions',
        no_reference_overlap=True,synthetic_only_for_training=True))
    print(dict(counts),flush=True)


def train(args):
    import torch
    from transformers import AutoTokenizer
    import numpy as np
    seed=20261006;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    encoder=Path(os.environ.get('ENTITY_VERIFIER_ENCODER',str(ROOT/'models/entity_ranker/encoder'))).resolve();tokenizer=AutoTokenizer.from_pretrained(encoder)
    train_rows=read_jsonl(OUT/'pairs_train.jsonl');cal=read_jsonl(OUT/'pairs_calibration.jsonl')
    # AISHELL can repeat a news sentence across official speaker splits.
    # Exclusion uses reference hashes only; no evaluation spelling/label is
    # learned. Keep the original pair files as a reproducible audit trail.
    excluded={hashlib.sha256(normalized(r['reference']).encode()).hexdigest()
              for source,_,_ in sets().values() for r in read_jsonl(source)}
    overlap=dict(train=sum(r['reference_group'] in excluded for r in train_rows),
                 calibration=sum(r['reference_group'] in excluded for r in cal))
    train_rows=[r for r in train_rows if r['reference_group'] not in excluded]
    cal=[r for r in cal if r['reference_group'] not in excluded]
    write_json(OUT/'training_reference_exclusion.json',dict(removed_pairs=overlap,
        rule='Exclude reference hashes of all existing development, holdout and control sets before fitting or calibration',
        remaining_train=len(train_rows),remaining_calibration=len(cal),evaluation_reference_overlap=0))
    if args.real_error_repeats>1:
        real=[r for r in train_rows if r['kind']=='real_asr']
        train_rows+=real*(args.real_error_repeats-1)
    print('Pairs',len(train_rows),len(cal),'positive',sum(r['label'] for r in train_rows),flush=True)
    net=build_network(encoder).to(args.device);opt=torch.optim.AdamW(net.parameters(),lr=args.learning_rate,weight_decay=0.01)
    destination=Path(os.environ.get('ENTITY_VERIFIER_MODEL',str(ROOT/'models/entity_ranker/verifier'))).resolve();destination.mkdir(parents=True,exist_ok=True)
    if not (destination/'encoder').exists():shutil.copytree(encoder,destination/'encoder')
    def tensors(rows):
        encoded=tokenizer([r['original_input'] for r in rows],[r['candidate_input'] for r in rows],padding=True,truncation=True,max_length=160,return_tensors='pt')
        encoded={k:v.to(args.device) for k,v in encoded.items()}
        return encoded,torch.tensor([r['features'] for r in rows],dtype=torch.float32,device=args.device),torch.tensor([r['label'] for r in rows],dtype=torch.float32,device=args.device)
    for epoch in range(args.epochs):
        random.Random(seed+epoch).shuffle(train_rows);net.train();loss_sum=0;started=time.perf_counter()
        for offset in range(0,len(train_rows),args.batch_size):
            rows=train_rows[offset:offset+args.batch_size];encoded,features,labels=tensors(rows)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=args.mixed_precision):
                logits=net(**encoded,features=features)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),labels)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
            loss_sum+=loss.item()*len(rows)
        net.eval();scored=[]
        with torch.inference_mode():
            for offset in range(0,len(cal),128):
                rows=cal[offset:offset+128];encoded,features,labels=tensors(rows)
                probs=torch.sigmoid(net(**encoded,features=features)).cpu().tolist()
                scored.extend(dict(id=r['id'],kind=r['kind'],domain=r['domain'],reference_group=r['reference_group'],label=r['label'],score=p) for r,p in zip(rows,probs))
        torch.save(net.cpu().state_dict(),destination/f'verifier_epoch{epoch+1}.pt');net.to(args.device)
        write_json(OUT/f'calibration_epoch{epoch+1}.json',scored)
        print('EPOCH',epoch+1,'loss',loss_sum/len(train_rows),'seconds',time.perf_counter()-started,flush=True)
    write_json(destination/'metadata.json',dict(version=1,feature_names=FEATURE_NAMES,seed=seed,encoder=str(encoder),
        training_protocol=json.loads((OUT/'pair_protocol.json').read_text()),epochs=args.epochs,
        real_error_repeats=args.real_error_repeats,learning_rate=args.learning_rate,batch_size=args.batch_size,mixed_precision=args.mixed_precision,
        note='Experimental checkpoint, not enabled in live matching until full calibration and heldout validation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pairs','train']);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--real-error-repeats',type=int,default=1)
    p.add_argument('--mixed-precision',action='store_true');a=p.parse_args()
    build_pairs() if a.stage=='pairs' else train(a)
