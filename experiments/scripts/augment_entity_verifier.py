"""Second training round: near sounds, omissions and whole-span negatives.

All additions use the original training sources. No development/test
transcript supplies a replacement spelling, alias or training target.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil

from evaluate_entity_library import ROOT,read_jsonl,write_json,append_jsonl,normalized
from evaluate_entity_ranking import OUT,GENERAL,entity_targets,sets
from system.entity_store import EntityStore
from system.entity_retrieval import EntityRetriever,syllable_cost
from system.entity_matcher import _pinyin_tokens
from system.entity_ranker import candidate_input
from system.protection import EntityProtector


def main():
    base=ROOT/'results/entity_ranker_20261006';destination=base/'round2';destination.mkdir(exist_ok=True)
    if (destination/'pair_protocol.json').exists():raise ValueError('Round 2 data is frozen')
    for name in ['expanded_medical.db','recall_news_dev.jsonl','recall_medical_dev.jsonl','recall_news_test.jsonl','recall_medical_test.jsonl']:
        shutil.copyfile(base/name,destination/name)
    excluded={hashlib.sha256(normalized(r['reference']).encode()).hexdigest()
              for source,_,_ in sets().values() for r in read_jsonl(source)}
    docs={};counts=Counter();rng=random.Random(20261006)
    for partition in ['train','calibration']:
        path=destination/f'pairs_{partition}.jsonl'
        if path.exists():raise ValueError('Refusing partial-data overwrite')
        for r in read_jsonl(base/f'pairs_{partition}.jsonl'):
            if r['reference_group'] in excluded:continue
            append_jsonl(path,r);counts[partition+':'+r['kind']+':'+str(r['label'])]+=1
            if r['kind']=='synthetic_spelling' and r['label']:
                c=r['candidate'];text=r['source'][:c['start']]+c['canonical']+r['source'][c['end']:]
                if hashlib.sha256(normalized(text).encode()).hexdigest()!=r['reference_group']:continue
                key=(partition,r['domain'],r['reference_group'])
                d=docs.setdefault(key,dict(text=text,id=r['id'].split(':')[0],entities=set()))
                d['entities'].add((c['start'],c['start']+len(c['canonical']),c['canonical']))
    definitions={'general':EntityStore(GENERAL/'typed.db').list_entities(domain='general'),
                 'medical':EntityStore(base/'expanded_medical.db').list_entities(domain='medical')}
    retrievers={domain:EntityRetriever(defs) for domain,defs in definitions.items()}
    protectors={domain:EntityProtector(defs) for domain,defs in definitions.items()}
    char_counts=Counter(c for d in docs.values() for c in d['text'] if '\u3400'<=c<='\u9fff')
    phon_chars=defaultdict(list)
    for c,n in char_counts.most_common():phon_chars[_pinyin_tokens(c)[0]].append(c)
    near={p:[q for q in phon_chars if p!=q and syllable_cost(p,q)<=.5] for p in phon_chars}
    def save(text,c,label,partition,domain,group,record_id,kind):
        original,pair,features=candidate_input(text,c)
        append_jsonl(destination/f'pairs_{partition}.jsonl',dict(id=record_id,kind=kind,domain=domain,reference_group=group,
              source=text,candidate=c,label=label,original_input=original,candidate_input=pair,features=features))
        counts[partition+':'+kind+':'+str(label)]+=1
    for index,((partition,domain,group),doc) in enumerate(docs.items()):
        text=doc['text'];retriever=retrievers[domain]
        for a,b,word in sorted(doc['entities']):
            for variant in range(2):
                observed=list(word);position=rng.randrange(len(word))
                if len(word)>=3 and rng.random()<.2:
                    del observed[position];kind='synthetic_omission'
                else:
                    p=_pinyin_tokens(observed[position])[0]
                    options=[c for q in near[p] for c in phon_chars[q][:4]]
                    if not options:continue
                    observed[position]=rng.choice(options);kind='synthetic_near_sound'
                observed=''.join(observed);source=text[:a]+observed+text[b:]
                for c in retriever.alternatives(source,a,a+len(observed)):
                    save(source,c.public_dict(),int(c.canonical==word),partition,domain,group,doc['id']+f':near{variant}',kind)
        # Medical whole-span negatives counter the boundary distribution gap.
        if domain=='medical' and index%16==0:
            a,b,_=min(doc['entities']);source=text[max(0,a-25):b+25]
            blocked=[(s.start,s.end) for s in protectors[domain].protect(source).spans]
            candidates=list(retriever.retrieve(source,blocked_spans=blocked))
            for c in sorted(candidates,key=lambda c:-c.retrieval_score)[:20]:
                save(source,c.public_dict(),0,partition,domain,group,doc['id'],'whole_span_keep')
        if index%200==0:print('Documents',index+1,'/',len(docs),dict(counts),flush=True)
    for index,row in enumerate(read_jsonl(base/'training_audio/asr.jsonl')):
        group=hashlib.sha256(normalized(row['reference']).encode()).hexdigest()
        if group in excluded:continue
        partition='calibration' if int(group[:8],16)%10<2 else 'train';domain='general'
        targets=entity_targets(row,definitions[domain]);source=row['raw_text']
        blocked=[(s.start,s.end) for s in protectors[domain].protect(source).spans]
        all_candidates=list(retrievers[domain].retrieve(source,blocked_spans=blocked))
        positive=[];negative=[]
        for c in all_candidates:
            if any(c.canonical==t['canonical'] and c.start==t['start'] and c.end==t['end'] and t['category']=='text_correctable_candidate' for t in targets):positive.append(c)
            elif not any(c.start<t['end'] and t['start']<c.end for t in targets):negative.append(c)
        ordered=sorted(negative,key=lambda c:-c.retrieval_score)
        selected=ordered[:20]+rng.sample(ordered[20:],min(10,max(0,len(ordered)-20)))
        for c in positive+selected:save(source,c.public_dict(),int(c in positive),partition,domain,group,row['id']+':'+row['variant'],'whole_span_real')
        if index%100==0:print('ASR negatives',index+1,flush=True)
    write_json(destination/'pair_protocol.json',dict(seed=20261006,counts=counts,source='Original round-1 train documents and real train ASR only',
        split='Same reference-hash train/calibration partitions; all known evaluation references excluded',no_reference_overlap=True,
        training_documents=len(docs),synthetic_only_for_training=True,
        additions='Near-phoneme substitutions, single-character omissions, full-window hard negatives'))
    print(dict(counts),flush=True)


if __name__=='__main__':main()
