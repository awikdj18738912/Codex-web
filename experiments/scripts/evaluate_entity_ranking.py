"""Audit, broaden recall and train an entity-specific verifier in isolation.

The existing ASR/Refiner are held fixed. References label evaluation records,
and never enter retrieval/model requests at inference time.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,append_jsonl,normalized,parse_ner,balanced_sample,audio_bytes
from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher,_pinyin_tokens,_normalize_with_offsets
from system.entity_retrieval import EntityRetriever,sound_similarity,syllable_cost
from system.protection import EntityProtector

OUT=Path(os.environ.get('ENTITY_VERIFIER_RUN',str(ROOT/'results/entity_ranker_20261006'))).resolve()
OLD=ROOT/'results/entity_context_20261006'
MED=ROOT/'results/medical_entity_20261003'
GENERAL=ROOT/'results/entity_library_20261003'
CORPUS=Path('/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell')


@lru_cache(maxsize=1024)
def aligned_boundaries(reference,hypothesis):
    """Sound-aware annotation alignment, not a claim about audible content."""
    rp,hp=_pinyin_tokens(reference),_pinyin_tokens(hypothesis)
    n,m=len(reference),len(hypothesis)
    costs=[[0.0]*(m+1) for _ in range(n+1)];moves={}
    for i in range(1,n+1):costs[i][0]=i*0.8
    for j in range(1,m+1):costs[0][j]=j*0.8
    for i in range(1,n+1):
        for j in range(1,m+1):
            sub=0 if reference[i-1]==hypothesis[j-1] else 0.05 if rp[i-1]==hp[j-1] else 0.2+0.8*syllable_cost(rp[i-1],hp[j-1])
            values=[costs[i-1][j-1]+sub,costs[i-1][j]+0.8,costs[i][j-1]+0.8]
            move=min(range(3),key=lambda k:values[k]);costs[i][j]=values[move];moves[i,j]=move
    mapping=[None]*(n+1);i,j=n,m;mapping[n]=m
    while i or j:
        move=moves.get((i,j),1 if i else 2)
        if move==0:i-=1;j-=1
        elif move==1:i-=1
        else:j-=1
        if mapping[i] is None:mapping[i]=j
    return tuple(mapping)


def aligned_span(reference,hypothesis,start,end):
    mapping=aligned_boundaries(reference,hypothesis)
    return mapping[start],mapping[end]


def entity_targets(row,definitions):
    byword={normalized(d.canonical_text):d for d in definitions}
    ref=normalized(row['reference']);source,offsets=_normalize_with_offsets(row['raw_text'],0)
    used=defaultdict(int);targets=[]
    for e in sorted(row['entities'],key=lambda e:e['start']):
        word=normalized(e['text']);d=byword.get(word)
        if d is None:continue
        positions=[m.start() for m in re.finditer(re.escape(word),ref)]
        if used[word]>=len(positions):continue
        start=positions[used[word]];used[word]+=1;end=start+len(word)
        a,b=aligned_span(ref,source,start,end);observed=source[a:b]
        if word in observed or source.count(word)>=ref.count(word):continue
        if a>=len(offsets) or b<=a:
            category='no_aligned_text'
            original_start=original_end=len(row['raw_text'])
        else:
            original_start=offsets[a];original_end=offsets[min(b-1,len(offsets)-1)]+1
            protected=[(m.start(),m.end()) for w in byword if len(w)>=len(observed) for m in re.finditer(re.escape(w),source)]
            if observed in byword or any(a<y and x<b for x,y in protected):category='different_valid_entity_needs_audio'
            elif len(observed)<max(2,0.6*len(word)) or len(observed)>len(word)+2:category='length_or_omission_needs_review'
            elif sound_similarity(_pinyin_tokens(observed),_pinyin_tokens(word))<0.4:category='weak_sound_alignment_needs_review'
            else:category='text_correctable_candidate'
        targets.append(dict(canonical=d.canonical_text,entity_id=d.entity_id,entity_type=d.entity_type,
                            observed=observed,start=original_start,end=original_end,category=category))
    return targets


def sets():
    return {
        'medical_dev':(MED/'development/asr.jsonl',MED/'medical.db','medical'),
        'medical_test':(MED/'asr.jsonl',MED/'medical.db','medical'),
        'clinical_test':(MED/'asr.jsonl',OUT/'expanded_medical.db','medical'),
        'news_dev':(GENERAL/'asr.jsonl',GENERAL/'typed.db','general'),
        'news_test':(OLD/'name_holdout_asr.jsonl',GENERAL/'typed.db','general'),
        'fresh_control':(OLD/'fresh_asr.jsonl',GENERAL/'typed.db','general'),
        'mmed_dev':(OLD/'mmed_dev_asr.jsonl',OUT/'expanded_medical.db','medical'),
        'mmed_test':(OLD/'mmed_test_asr.jsonl',OUT/'expanded_medical.db','medical'),
    }


def recall(args):
    source,db,domain=sets()[args.split]
    definitions=EntityStore(db).list_entities(domain=domain)
    old=EntityCandidateMatcher(definitions,selected_domain=domain,mode='auto')
    retriever=EntityRetriever(definitions);protector=EntityProtector(definitions)
    dest=OUT/f'recall_{args.split}.jsonl';done={(r['id'],r['variant']) for r in read_jsonl(dest)}
    rows=read_jsonl(source)
    if args.split.startswith('mmed') or args.split=='clinical_test':
        from evaluate_medical_entities import lexicon_occurrences
        rows=[dict(r,entities=lexicon_occurrences(r['reference'],[d.canonical_text for d in definitions])) for r in rows]
    for i,row in enumerate(rows):
        if (row['id'],row['variant']) in done:continue
        targets=entity_targets(row,definitions)
        blocked=[(s.start,s.end) for s in protector.protect(row['raw_text']).spans]
        started=time.perf_counter();new=retriever.retrieve(row['raw_text'],blocked_spans=blocked)
        new_ms=(time.perf_counter()-started)*1000
        previous=old.match(row['raw_text'],allow_auto=True,blocked_spans=blocked)
        for t in targets:
            t['old_recalled']=any(m.canonical==t['canonical'] and m.start<=t['start'] and t['end']<=m.end for m in previous.matches)
            t['new_recalled']=any(m.canonical==t['canonical'] and abs(m.start-t['start'])<=1 and abs(m.end-t['end'])<=1 for m in new)
            t['exact_span_recalled']=any(m.canonical==t['canonical'] and m.start==t['start'] and m.end==t['end'] for m in new)
        append_jsonl(dest,dict(id=row['id'],variant=row['variant'],source=row['raw_text'],reference=row['reference'],
            speaker=row['speaker'],entities=row['entities'],targets=targets,candidates=[c.public_dict() for c in new],retrieval_ms=new_ms))
        if targets or i%20==0:print(i+1,args.split,row['id'],len(new),[(t['canonical'],t['observed'],t['category'],t['new_recalled']) for t in targets],flush=True)
    recall_report(args.split)


def recall_report(split):
    counts=Counter();categories=Counter();cases=[]
    for r in read_jsonl(OUT/f'recall_{split}.jsonl'):
        counts['samples']+=1;counts['candidate_pairs']+=len(r['candidates'])
        for t in r['targets']:
            categories[t['category']]+=1
            if t['category']=='text_correctable_candidate':
                counts['eligible_occurrences']+=1;counts['old_recalled']+=t['old_recalled'];counts['new_recalled']+=t['new_recalled'];counts['exact_span_recalled']+=t['exact_span_recalled']
            cases.append(dict(id=r['id'],variant=r['variant'],source=r['source'],reference=r['reference'],**t))
    write_json(OUT/f'recall_{split}_summary.json',dict(counts=counts,categories=categories,
        note='Text alignment categories require reference/audio review. They are not an automatic assertion that the audio contains missing caption text.'))
    write_json(OUT/f'audit_{split}.json',cases)
    print(dict(counts),dict(categories),flush=True)


def prepare_training():
    # Speech used for supervised training is from AISHELL train speakers only.
    known={d.canonical_text for d in EntityStore(GENERAL/'typed.db').list_entities()}
    annotations={}
    for line in (GENERAL/'source/aishell_ner_transcript.txt').read_text().splitlines():
        key,text=line.split(maxsplit=1);annotations[key]=parse_ner(text)
    refpath=CORPUS/'transcript/aishell_transcript_v0.8.txt'
    refs={key:''.join(text.split()) for key,text in [l.split(maxsplit=1) for l in refpath.read_text().splitlines()]}
    pools={'covered':[],'unknown':[],'ordinary':[]}
    for path in sorted((CORPUS/'wav/train').rglob('*.wav')):
        if path.stem not in annotations:continue
        text,entities=annotations[path.stem]
        if normalized(text)!=normalized(refs[path.stem]):continue
        covered=any(e['text'] in known for e in entities)
        group='covered' if covered else 'unknown' if entities else 'ordinary'
        pools[group].append(dict(id=path.stem,speaker=path.parent.name,audio_path=str(path),reference=refs[path.stem],entities=entities,group=group))
    rng=random.Random(20261006);rows=[]
    for group,n in [('covered',160),('unknown',40),('ordinary',40)]:rows+=balanced_sample(pools[group],n,rng)
    write_json(OUT/'training_selection.json',dict(seed=20261006,source_split='AISHELL train only',noise_snr_db=5.0,
        annotation_sha256=hashlib.sha256((GENERAL/'source/aishell_ner_transcript.txt').read_bytes()).hexdigest(),samples=rows))
    print('Training recordings',len(rows),flush=True)


def decode_training():
    from evaluate_entity_library import decode
    selection=json.loads((OUT/'training_selection.json').read_text());dest=OUT/'training_audio';dest.mkdir(exist_ok=True)
    write_json(dest/'selection.json',selection)
    from types import SimpleNamespace
    decode(SimpleNamespace(output=dest,asr_url='http://127.0.0.1:8766',seed=20261006,noise_snr=5.0))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['recall','recall-report','prepare-training','decode-training']);p.add_argument('--split',choices=list(sets()),default='medical_dev');a=p.parse_args()
    {'recall':lambda:recall(a),'recall-report':lambda:recall_report(a.split),'prepare-training':prepare_training,'decode-training':decode_training}[a.stage]()


if __name__=='__main__':main()
