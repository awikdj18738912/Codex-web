"""Train-context, exact-pronunciation spelling pilot for registered names.

Scope is explicit: names registered in a glossary, not arbitrary typo repair.
No test ASR spelling is stored as an alias. Profiles come from train/dev NER
contexts; the independently sampled test recordings are excluded from fitting.
"""
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re

from evaluate_entity_library import ROOT,read_jsonl,write_json,append_jsonl,parse_ner,normalized
from system.entity_boundaries import _tokens,candidate_boundary_issue
from system.entity_matcher import EntityCandidateMatcher,EntityDecision,EntityMatchReport,_pinyin_tokens
from system.entity_store import EntityStore
from system.protection import EntityProtector
from rapidfuzz.distance import Levenshtein

OUT=ROOT/'results/entity_context_20261006'
BASE=ROOT/'results/entity_library_20261003'
CORPUS=Path('/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell')


def prepare():
    words={v['canonical_text'] for v in json.loads((BASE/'vocabulary.json').read_text()) if v['entity_type']=='PERSON'}
    allowed={p.stem for split in ['train','dev'] for p in (CORPUS/'wav'/split).rglob('*.wav')}
    counters=defaultdict(Counter);occ=Counter()
    source=BASE/'source/aishell_ner_transcript.txt'
    for line in source.read_text().splitlines():
        record_id,annotated=line.split(maxsplit=1)
        if record_id not in allowed:continue
        text,entities=parse_ner(annotated)
        for e in entities:
            if e['text'] not in words or e['entity_type']!='PERSON':continue
            surrounding=text[max(0,e['start']-24):e['start']]+'。'+text[e['end']:e['end']+24]
            tokens={word for word,_,_ in _tokens(surrounding) if len(word)>=2 and re.fullmatch(r'[\u3400-\u9fff]+',word)}
            counters[e['text']].update(tokens);occ[e['text']]+=1
    df=Counter(w for cs in counters.values() for w in cs)
    profiles={}
    for word,cs in counters.items():
        weights={t:math.log1p(c)*math.log1p(len(counters)/(1+df[t])) for t,c in cs.items() if t not in word}
        ranked=sorted(weights,key=lambda t:(-weights[t],t))[:24]
        profiles[word]=dict(keywords={t:round(weights[t],4) for t in ranked},min_score=2.0,
                            training_occurrences=occ[word],source='AISHELL-NER train+dev contexts',
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    write_json(OUT/'name_profiles.json',profiles)
    print('Profiles',len(profiles),flush=True)


def confirms(text,match,profiles,definitions):
    d=definitions[match.entity_id];p=profiles.get(d.canonical_text)
    if not p or d.entity_type!='PERSON' or d.normalization_policy!='normalize':return False
    if len(match.observed)!=len(d.canonical_text) or len(match.observed)<3:return False
    if match.observed[0]!=d.canonical_text[0]:return False
    py=_pinyin_tokens(match.observed)
    if py!=_pinyin_tokens(d.canonical_text):return False
    if sum(_pinyin_tokens(v.canonical_text)==py for v in definitions.values())!=1:return False
    if candidate_boundary_issue(text,match.start,match.end):return False
    context=text[max(0,match.start-32):match.start]+'。'+text[match.end:match.end+32]
    evidence=[w for w in p['keywords'] if w in context]
    score=sum(p['keywords'][w] for w in evidence)
    return score>=p['min_score']


def evaluate():
    profiles=json.loads((OUT/'name_profiles.json').read_text())
    defs=EntityStore(BASE/'typed.db').list_entities(domain='general');byid={d.entity_id:d for d in defs}
    matcher=EntityCandidateMatcher(defs,selected_domain='general',mode='auto');protector=EntityProtector(defs)
    dest=OUT/'name_pilot.jsonl'
    if dest.exists():raise ValueError('Frozen pilot exists')
    stats=Counter();changes=[]
    for r in read_jsonl(BASE/'asr.jsonl'):
        exact=protector.protect(r['raw_text'])
        report=matcher.match(r['raw_text'],allow_auto=True,blocked_spans=[(s.start,s.end) for s in exact.spans])
        confirmed=[replace(m,decision=EntityDecision.AUTO_NORMALIZE,reasons=('registered_pronunciation_context',),match_type='CONTEXT_PINYIN')
                   for m in report.matches if confirms(r['raw_text'],m,profiles,byid)]
        protected=protector.protect(r['raw_text'],fuzzy_matches=[*report.auto_matches,*confirmed]);out=protector.restore(protected.masked_text,protected).text
        a,b,ref=normalized(r['raw_text']),normalized(out),normalized(r['reference'])
        ae,be=Levenshtein.distance(ref,a),Levenshtein.distance(ref,b)
        stats['samples']+=1;stats['error_delta']+=be-ae;stats['improved']+=be<ae;stats['worsened']+=be>ae
        for e in r['entities']:
            term=normalized(e['text']);stats['entity_gain']+=term in b and term not in a;stats['entity_loss']+=term in a and term not in b
        record=dict(id=r['id'],variant=r['variant'],source=r['raw_text'],output=out,reference=r['reference'],
                    confirmed=[m.public_dict() for m in confirmed],error_delta=be-ae)
        append_jsonl(dest,record)
        if a!=b:changes.append(record)
    write_json(OUT/'name_pilot_summary.json',dict(stats));write_json(OUT/'name_pilot_changes.json',changes)
    print(dict(stats),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);a=p.parse_args()
    prepare() if a.stage=='prepare' else evaluate()
