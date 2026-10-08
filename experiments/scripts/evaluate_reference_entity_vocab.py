"""Reference-derived medical vocabulary, frozen before audio decoding.

This is an intentionally known-vocabulary coverage experiment requested by
the user, not an unseen-entity generalization benchmark. References supply
standard entity names only; ASR never receives the reference transcript.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import random
import re
from types import SimpleNamespace

from rapidfuzz.distance import Levenshtein

from evaluate_entity_library import ROOT, read_jsonl, write_json, append_jsonl, decode, normalized
from evaluate_medical_entities import lexicon_occurrences, EXCLUDED_FRAGMENTS
from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.protection import EntityProtector

OUT = ROOT/'results/entity_reference_vocab_20261007'
SOURCE = ROOT/'results/medical_entity_20261003/source'
KEYWORDS = ('患者','病人','医生','医院','手术','治疗','检查','诊断','症状','疾病','接受','进行','患有','患上','需要','开刀')
GENERIC = {'手术','检查','治疗','患者','病人','疾病','症状','细胞','器官','组织','疼痛','切除','变性','增生',
           '意外损伤','切除肿瘤组织','肠切除'}
# Whole phrases reviewed directly in the selected references BEFORE ASR.
# These extend the dictionary-assisted extraction; none is a guessed error alias.
REVIEWED_PHRASES = ('腹腔镜手术','ECMO手术','动静脉瘘手术','胰十二指肠切除术',
                    '混合静脉血氧饱和度监视仪','心血管内科')


def extract(text, labels):
    """Match independently annotated medical spellings in this reference."""
    candidates=[]
    for word, types in labels.items():
        if len(word)<3 or word in GENERIC or word in EXCLUDED_FRAGMENTS:
            continue
        start = text.find(word)
        while start >= 0:
            candidates.append(dict(text=word,start=start,end=start+len(word),annotation_types=sorted(types)))
            start = text.find(word, start+len(word))
    # Independently declared common acronyms; do not extract ASCII substrings.
    for m in re.finditer(r'(?<![A-Za-z])(?:TAVI|ENB|PSA|MRI|ECMO)(?![A-Za-z])',text):
        candidates.append(dict(text=m.group(),start=m.start(),end=m.end(),annotation_types=['reviewed_acronym']))
    for word in REVIEWED_PHRASES:
        for m in re.finditer(re.escape(word),text):
            candidates.append(dict(text=word,start=m.start(),end=m.end(),annotation_types=['reference_phrase_review']))
    selected=[]
    for entry in sorted(candidates,key=lambda e:(-(e['end']-e['start']),e['start'],e['text'])):
        if not any(entry['start']<e['end'] and e['start']<entry['end'] for e in selected):
            selected.append(entry)
    return sorted(selected,key=lambda e:e['start'])


def prepare():
    import pyarrow.parquet as pq
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    if (OUT/'selection.json').exists():raise FileExistsError('Selection already frozen')
    cmeee=json.loads((SOURCE/'CMeEE_train.json').read_text())
    labels=defaultdict(set)
    for row in cmeee:
        for e in row['entities']:
            word=e['entity']
            if 3<=len(word)<=24 and re.fullmatch(r'[\u3400-\u9fffA-Za-z0-9-]+',word):
                labels[word].add(e['type'])
    previous=[]
    for folder in [ROOT/'results/entity_medical_scenes_20261006',ROOT/'results/entity_medical_scenes_20261006/holdout']:
        previous+=json.loads((folder/'selection.json').read_text())['samples']
    excluded={r['source_row'] for r in previous}
    rows=pq.read_table(SOURCE/'chinese_train.parquet').to_pylist()
    pool=[]
    for index,row in enumerate(rows):
        if index in excluded or not 3<=row['duration']<=20:continue
        # Pre-ASR caption-quality screening, not selection by recognizer errors.
        without_acronyms=re.sub(r'TAVI|ENB|PSA|MRI|ECMO','',row['text'])
        if re.search(r'[A-Za-z]',without_acronyms):continue
        chinese=sum('\u3400'<=c<='\u9fff' for c in row['text'])
        if not 2<=chinese/row['duration']<=8:continue
        entities=extract(row['text'],labels)
        if entities:pool.append((index,row,entities))
    random.Random(20261008).shuffle(pool)
    chosen=pool[:64]
    samples=[];vocabulary={}
    for index,row,entities in chosen:
        record_id=f'MultiMed_Chinese_train_{index:04d}'
        path=OUT/'audio'/(record_id+'.wav');path.parent.mkdir(exist_ok=True)
        audio,rate=sf.read(io.BytesIO(row['audio']['bytes']),dtype='float32')
        if audio.ndim>1:audio=audio.mean(axis=1)
        if rate!=16000:
            d=gcd(rate,16000);audio=resample_poly(audio,16000//d,rate//d).astype('float32')
        sf.write(path,audio,16000,subtype='FLOAT')
        samples.append(dict(id=record_id,audio_path=str(path.resolve()),reference=row['text'],entities=entities,
            group='reference_entities',source_row=index,actual_duration_seconds=len(audio)/16000,
            original_audio_sha256=hashlib.sha256(row['audio']['bytes']).hexdigest()))
        for e in entities:
            word=e['text'];item=vocabulary.setdefault(word,dict(canonical_text=word,annotation_types=e['annotation_types'],evidence=[]))
            item['evidence'].append(dict(id=record_id,reference=row['text'],start=e['start'],end=e['end']))
    store=EntityStore(OUT/'reference_terms.db')
    for word,item in sorted(vocabulary.items()):
        definition=store.upsert_entity(word,domain='medical',entity_type='TERM',normalization_policy='normalize',
            source='Entity extracted from evaluated MultiMed reference; controlled known-vocabulary benchmark')
        if re.fullmatch(r'[\u3400-\u9fff]{3,16}',word):
            words={w:1.0 for w in KEYWORDS if w not in word}
            store.set_spelling_context(definition.entity_id,keywords=words,minimum_score=1,
                source='Fixed generic medical context list; no per-record transcript provided at inference')
    write_json(OUT/'reference_entities.json',list(vocabulary.values()))
    write_json(OUT/'selection.json',dict(seed=20261008,noise_snr_db=5,samples=samples,
        source_url='https://huggingface.co/datasets/leduckhai/MultiMed',
        extraction='CMeEE-annotated medical spellings matched in evaluated reference transcripts; longest non-overlapping names',
        coverage_role='Intentional reference-derived vocabulary; not independent generalization',
        excluded_previously_decoded_train_rows=sorted(excluded),eligible_pool=len(pool),
        source_sha256=hashlib.sha256((SOURCE/'chinese_train.parquet').read_bytes()).hexdigest(),
        caveats=['Human public audio; supplied captions may omit speech or contain typos',
                 'Pre-ASR language/duration screening cannot certify verbatim correctness',
                 'All successes must be checked against their local reference and actual edit',
                 'Unknown speaker/source video overlap; noisy versions not independent recordings']))
    write_json(OUT/'frozen_inputs.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [OUT/'selection.json',OUT/'reference_entities.json',OUT/'reference_terms.db']})
    print('FROZEN',len(samples),'recordings',len(vocabulary),'entities',sum(len(r['entities']) for r in samples),'mentions',flush=True)


def evaluate(label):
    dest=OUT/f'{label}.jsonl'
    if dest.exists():raise FileExistsError(dest)
    definitions=EntityStore(OUT/('optimized_terms.db' if label=='final' else 'reference_terms.db')).list_entities(domain='medical')
    conditions={'standard_only':tuple(replace(d,spelling_context=None) for d in definitions),'with_context':definitions}
    if label=='final':
        conditions['peer_context_without_near']=tuple(replace(d,spelling_context=replace(d.spelling_context,allow_near_spelling=False)) if d.spelling_context else d for d in definitions)
    tools={k:(EntityProtector(v,protect_numeric_spans=False),EntityCandidateMatcher(v,selected_domain='medical',mode='auto')) for k,v in conditions.items()}
    baseline={(r['id'],r['variant'],r['condition']):r for r in read_jsonl(OUT/'baseline.jsonl')} if label!='baseline' else {}
    summary={k:Counter() for k in conditions};changes=[]
    for row in read_jsonl(OUT/'asr.jsonl'):
        for condition,(protector,matcher) in tools.items():
            p=prepare_entity_segment(row['raw_text'],protector,matcher,allow_auto=True)
            r=dict(id=row['id'],variant=row['variant'],condition=condition,source=row['raw_text'],reference=row['reference'],
                output=p.baseline_text,normalizations=p.normalizations,hints=p.hints,candidates=[m.public_dict() for m in p.report.matches])
            append_jsonl(dest,r)
            old=baseline.get((row['id'],row['variant'],condition if condition!='peer_context_without_near' else 'with_context'),dict(output=row['raw_text']))['output']
            ref=normalized(row['reference']);before=Levenshtein.distance(ref,normalized(old));after=Levenshtein.distance(ref,normalized(p.baseline_text))
            s=summary[condition];s.update(samples=1,changed=int(old!=p.baseline_text),improved=int(after<before),regressed=int(after>before),error_delta=after-before)
            # Longest occurrences prevent counting a short name inside a
            # different registered longer name as a correct independent mention.
            words=[d.canonical_text for d in definitions]
            have=Counter(e['text'] for e in lexicon_occurrences(p.baseline_text,words))
            expected=Counter(e['text'] for e in row['entities'])
            s['expected_mentions']+=sum(expected.values())
            s['recovered_mentions']+=sum(min(n,have[w]) for w,n in expected.items())
            if old!=p.baseline_text:changes.append(dict(r,baseline=old))
    write_json(OUT/f'{label}_summary.json',{k:dict(v) for k,v in summary.items()})
    write_json(OUT/f'{label}_changes.json',changes)
    print(json.dumps({k:dict(v) for k,v in summary.items()},ensure_ascii=False,indent=2),flush=True)


def review_prepared():
    """Correct extraction boundaries before the first audio decode."""
    if (OUT/'asr.jsonl').exists():raise ValueError('Refusing to change vocabulary after decoding')
    data=json.loads((OUT/'selection.json').read_text())
    annotations=json.loads((SOURCE/'CMeEE_train.json').read_text());labels=defaultdict(set)
    for row in annotations:
        for e in row['entities']:
            if 3<=len(e['entity'])<=24 and re.fullmatch(r'[\u3400-\u9fffA-Za-z0-9-]+',e['entity']):labels[e['entity']].add(e['type'])
    before=OUT/'extraction_before_phrase_review';before.mkdir(exist_ok=True)
    import shutil
    for name in ['selection.json','reference_entities.json','reference_terms.db','frozen_inputs.json']:
        shutil.copy2(OUT/name,before/name)
    vocabulary={}
    for row in data['samples']:
        row['entities']=extract(row['reference'],labels)
        for e in row['entities']:
            item=vocabulary.setdefault(e['text'],dict(canonical_text=e['text'],annotation_types=e['annotation_types'],evidence=[]))
            item['evidence'].append(dict(id=row['id'],reference=row['reference'],start=e['start'],end=e['end']))
    store=EntityStore(OUT/'reference_terms.db')
    for d in store.list_entities(include_disabled=True):store.delete_entity(d.entity_id)
    for word in sorted(vocabulary):
        d=store.upsert_entity(word,entity_type='TERM',domain='medical',normalization_policy='normalize',
            source='Reference-derived entity with reviewed phrase boundaries, before ASR')
        if re.fullmatch(r'[\u3400-\u9fff]{3,16}',word):
            store.set_spelling_context(d.entity_id,keywords={w:1.0 for w in KEYWORDS if w not in word},minimum_score=1,
                source='Fixed generic medical context; not a per-record answer')
    data['phrase_review']='Whole procedure/device phrases reviewed in selected references before decoding; generic fragments excluded'
    write_json(OUT/'selection.json',data);write_json(OUT/'reference_entities.json',list(vocabulary.values()))
    write_json(OUT/'frozen_inputs.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'selection.json',OUT/'reference_entities.json',OUT/'reference_terms.db']})
    print('REVIEWED AND FROZEN',len(data['samples']),'recordings',len(vocabulary),'names',sum(len(r['entities']) for r in data['samples']),'mentions',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='review':review_prepared()
    elif a.stage=='asr':decode(SimpleNamespace(output=OUT,asr_url='http://127.0.0.1:8766'))
    else:evaluate(a.stage)
