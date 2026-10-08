"""Paired live ASR of a one-term prebuilt glossary; no production mutation."""
from collections import Counter
import hashlib
import io
import json
from math import gcd
from pathlib import Path
from types import SimpleNamespace
import time
import httpx
from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,append_jsonl,audio_bytes,normalized
from system.entity_store import EntityStore
from system.entity_asr_context import build_entity_asr_context

OUT=ROOT/'results/entity_chinese_rules_20261008/narrow_asr_context'
TERM='腹股沟疝修补手术'


def prepare():
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen')
    rows=pq.read_table(ROOT/'results/medical_entity_20261003/source/chinese_train.parquet').to_pylist()
    selected=[]
    # Known positive and nearby same-illness/other-procedure controls. This
    # is a development pilot; the IDs are fixed before these context calls.
    for i in [181,136,157,149,259,308,365]:
        r=rows[i];rid=f'MultiMed_Chinese_train_{i:04d}';path=OUT/'audio'/(rid+'.wav');path.parent.mkdir(exist_ok=True)
        wave,rate=sf.read(io.BytesIO(r['audio']['bytes']),dtype='float32')
        if wave.ndim>1:wave=wave.mean(axis=1)
        if rate!=16000:
            f=gcd(rate,16000);wave=resample_poly(wave,16000//f,rate//f).astype('float32')
        sf.write(path,wave,16000,subtype='FLOAT')
        selected.append(dict(id=rid,source_row=i,audio_path=str(path),reference=r['text'],group='known_positive' if i==181 else 'control'))
    store=EntityStore(OUT/'hernia.db');d=store.upsert_entity(TERM,domain='hernia_surgery',entity_type='TERM',normalization_policy='normalize',source='Medical paired reference; narrow one-term development pilot')
    context=build_entity_asr_context((d,),selected_domain='hernia_surgery').text
    write_json(OUT/'protocol.json',dict(seed=20261010,noise_snr_db=5,samples=selected,term=TERM,context=context,
        conditions=['off_before','glossary','off_after'],role='Known development positive and controls, not independent acceptance',only_canonical_name=True,no_error_aliases=True,production_changes=False))
    print('FROZEN',len(selected),'originals, 42 real ASR sessions',flush=True)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text());done={(r['id'],r['variant'],r['condition']) for r in read_jsonl(OUT/'asr.jsonl')}
    args=SimpleNamespace(seed=protocol['seed'],noise_snr=protocol['noise_snr_db'])
    with httpx.Client(base_url='http://127.0.0.1:8766',timeout=180) as client:
        for row in protocol['samples']:
            for variant in ['clean','noise']:
                pcm,duration=audio_bytes(row,variant,args)
                for condition in protocol['conditions']:
                    if (row['id'],variant,condition) in done:continue
                    params=dict(language='Chinese',chunk_seconds=.5)
                    if condition=='glossary':params['context']=protocol['context']
                    started=time.perf_counter();start=client.post('/stream/start',params=params).raise_for_status().json();sid=start['session_id'];chunks=[]
                    try:
                        for offset in range(0,len(pcm),64000):chunks.append(client.post('/stream/chunk',params=dict(session_id=sid),content=pcm[offset:offset+64000]).raise_for_status().json())
                        final=client.post('/stream/finish',params=dict(session_id=sid)).raise_for_status().json()
                    except BaseException:
                        client.post('/stream/cancel',params=dict(session_id=sid));raise
                    append_jsonl(OUT/'asr.jsonl',dict(**row,variant=variant,condition=condition,start_payload=start,raw_text=final['text'],chunk_payloads=chunks,final_payload=final,pcm_sha256=hashlib.sha256(pcm).hexdigest(),duration_seconds=duration,elapsed_seconds=time.perf_counter()-started))
                    print(row['id'],variant,condition,final['text'],flush=True)


def report():
    pairs={}
    for r in read_jsonl(OUT/'asr.jsonl'):pairs.setdefault((r['id'],r['variant']),{})[r['condition']]=r
    counts=Counter();changes=[]
    for key,p in pairs.items():
        if not all(c in p for c in ['off_before','glossary','off_after']):continue
        a=p['off_before']['raw_text'];b=p['glossary']['raw_text'];c=p['off_after']['raw_text'];reference=p['off_before']['reference']
        counts.update(pairs=1,off_unstable=int(a!=c),changed=int(a!=b),target_gain=b.count(TERM)-a.count(TERM),false_target=int(TERM not in reference and TERM in b))
        old=Levenshtein.distance(normalized(reference),normalized(a));new=Levenshtein.distance(normalized(reference),normalized(b));counts.update(improved=int(new<old),regressed=int(new>old),error_delta=new-old)
        if a!=b:changes.append(dict(id=key[0],variant=key[1],reference=reference,before=a,with_context=b,after_off=c,error_delta=new-old))
    write_json(OUT/'summary.json',dict(counts=dict(counts),changes=changes));print(json.dumps(dict(counts),ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','report']);a=p.parse_args()
    {'prepare':prepare,'run':run,'report':report}[a.stage]()
