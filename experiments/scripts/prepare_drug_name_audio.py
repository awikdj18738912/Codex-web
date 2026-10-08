"""Drug-name-focused probe on publicly released medical-assistant TTS audio.

The train-only dictionary is fixed before ASR. No misspelling alias comes
from the audio/reference. TTS results must remain separate from human speech.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import random
import re
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from evaluate_entity_library import ROOT,read_jsonl,write_json,append_jsonl,audio_bytes
from evaluate_medical_entities import lexicon_occurrences
from system.entity_store import EntityStore

OUT=ROOT/'results/entity_ranker_20261006/drug_names'
BASE=OUT.parent


def prepare():
    import pyarrow.parquet as pq
    OUT.mkdir(exist_ok=True)
    annotations=json.loads((ROOT/'results/medical_entity_20261003/source/CMeEE_train.json').read_text())
    words=Counter(e['entity'] for row in annotations for e in row['entities']
                  if e['type']=='dru' and re.fullmatch('[\u3400-\u9fff]{4,16}',e['entity']))
    store=EntityStore(OUT/'context_drug_names.db');items=[]
    keys=['药物','口服','服用','用药','治疗','剂量','毫克','注射','症状','使用','建议']
    for word,count in sorted(words.items()):
        d=store.upsert_entity(word,entity_type='TERM',domain='drug_names',normalization_policy='normalize',source='CMeEE official training drug entity annotation')
        profile={k:1.0 for k in keys if k not in word}
        store.set_spelling_context(d.entity_id,keywords=profile,minimum_score=1.0,source='Declared medicine context; full-pronunciation spelling, no test aliases')
        items.append(dict(canonical_text=word,train_occurrences=count))
    write_json(OUT/'vocabulary.json',items)
    rows=pq.read_table(BASE/'source/mmedfd_agent_eval.parquet').to_pylist()
    taken=set();pool=[]
    for index,row in enumerate(rows):
        if not 2<=row['duration']<=60 or row['chat_id'] in taken:continue
        entities=lexicon_occurrences(row['text'],list(words))
        if entities:
            taken.add(row['chat_id']);pool.append((index,row,entities))
    random.Random(20261006).shuffle(pool)
    pool=pool[:48];samples=[]
    audio_dir=OUT/'audio';audio_dir.mkdir(exist_ok=True)
    for index,row,entities in pool:
        wave,rate=sf.read(io.BytesIO(row['audio']['bytes']),dtype='float32')
        if wave.ndim>1:wave=wave.mean(axis=1)
        if rate!=16000:
            import math
            g=math.gcd(rate,16000);wave=resample_poly(wave,16000//g,rate//g).astype('float32')
        record_id=f'MMedFD_agent_{index:04d}';path=audio_dir/(record_id+'.wav');sf.write(path,wave,16000,subtype='FLOAT')
        samples.append(dict(id=record_id,speaker='public_tts_identity_unknown',chat_id=row['chat_id'],audio_path=str(path),
            reference=row['text'],entities=entities,group='drug_names',dataset='MMedFD Agent public TTS eval',actual_duration_seconds=len(wave)/16000))
    write_json(OUT/'selection.json',dict(seed=20261006,noise_snr_db=5.0,source='MMedFD Agent eval TTS, not original doctor speech',samples=samples,
         dictionary='All CMeEE train drug annotations with 4-16 CJK characters, including drug groups; no test-derived aliases',variants=['clean','noise']))
    print('dictionary',len(words),'recordings',len(samples),'expected drug-name mentions',sum(len(s['entities']) for s in samples),flush=True)


def decode():
    from evaluate_entity_library import decode as run_decode
    run_decode(SimpleNamespace(output=OUT,asr_url='http://127.0.0.1:8766',seed=20261006,noise_snr=5.0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','decode']);a=p.parse_args()
    prepare() if a.stage=='prepare' else decode()
