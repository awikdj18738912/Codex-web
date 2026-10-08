"""Disjoint public MMedFD eval dialogs, with frozen train-only terminology.

The dataset's audio is privacy-preserving TTS re-encoding; do not describe it
as original human speech. It is used for evaluation, not vocabulary training.
"""
import hashlib
import io
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace
import httpx
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf
from evaluate_entity_library import ROOT,read_jsonl,write_json,append_jsonl,audio_bytes
from evaluate_medical_entities import lexicon_occurrences

OUT=ROOT/'results/entity_context_20261006'


def prepare():
    src=OUT/'source/mmedfd_user_eval.parquet'
    rows=pq.read_table(src).to_pylist()
    vocabulary=[v['canonical_text'] for v in json.loads((OUT/'expanded_vocabulary.json').read_text())]
    seen=set();pool=[]
    for i,r in enumerate(rows):
        if r['chat_id'] in seen or not 2<=r['duration']<=30:continue
        seen.add(r['chat_id'])
        pool.append((i,r))
    random.Random(20261006).shuffle(pool)
    splits={'mmed_dev':pool[:24],'mmed_test':pool[24:88]}
    audio_dir=OUT/'mmedfd_audio';audio_dir.mkdir(exist_ok=True)
    manifest=dict(seed=20261006,dataset='YDMYy/MMedFD User/eval public subset',
                  dataset_revision='fa0df22392618f127ed4de21ac9dba19d6ee9977',
                  source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                  audio_role='Public TTS re-encoding of deidentified real-dialog content, not original human audio',
                  vocabulary_source='Frozen CMeEE + MultiMed train-only pack',splits={})
    for split,selected in splits.items():
        samples=[]
        for i,r in selected:
            wave,rate=sf.read(io.BytesIO(r['audio']['bytes']),dtype='float32')
            if wave.ndim>1:wave=wave.mean(axis=1)
            if rate!=16000:
                d=__import__('math').gcd(rate,16000);wave=resample_poly(wave,16000//d,rate//d).astype('float32')
            record_id=f'MMedFD_user_{i:04d}';path=audio_dir/(record_id+'.wav')
            sf.write(path,wave,16000,subtype='FLOAT')
            samples.append(dict(id=record_id,speaker='public_tts_identity_unknown',chat_id=r['chat_id'],
                                audio_path=str(path),reference=r['text'],entities=lexicon_occurrences(r['text'],vocabulary),
                                group='medical',dataset='MMedFD User public TTS eval',actual_duration_seconds=len(wave)/16000))
        manifest['splits'][split]=samples
    path=OUT/'mmedfd_selection.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Frozen selection changed')
    write_json(path,manifest)
    print({s:len(v) for s,v in manifest['splits'].items()},flush=True)


def decode(split):
    samples=json.loads((OUT/'mmedfd_selection.json').read_text())['splits'][split]
    dest=OUT/f'{split}_asr.jsonl';complete={(r['id'],r['variant']) for r in read_jsonl(dest)}
    with httpx.Client(base_url='http://127.0.0.1:8766',timeout=180) as client:
        for i,row in enumerate(samples):
            for variant,snr in [('clean',None),('noise10',10.0)]:
                if (row['id'],variant) in complete:continue
                pcm,duration=audio_bytes(row,'clean' if snr is None else 'noise',SimpleNamespace(seed=20261006,noise_snr=snr))
                started=time.perf_counter();sid=client.post('/stream/start',params=dict(language='Chinese',chunk_seconds=0.5)).raise_for_status().json()['session_id']
                chunks=[]
                try:
                    for offset in range(0,len(pcm),64000):
                        chunks.append(client.post('/stream/chunk',params=dict(session_id=sid),content=pcm[offset:offset+64000]).raise_for_status().json())
                    final=client.post('/stream/finish',params=dict(session_id=sid)).raise_for_status().json()
                except BaseException:
                    client.post('/stream/cancel',params=dict(session_id=sid));raise
                append_jsonl(dest,dict(**row,variant=variant,noise_snr_db=snr,duration_seconds=duration,
                                      asr_elapsed_seconds=time.perf_counter()-started,pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                                      raw_text=final['text'],chunk_payloads=chunks,final_payload=final))
                print(i+1,'/',len(samples),variant,row['id'],final['text'],flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','decode']);p.add_argument('--split',choices=['mmed_dev','mmed_test'],default='mmed_dev');a=p.parse_args()
    prepare() if a.stage=='prepare' else decode(a.split)
