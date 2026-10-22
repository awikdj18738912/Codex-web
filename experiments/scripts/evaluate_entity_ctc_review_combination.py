"""Actual Qwen + CTC + existing audio-review quality run on frozen PCM.

Inference reads only inputs, PCM and entity libraries. Labels are loaded by a
separate report script. Audio packets are sent as fast as the ASR responds;
wall times here are compute/queue observations, not paced user latency.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts.evaluate_entity_ctc_fusion import definitions
from system.entity_audio_repair import EntityAudioRepair
from system.entity_ctc_sidecar import EntityCtcRuntime

ASR_URL='http://127.0.0.1:8766'


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []


class LiveAsr:
    def __init__(self,client):
        self.client=client
        self.active=set()
        self.primary_id=None
        self.primary_final=None

    async def __call__(self,endpoint,session_id=None,data=b'',params=None):
        arguments=dict(params or {})
        if session_id is not None:arguments['session_id']=session_id
        response=await self.client.post(ASR_URL+endpoint,params=arguments,content=data)
        response.raise_for_status()
        result=response.json()
        if endpoint=='/stream/start':
            self.active.add(result['session_id'])
            if not arguments.get('context'):
                self.primary_id=result['session_id']
        if endpoint=='/stream/finish' and session_id==self.primary_id:
            self.primary_final=str(result.get('text',''))
        if endpoint in {'/stream/finish','/stream/cancel'}:self.active.discard(session_id)
        return result

    async def close(self):
        for session_id in tuple(self.active):
            try:await self('/stream/cancel',session_id)
            except Exception:pass


async def evaluate(out,destination,split,limit,ctc_mode='on',shard_index=0,
                   num_shards=1,ids_file=None):
    if split not in {'short_test','control','all','test_all'}:raise ValueError(split)
    if ctc_mode not in {'off','on'}:raise ValueError(ctc_mode)
    inputs=rows(out/'inputs.jsonl')
    short=ROOT/'results/entity_short_finish_20261015_v1/evaluation/comparisons.jsonl'
    short_ids={r['id'] for r in rows(short)}
    selected=[r for r in inputs if (r['split']=='test' and
                  (split=='test_all' or r['id'] in short_ids and split in {'short_test','all'})) or
              (r['split']=='control' and split in {'control','all'})]
    if ids_file is not None:
        requested={line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()}
        unknown=requested-{r['id'] for r in selected}
        if unknown:raise ValueError('IDs outside selected split: '+str(sorted(unknown)[:10]))
        selected=[r for r in selected if r['id'] in requested]
    selected.sort(key=lambda r:(r['split']!='test',r['id']))
    if limit is not None:selected=selected[:limit]
    if num_shards<1 or not 0<=shard_index<num_shards:
        raise ValueError('invalid shard configuration')
    selected=selected[shard_index::num_shards]
    base_destination=destination
    if num_shards>1:
        destination=destination.with_name(destination.stem+f'_shard{shard_index}'+destination.suffix)
    destination.parent.mkdir(parents=True,exist_ok=True)
    existing=rows(destination)
    prior=rows(base_destination) if base_destination!=destination else []
    done={r['id'] for r in [*prior,*existing]}
    assert len(done)==len(prior)+len(existing)
    if any(r.get('entity_ctc_mode') != ctc_mode for r in [*prior,*existing]):
        raise ValueError('Existing output contains a different CTC mode')
    modeldir=out/'models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30'
    runtime=EntityCtcRuntime(modeldir) if ctc_mode=='on' else None
    defs_cache={}
    async with httpx.AsyncClient(timeout=90,trust_env=False) as client:
        health=(await client.get(ASR_URL+'/health')).raise_for_status().json()
        if health.get('ok') is not True:raise RuntimeError('Qwen ASR is unhealthy')
        for index,row in enumerate(selected,1):
            if row['id'] in done:continue
            pcm=Path(row['pcm_path']).read_bytes()
            if hashlib.sha256(pcm).hexdigest()!=row['pcm_sha256']:
                raise ValueError('PCM hash mismatch: '+row['id'])
            lib=row['library_path']
            if lib not in defs_cache:defs_cache[lib]=definitions(lib)
            defs=defs_cache[lib]
            ctc=await asyncio.to_thread(runtime.create_session,defs) if runtime is not None else None
            backend=LiveAsr(client)
            guard=EntityAudioRepair(backend,defs,ctc_session=ctc,ctc_mode=ctc_mode)
            began=time.perf_counter()
            try:
                start=await guard.request('/stream/start',params=dict(language='Chinese',chunk_seconds=2.))
                if start.get('context_applied'):
                    raise RuntimeError('Primary ASR unexpectedly applied context')
                sid=start['session_id']
                for offset in range(0,len(pcm),64000):
                    await guard.request('/stream/chunk',sid,pcm[offset:offset+64000])
                    await asyncio.sleep(0)
                # Accelerated replay can outrun the CPU encoder. Drain the
                # real-audio work before Qwen finish; no reference is consulted.
                if ctc is not None and ctc._futures:
                    await asyncio.wrap_future(ctc._futures[-1])
                result=await guard.request('/stream/finish',sid)
                snapshot=guard.snapshot()
                record=dict(id=row['id'],split=row['split'],domain=row['domain'],
                    pcm_sha256=row['pcm_sha256'],audio_seconds=row['duration_seconds'],
                    primary_text=backend.primary_final,
                    text=str(result.get('text','')),changes=snapshot.get('changes',[]),
                    entity_audio_repair=snapshot,
                    entity_ctc_mode=ctc_mode,real_primary_and_review=True,
                    elapsed_seconds=time.perf_counter()-began,
                    timing_scope='Accelerated actual Qwen HTTP plus CPU CTC/review; not a paced latency measurement')
                with destination.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                done.add(row['id'])
                print('COMBINATION',index,'/',len(selected),row['id'],
                      round(record['elapsed_seconds'],3),flush=True)
            finally:
                guard.close()
                await backend.close()
    print('COMBINATION_COMPLETE',len(selected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/entity_ctc_live_integration_20261016/combination_live_results.jsonl')
    p.add_argument('--split',choices=['short_test','control','all','test_all'],default='all')
    p.add_argument('--ctc-mode',choices=['off','on'],default='on')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--ids-file',type=Path,
                   help='optional newline-separated frozen IDs for primary-drift retries')
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    asyncio.run(evaluate(a.run_dir.resolve(),a.output.resolve(),a.split,a.limit,
                         a.ctc_mode,a.shard_index,a.num_shards,a.ids_file))
