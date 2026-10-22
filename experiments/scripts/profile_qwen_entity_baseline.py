"""Actual Qwen primary-service timing and no-target control collection."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s] if p.exists() else []
def append(p,r):
    with p.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')


async def main(out,kind,limit,realtime):
    inputs=rows(out/'inputs.jsonl')
    if kind=='control':selected=[r for r in inputs if r['split']=='control']
    else:
        pool=[r for r in inputs if r['split']=='test'];groups={}
        for r in sorted(pool,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()):groups.setdefault(r['domain'],r)
        selected=list(groups.values());already={r['id'] for r in selected}
        selected += [r for r in sorted(pool,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()) if r['id'] not in already]
    selected=selected[:limit]
    destination=out/('qwen_realtime_profile.jsonl' if realtime else 'qwen_live_profile.jsonl')
    done={r['id'] for r in rows(destination)}
    cached={r['id'] for r in rows(out/'qwen_primary_streams.jsonl')}
    async with httpx.AsyncClient(timeout=90,trust_env=False) as client:
        health=(await client.get('http://127.0.0.1:8766/health')).raise_for_status().json()
        (out/'qwen_health_live_profile.json').write_text(json.dumps(health,ensure_ascii=False,indent=2)+'\n')
        for n,s in enumerate(selected):
            if s['id'] in done:continue
            pcm=Path(s['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==s['pcm_sha256']
            started=time.perf_counter();http_seconds=0.;partials=[];chunks=[];sid=None
            async def call(endpoint,data=b'',params=None):
                nonlocal http_seconds
                q=dict(params or {})
                if sid:q['session_id']=sid
                before=time.perf_counter();v=(await client.post('http://127.0.0.1:8766'+endpoint,params=q,content=data)).raise_for_status().json()
                elapsed=time.perf_counter()-before;http_seconds+=elapsed;return v,elapsed
            try:
                ready,_=await call('/stream/start',params=dict(language='Chinese',chunk_seconds=2.))
                assert not ready.get('context_applied');sid=ready['session_id'];audio_started=time.perf_counter()
                for offset in range(0,len(pcm),10240):
                    part=pcm[offset:offset+10240];position=(offset+len(part))/64000
                    if realtime:await asyncio.sleep(max(0,audio_started+position-time.perf_counter()))
                    payload,elapsed=await call('/stream/chunk',data=part)
                    partials.append(dict(input_audio_seconds=position,text=payload.get('text',''),
                        wall_seconds=time.perf_counter()-audio_started,http_seconds=elapsed))
                    chunks.append(dict(data_bytes=len(part),payload=payload,http_seconds=elapsed))
                before_finish=time.perf_counter();final,_=await call('/stream/finish');sid=None
                record=dict(id=s['id'],split=s['split'],domain=s['domain'],pcm_sha256=s['pcm_sha256'],
                    audio_seconds=s['duration_seconds'],text=final.get('text',''),partials=partials,
                    compute_seconds_total=http_seconds,finish_seconds=time.perf_counter()-before_finish,
                    wall_seconds=time.perf_counter()-started,realtime_paced=realtime,
                    timing_scope='Actual HTTP service on RTX3090 GPU0, total HTTP wall including queue/transport; native ASR uses CPU2threads')
                append(destination,record);done.add(s['id'])
                if kind=='control' and s['id'] not in cached:
                    append(out/'qwen_primary_streams.jsonl',dict(id=s['id'],chunk_payloads=chunks,
                        final_payload=final,source=str(destination),timing_scope='Fresh actual no-context primary control stream'))
                    cached.add(s['id'])
                print('QWEN_PROFILE',kind,n+1,'/',len(selected),s['id'],round(http_seconds,3),flush=True)
            finally:
                if sid:
                    try:await client.post('http://127.0.0.1:8766/stream/cancel',params=dict(session_id=sid),timeout=3)
                    except Exception:pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--kind',choices=['profile','control'],default='profile');p.add_argument('--limit',type=int,default=64)
    p.add_argument('--realtime',action='store_true');a=p.parse_args();asyncio.run(main(a.run_dir.resolve(),a.kind,a.limit,a.realtime))
