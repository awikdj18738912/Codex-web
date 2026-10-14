"""Serial fresh full-Web old/new sessions, no scoring labels at inference."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlencode

import websockets

OUT=Path(__file__).resolve().parent

async def main(split,limit=None):
    worker=json.loads((OUT/'worker.json').read_text());port=worker['port']
    plan=json.loads((OUT/'frozen_test_plan.json').read_text())
    grouped={}
    for s in plan['samples']:
        if s['split']==split:grouped.setdefault(s['stratum'],[]).append(s)
    for rows in grouped.values():rows.sort(key=lambda s:s['selection_hash'])
    keys=sorted(grouped);samples=[grouped[k][i] for i in range(max(map(len,grouped.values()))) for k in keys if i<len(grouped[k])]
    policy_hash=hashlib.sha256((OUT/'candidate.py').read_bytes()).hexdigest()
    protocol={'split':split,'sample_order':[s['id'] for s in samples],
        'candidate_sha256':policy_hash,'source_hashes':worker['source_hashes'],
        'settings':{'ASR_chunk_seconds':2,'transport_bytes':64000,'max_outstanding':4,'serial_samples':True,'artificial_delay':0,'settle_before_next_sample':True},
        'timing_scope':'Local WS connect to full final; fresh ASR and actual refiner. Excludes file reads, decode, browser/proxy.',
        'gold_at_inference':False}
    pp=OUT/f'{split}_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    dest=OUT/f'{split}_results.jsonl'
    existing=[json.loads(l) for l in dest.open()] if dest.exists() else []
    done={(r['id'],r['arm']) for r in existing};assert len(done)==len(existing)
    started_run=time.perf_counter()
    for index,s in enumerate(samples):
        if limit and index>=limit:break
        data=Path(s['pcm_path']).read_bytes();assert hashlib.sha256(data).hexdigest()==s['pcm_sha256']
        order=('old','new') if int(s['selection_hash'][:2],16)%2 else ('new','old')
        for arm in order:
            if (s['id'],arm) in done:continue
            q=urlencode(dict(mode='streaming',language='Chinese',domain='general',asr_chunk_seconds=2,
                entity_asr_context=0,entity_audio_repair=1,eval_library=s['library_key'],eval_policy=arm))
            started=time.perf_counter();acked=0;condition=asyncio.Condition();final=None;counts={};updates=[];max_pending=0
            async with websockets.connect(f'ws://127.0.0.1:{port}/ws/stream?{q}',max_size=32*1024*1024,compression=None) as ws:
                ready=json.loads(await asyncio.wait_for(ws.recv(),90));assert ready['event']=='ready'
                assert ready['entity_audio_repair_requested'] and not ready['entity_asr_context_requested']
                assert ('incremental' in ready['entity_audio_repair'])==(arm=='new')
                prepared=time.perf_counter()
                async def receive():
                    nonlocal acked,final
                    while True:
                        p=json.loads(await asyncio.wait_for(ws.recv(),240));event=p.get('event');counts[event]=counts.get(event,0)+1
                        if event=='error':raise RuntimeError(p)
                        if event=='chunk_ack':
                            assert p['chunk_index']==acked+1
                            async with condition:acked+=1;condition.notify_all()
                        if event in {'transcript','partial','refined'}:
                            updates.append(dict(event=event,elapsed_seconds=time.perf_counter()-started,
                                acked_frames=acked,raw_text=p.get('raw_text'),clean_text=p.get('clean_text')))
                        if event=='final':final=p;return
                reader=asyncio.create_task(receive())
                async def wait_ack(target):
                    async def waiting():
                        async with condition:await condition.wait_for(lambda:acked>=target)
                    waiter=asyncio.create_task(waiting())
                    completed,_=await asyncio.wait((waiter,reader),timeout=240,return_when=asyncio.FIRST_COMPLETED)
                    if reader in completed:
                        waiter.cancel();await asyncio.gather(waiter,return_exceptions=True);await reader
                        raise RuntimeError('premature final')
                    if waiter not in completed:waiter.cancel();raise TimeoutError('ACK timeout')
                    await waiter
                try:
                    sending=time.perf_counter();frames=(len(data)+63999)//64000
                    for n,offset in enumerate(range(0,len(data),64000),1):
                        await wait_ack(max(0,n-4));await ws.send(data[offset:offset+64000]);max_pending=max(max_pending,n-acked)
                    await wait_ack(frames);audio_end=time.perf_counter()
                    await ws.send(json.dumps(dict(event='finish')));await reader;ended=time.perf_counter()
                finally:
                    if not reader.done():reader.cancel();await asyncio.gather(reader,return_exceptions=True)
            assert final and max_pending<=4 and acked==frames
            def quiescent():
                return json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/evaluation/quiescence',timeout=3))['pending_tasks']==0
            cleanup_begin=time.perf_counter()
            while not await asyncio.to_thread(quiescent):
                if time.perf_counter()-cleanup_begin>30:raise RuntimeError('late ASR cleanup did not settle')
                await asyncio.sleep(.02)
            settled=time.perf_counter()
            row=dict(id=s['id'],arm=arm,split=split,stratum=s['stratum'],corpus=s['corpus'],library_key=s['library_key'],
                audio_seconds=s['duration_seconds'],pcm_sha256=s['pcm_sha256'],trace_id=ready['trace_id'],ready=ready,
                timing=dict(total_seconds=ended-started,settled_total_seconds=settled-started,
                    post_return_cleanup_seconds=settled-ended,setup_seconds=prepared-started,stream_seconds=audio_end-sending,tail_seconds=ended-audio_end),
                transport=dict(frames=frames,acks=acked,max_pending=max_pending),event_counts=counts,updates=updates,result=final)
            with dest.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            done.add((s['id'],arm))
            progress=dict(split=split,sessions=len(done),expected_sessions=len(samples)*2,
                pairs=sum(all((s['id'],a) in done for a in ('old','new')) for s in samples),
                last_id=s['id'],last_arm=arm,last_seconds=round(ended-started,3),elapsed_seconds=round(time.perf_counter()-started_run,2))
            (OUT/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
            print('PROGRESS',json.dumps(progress,ensure_ascii=False),flush=True)
    print('COMPLETE',len(done),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','confirmation'],default='development');p.add_argument('--limit',type=int)
    a=p.parse_args();asyncio.run(main(a.split,a.limit))
