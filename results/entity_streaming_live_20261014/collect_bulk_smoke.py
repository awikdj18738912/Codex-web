"""Actual Web/ASR regression for one PCM packet and offline buffering."""
import asyncio,hashlib,json,time,urllib.request
from pathlib import Path
from urllib.parse import urlencode
import websockets
OUT=Path(__file__).parent
async def main():
 w=json.loads((OUT/'worker.json').read_text());p=json.loads((OUT/'long_test_plan.json').read_text())
 s=next(v for v in p['samples'] if v['library_key']=='li_feiyu')
 pcm=Path(s['pcm_path']).read_bytes()[40*64000:72*64000]
 rows=[]
 for mode in ['streaming','offline']:
  for arm in ['old','new']:
   q=urlencode(dict(mode=mode,language='Chinese',domain='general',asr_chunk_seconds=2,entity_asr_context=0,entity_audio_repair=1,eval_library=s['library_key'],eval_policy=arm))
   began=time.perf_counter();events=[]
   async with websockets.connect(f"ws://127.0.0.1:{w['port']}/ws/stream?{q}",max_size=32*1024*1024,compression=None) as ws:
    ready=json.loads(await ws.recv());await ws.send(pcm)
    while True:
     e=json.loads(await ws.recv());events.append(e)
     if e['event']=='error':raise RuntimeError(e)
     if e['event']=='chunk_ack':break
    before_finish=time.perf_counter();await ws.send(json.dumps(dict(event='finish')))
    while True:
     e=json.loads(await ws.recv());events.append(e)
     if e['event']=='error':raise RuntimeError(e)
     if e['event']=='final':break
   end=time.perf_counter()
   for _ in range(1500):
    pending=await asyncio.to_thread(lambda:json.load(urllib.request.urlopen(f"http://127.0.0.1:{w['port']}/evaluation/quiescence",timeout=3))['pending_tasks'])
    if not pending:break
    await asyncio.sleep(.02)
   else:raise RuntimeError('cleanup not settled')
   row=dict(mode=mode,arm=arm,pcm_seconds=len(pcm)/64000,pcm_sha256=hashlib.sha256(pcm).hexdigest(),transport='one_binary_packet',total_seconds=end-began,tail_seconds=end-before_finish,trace_id=ready['trace_id'],result=e,event_counts={k:sum(x['event']==k for x in events) for k in {x['event'] for x in events}},target_in_raw='厉飞宇' in e.get('raw_text',''),target_in_clean='厉飞宇' in e.get('clean_text',''),source_hashes=w['source_hashes'])
   rows.append(row);(OUT/'bulk_smoke.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
   print(mode,arm,round(end-began,3),row['target_in_raw'],e['entity_audio_repair']['replay_audio_seconds'],flush=True)
asyncio.run(main())
