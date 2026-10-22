"""Read actual deployed Web events for the previously used Li audio slice."""
import asyncio,hashlib,json,time,urllib.request,sys
from pathlib import Path
from urllib.parse import urlencode
import websockets
OUT=Path(__file__).parent
async def main():
 p=json.loads((OUT/'long_test_plan.json').read_text());s=next(v for v in p['samples'] if v['library_key']=='li_feiyu')
 data=Path(s['pcm_path']).read_bytes()
 if '--full' not in sys.argv:data=data[40*64000:72*64000]
 started=time.perf_counter();events=[]
 q=urlencode(dict(mode='streaming',language='Chinese',domain='general',asr_chunk_seconds=2,entity_audio_repair=1,entity_asr_context=0))
 async with websockets.connect(f'ws://127.0.0.1:8081/ws/stream?{q}',max_size=32*1024*1024,compression=None) as ws:
  ready=json.loads(await ws.recv());assert ready['event']=='ready'
  rule=ready['entity_audio_repair']['incremental'];assert not rule['whole_audio_replay_enabled'] and not rule['finish_starts_review']
  for i in range(0,len(data),64000):
   await ws.send(data[i:i+64000])
   while True:
    e=json.loads(await asyncio.wait_for(ws.recv(),90));events.append(dict(payload=e,elapsed_seconds=time.perf_counter()-started,uploaded_audio_seconds=(i+64000)/64000))
    if e['event']=='error':raise RuntimeError(e)
    if e['event']=='chunk_ack':break
   await asyncio.sleep(.08)
  audio_end=time.perf_counter();await ws.send(json.dumps(dict(event='finish')))
  while True:
   e=json.loads(await asyncio.wait_for(ws.recv(),90));events.append(dict(payload=e,elapsed_seconds=time.perf_counter()-started,uploaded_audio_seconds=len(data)/64000))
   if e['event']=='error':raise RuntimeError(e)
   if e['event']=='final':break
 final=e;repair=e['entity_audio_repair']
 assert '厉飞宇' in final['raw_text'] and '厉飞宇' in final['clean_text']
 assert any(c['observed']=='厉飞羽' and c['canonical']=='厉飞宇' for c in repair['changes'])
 assert repair['incremental']['full_audio_fallbacks']==0 and repair['replay_mode']=='streaming_local'
 first=next((dict(elapsed_seconds=x['elapsed_seconds'],uploaded_audio_seconds=x['uploaded_audio_seconds']) for x in events if x['payload']['event']=='transcript' and '厉飞宇' in (x['payload'].get('raw_text') or '')),None)
 report=dict(trace_id=ready['trace_id'],ready=ready,pcm_seconds=len(data)/64000,pcm_sha256=hashlib.sha256(data).hexdigest(),total_seconds=time.perf_counter()-started,tail_seconds=time.perf_counter()-audio_end,first_corrected_raw=first,final=final,events=events,timing_scope='Local deployed Web socket; one PCM-second per ACK, artificial .08s delay; excludes browser/proxy/decode.')
 (OUT/('live_web_full_smoke.json' if '--full' in sys.argv else 'live_web_smoke.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['trace_id','total_seconds','tail_seconds','first_corrected_raw']},ensure_ascii=False),flush=True)
 print('ENTITY',repair['checks'],repair['change_count'],repair['replay_audio_seconds'],flush=True)
asyncio.run(main())
