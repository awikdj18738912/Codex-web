"""Actual same-audio old/new Web sessions with fixed client pacing."""
import asyncio,hashlib,json,math,time
from pathlib import Path
import numpy as np
import soundfile as sf
import websockets
OUT=Path(__file__).parent;ROOT=OUT.parents[1]
wave,rate=sf.read(ROOT/'测试音频3.mp3',dtype='float32',always_2d=True);mono=wave.mean(axis=1,dtype=np.float32)
ratio=rate/16000
pos=np.arange(math.floor(len(mono)/ratio),dtype=np.float64)*ratio
left=np.floor(pos).astype(np.int64);right=np.minimum(left+1,len(mono)-1)
pcm=(mono[left]*(1-(pos-left))+mono[right]*(pos-left)).astype('<f4').tobytes()
plan={'pcm_sha256':hashlib.sha256(pcm).hexdigest(),'audio_seconds':len(pcm)/64000,'cases':[{'round':1,'gap_seconds':.08,'order':['old','new']},{'round':2,'gap_seconds':.08,'order':['new','old']},{'round':3,'gap_seconds':0,'order':['old','new']}],'settings':'Same 1s transport, ASR chunk interval2, general, context0, audio repair1. Only entity module variant changes.'}
(OUT/'web_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
async def run(n,gap,arm):
 folder=OUT/'web'/f'round_{n}'/arm;folder.mkdir(parents=True,exist_ok=True)
 events=(folder/'events.jsonl').open('w');start=time.perf_counter();last=start
 async with websockets.connect(f'ws://127.0.0.1:18187/ws/stream?mode=streaming&domain=general&entity_asr_context=0&entity_audio_repair=1&asr_chunk_seconds=2&repair_variant={arm}',max_size=32*1024*1024) as ws:
  ready=json.loads(await ws.recv());assert ready['event']=='ready'
  async def receive_until(name):
   nonlocal last
   while True:
    r=json.loads(await asyncio.wait_for(ws.recv(),180));events.write(json.dumps(r,ensure_ascii=False)+'\n');events.flush()
    if r.get('event')=='error':raise RuntimeError(r)
    if time.perf_counter()-last>20:print('PROGRESS',n,arm,round(time.perf_counter()-start,2),r.get('entity_audio_repair'),flush=True);last=time.perf_counter()
    if r.get('event')==name:return r
  for offset in range(0,len(pcm),64000):
   await ws.send(pcm[offset:offset+64000]);await receive_until('chunk_ack')
   if gap:await asyncio.sleep(gap)
  chunk_end=time.perf_counter();await ws.send(json.dumps({'event':'finish'}));final=await receive_until('final')
 events.close();ended=time.perf_counter()
 (folder/'final.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
 result={'round':n,'gap':gap,'arm':arm,'trace_id':ready['trace_id'],'total_seconds':round(ended-start,3),'chunk_phase_seconds':round(chunk_end-start,3),'tail_seconds':round(ended-chunk_end,3),'target_corrected':'在下厉飞宇' in final['clean_text'],'raw_target_corrected':'厉飞宇' in final['raw_text'],'raw_sha':hashlib.sha256(final['raw_text'].encode()).hexdigest(),'clean_sha':hashlib.sha256(final['clean_text'].encode()).hexdigest(),'entity_normalizations':final['entity_normalizations'],'repair':final['entity_audio_repair']}
 with (OUT/'web_results.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
 print('COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)
async def main():
 done={(r['round'],r['arm']) for r in map(json.loads,(OUT/'web_results.jsonl').open())} if (OUT/'web_results.jsonl').exists() else set()
 for case in plan['cases']:
  for arm in case['order']:
   if (case['round'],arm) not in done:await run(case['round'],case['gap_seconds'],arm)
asyncio.run(main())
