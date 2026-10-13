"""Real audio replay via deployed Web, using the previous client settings."""
import asyncio
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import websockets

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent/'actual_audio_rerun'
OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'测试音频3.mp3'
audio,rate=sf.read(SOURCE,dtype='float32',always_2d=True)
mono=audio.mean(axis=1,dtype=np.float32)
# Linear interpolation; floor output length (the browser uses round).
if rate!=16000:
 ratio=rate/16000
 positions=np.arange(math.floor(len(mono)/ratio),dtype=np.float64)*ratio
 left=np.floor(positions).astype(np.int64)
 right=np.minimum(left+1,len(mono)-1)
 mono=(mono[left]*(1-(positions-left))+mono[right]*(positions-left)).astype(np.float32)
pcm=mono.astype('<f4').tobytes()
prior=[]
with (ROOT/'results/web/session.jsonl').open() as f:
 for line in f:
  r=json.loads(line)
  if r.get('trace_id')=='bb3740bf411e4b01835c264751d66157':prior.append(r)
assert len(prior)==1
before=prior[0]['output']
plan={'audio_path':str(SOURCE),'audio_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'duration_seconds':len(mono)/16000,'source_sample_rate':rate,'source_channels':audio.shape[1],
      'source_identity_evidence':'323 transport chunks and initial transcript must match previous recording prefix',
      'pcm_sha256':hashlib.sha256(pcm).hexdigest(),'previous_trace_id':prior[0]['trace_id'],
      'entity_asr_context':False,'entity_audio_repair':False,'asr_chunk_seconds':2,'transport_seconds':1,
      'new_ASR_and_refiner_requests':True}
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')

async def main():
 started=time.perf_counter();latest_raw='';checked=False;counts={}
 events=(OUT/'events.jsonl').open('w')
 async with websockets.connect('ws://127.0.0.1:8081/ws/stream?language=Chinese&mode=streaming&domain=general&entity_asr_context=0&entity_audio_repair=0&asr_chunk_seconds=2',max_size=32*1024*1024,open_timeout=20) as ws:
  ready=json.loads(await asyncio.wait_for(ws.recv(),30));assert ready['event']=='ready',ready
  (OUT/'ready.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n')
  async def read_until(event_type):
   nonlocal latest_raw,checked
   while True:
    result=json.loads(await asyncio.wait_for(ws.recv(),180))
    events.write(json.dumps(result,ensure_ascii=False)+'\n');events.flush()
    name=result.get('event','');counts[name]=counts.get(name,0)+1
    if name=='error':raise RuntimeError(result)
    if result.get('raw_text'):latest_raw=result['raw_text']
    if not checked and len(latest_raw)>=10:
     if not latest_raw.startswith('天南一个能打的都没有'):
      raise RuntimeError('Local audio does not match previous opening: '+latest_raw[:100])
     checked=True;print('SOURCE_CONFIRMED',latest_raw[:80],flush=True)
    if name==event_type:return result
  for index,offset in enumerate(range(0,len(pcm),64000),1):
   await ws.send(pcm[offset:offset+64000])
   await read_until('chunk_ack')
   if index%40==0:print('PROGRESS',index,math.ceil(len(pcm)/64000),'seconds',round(time.perf_counter()-started,1),'tail',latest_raw[-70:],flush=True)
  assert checked
  await ws.send(json.dumps({'event':'finish'}))
  final=await read_until('final')
 events.close()
 (OUT/'final.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
 def excerpt(text):
  index=text.find('厉飞')
  return text[max(0,index-22):index+35] if index>=0 else None
 summary={'elapsed_seconds':round(time.perf_counter()-started,3),'actual_audio_inference':True,
          'previous_raw_excerpt':excerpt(before['raw_text']),'previous_clean_excerpt':excerpt(before['clean_text']),
          'raw_excerpt':excerpt(final['raw_text']),'clean_excerpt':excerpt(final['clean_text']),
          'raw_text_identical_to_previous':final['raw_text']==before['raw_text'],
          'entity_normalizations':final.get('entity_normalizations'),
          'target_corrected':'厉飞宇' in final['clean_text'] and '厉飞羽' not in final['clean_text'],
          'event_counts':counts,'refiner_reject_reasons':final.get('refiner_reject_reasons'),
          'refiner_session_stats':final.get('refiner_session_stats')}
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print('FINAL',json.dumps(summary,ensure_ascii=False),flush=True)

asyncio.run(main())
