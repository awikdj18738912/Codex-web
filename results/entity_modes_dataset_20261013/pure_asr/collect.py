"""Fresh serial paired full-Web comparison; scorer references never loaded here."""
import argparse,asyncio,hashlib,json,sys,time
from pathlib import Path
from urllib.parse import urlencode
import websockets
OUT=Path(__file__).resolve().parent
ARMS={'pure_asr':{}}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def planned_order(samples):
 # Interleave libraries, including control recordings, before the long news tail.
 groups={}
 for s in samples:groups.setdefault(s['library_key'],[]).append(s)
 for rows in groups.values():rows.sort(key=lambda s:hashlib.sha256(('mode-speed-v1:'+s['id']).encode()).hexdigest())
 keys=sorted(groups);ordered=[]
 for index in range(max(map(len,groups.values()))):
  for key in keys:
   if index<len(groups[key]):ordered.append(groups[key][index])
 return ordered

async def main(limit=None):
 worker=json.loads((OUT/'worker.json').read_text());port=worker['port'];samples=planned_order([json.loads(l) for l in (OUT.parent/'inference_inputs.jsonl').open()])
 protocol_path=OUT/'protocol.json'
 protocol={'arms':ARMS,'input_sha256':digest(OUT.parent/'inference_inputs.jsonl'),'library_hashes':{},'sample_order':[s['id'] for s in samples],'source_hashes':worker['source_hashes'],'settings':{'mode':'streaming','domain':'general','language':'Chinese','asr_chunk_seconds':2,'transport_samples':16000,'outstanding_frames':4,'artificial_delay':0,'concurrent_samples':1},'accuracy_scope':'ASR text unchanged: no library/context/audio review/refiner/numeric cleanup. No references/gold read at inference. Comparison arms retained from previous serial paired run.','timing_scope':'Local raw-ASR-only WebSocket relay connect-to-final, identical HTTP adapter/upload chunks. Excludes disk read/decode/browser/proxy. Supplemental run after original arms, not three-arm interleaving; no response cache.'}
 if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol,'protocol changed'
 else:protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
 destination=OUT/'results.jsonl';existing=[json.loads(l) for l in destination.open()] if destination.exists() else []
 done={(r['id'],r['arm']) for r in existing};assert len(done)==len(existing)
 started_run=time.perf_counter();new_count=0
 for sample_index,sample in enumerate(samples):
  if limit and sample_index>=limit:break
  pcm=Path(sample['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']
  arm_order=('pure_asr',)
  for arm in arm_order:
   if (sample['id'],arm) in done:continue
   query={'language':'Chinese','mode':'streaming','asr_chunk_seconds':2,**ARMS[arm]}
   counts={};max_outstanding=0;events=[];acknowledged=0;condition=asyncio.Condition();final=None
   started=time.perf_counter()
   async with websockets.connect(f'ws://127.0.0.1:{port}/ws/stream?'+urlencode(query),max_size=32*1024*1024,open_timeout=30,compression=None) as ws:
    ready=json.loads(await asyncio.wait_for(ws.recv(),90));assert ready['event']=='ready',ready
    assert ready['entity_asr_context_requested']==(arm=='context')
    assert ready['entity_audio_repair_requested']==(arm=='audio')
    assert not ready['entity_asr_context_applied'] and ready['entity_asr_context_term_count']==0
    assert not ready['postprocessing_enabled'] and not ready['refiner_enabled']
    ready_time=time.perf_counter();total_chunks=(len(pcm)+63999)//64000
    async def receiver():
     nonlocal acknowledged,final
     while True:
      r=json.loads(await asyncio.wait_for(ws.recv(),240));name=r.get('event','');counts[name]=counts.get(name,0)+1
      if name=='error':raise RuntimeError(r)
      if name=='chunk_ack':
       index=r['chunk_index'];assert index==acknowledged+1,(sample['id'],index,acknowledged)
       async with condition:acknowledged=index;condition.notify_all()
      if name=='final':final=r;return
    reader=asyncio.create_task(receiver())
    async def ack_until(target):
     async def wait():
      async with condition:await condition.wait_for(lambda:acknowledged>=target)
     waiter=asyncio.create_task(wait())
     completed,_=await asyncio.wait((waiter,reader),return_when=asyncio.FIRST_COMPLETED,timeout=240)
     if reader in completed:
      waiter.cancel();await asyncio.gather(waiter,return_exceptions=True)
      await reader
      raise RuntimeError('Unexpected final before finish')
     if waiter not in completed:
      waiter.cancel();await asyncio.gather(waiter,return_exceptions=True);raise TimeoutError('Audio ack timed out')
     await waiter
    try:
     first_send=time.perf_counter()
     for index,offset in enumerate(range(0,len(pcm),64000),1):
      await ack_until(max(0,index-4))
      await ws.send(pcm[offset:offset+64000]);max_outstanding=max(max_outstanding,index-acknowledged)
     await ack_until(total_chunks);audio_end=time.perf_counter()
     await ws.send(json.dumps({'event':'finish'}));await asyncio.wait_for(reader,240);ended=time.perf_counter()
    finally:
     if not reader.done():reader.cancel();await asyncio.gather(reader,return_exceptions=True)
   assert acknowledged==total_chunks and max_outstanding<=4 and final is not None
   assert final['raw_text']==final['clean_text']==final['asr_payload'].get('text','')
   assert final['session_refiner_call_count']==0
   result={'id':sample['id'],'arm':arm,'corpus':sample['corpus'],'library_key':sample['library_key'],'cohort':sample['cohort'],'topics':sample['topics'],'pcm_sha256':sample['pcm_sha256'],'audio_seconds':sample['duration_seconds'],'trace_id':ready['trace_id'],'ready':ready,'timing':{'total_seconds':ended-started,'session_setup_seconds':ready_time-started,'audio_stream_seconds':audio_end-first_send,'tail_seconds':ended-audio_end},'transport':{'frames':total_chunks,'acks':acknowledged,'max_outstanding':max_outstanding},'event_counts':counts,'result':final}
   with destination.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
   done.add((sample['id'],arm));new_count+=1
   progress={'completed_sessions':len(done),'expected_sessions':len(samples),'completed_pairs':sum(all((s['id'],a) in done for a in ARMS) for s in samples),'last_sample':sample['id'],'last_arm':arm,'last_seconds':round(ended-started,3),'this_process_seconds':round(time.perf_counter()-started_run,2),'new_sessions':new_count}
   (OUT/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
   if new_count<=24 or new_count%40==0:print('PROGRESS',json.dumps(progress,ensure_ascii=False),flush=True)
 print('COLLECT_COMPLETE',len(done),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args();asyncio.run(main(args.limit))
