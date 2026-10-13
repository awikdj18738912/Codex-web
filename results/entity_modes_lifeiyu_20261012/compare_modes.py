"""Same real audio, deployed Web modes, fixed PCM and three interleaved rounds."""
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
import urllib.request

import numpy as np
import soundfile as sf
import websockets

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from system.entity_store import EntityStore
from system.entity_asr_context import build_entity_asr_context
OUT=Path(__file__).parent
SOURCE=ROOT/'测试音频3.mp3'
DB=Path('/home/aim0/桌面/tan/ASR/AgenticASR_entity_e5bedb3/data/entities.db')
store=EntityStore(DB)
EMPTY='__entity_modes_no_library_20261012__'
# Baseline uses a temporary service started without --entity-db.
definitions=store.list_entities(domain='general')
context=build_entity_asr_context(definitions,'general')
assert '厉飞宇' in context.terms
wave,rate=sf.read(SOURCE,dtype='float32',always_2d=True)
mono=wave.mean(axis=1,dtype=np.float32)
if rate!=16000:
 ratio=rate/16000
 positions=np.arange(math.floor(len(mono)/ratio),dtype=np.float64)*ratio
 left=np.floor(positions).astype(np.int64);right=np.minimum(left+1,len(mono)-1)
 mono=(mono[left]*(1-(positions-left))+mono[right]*(positions-left)).astype(np.float32)
pcm=mono.astype('<f4').tobytes()
assert hashlib.sha256(pcm).hexdigest()==json.loads((ROOT/'results/entity_prose_projection_20261012/actual_audio_rerun/plan.json').read_text())['pcm_sha256']
ARMS={
 'no_library':{'label':'不使用登记词库','domain':'general','entity_asr_context':0,'entity_audio_repair':0},
 'post_only':{'label':'仅实体替换（含新正文确认）','domain':'general','entity_asr_context':0,'entity_audio_repair':0},
 'context':{'label':'词库辅助识别＋实体替换','domain':'general','entity_asr_context':1,'entity_audio_repair':0},
 'audio':{'label':'实体原音复核＋实体替换','domain':'general','entity_asr_context':0,'entity_audio_repair':1},
}
ORDERS=[['no_library','post_only','context','audio'],['audio','context','post_only','no_library'],['post_only','no_library','audio','context']]
health=json.load(urllib.request.urlopen('http://127.0.0.1:8081/health',timeout=10))
plan={'audio_path':str(SOURCE),'audio_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'pcm_sha256':hashlib.sha256(pcm).hexdigest(),'duration_seconds':len(mono)/16000,'entity_db_sha256':hashlib.sha256(DB.read_bytes()).hexdigest(),
 'entity_definitions':[{'id':d.entity_id,'canonical':d.canonical_text,'type':d.entity_type,'policy':d.normalization_policy,'aliases':d.aliases} for d in definitions],
 'prompt_terms':context.terms,'settings':{'mode':'streaming','language':'Chinese','asr_chunk_seconds':2,'transport_seconds':1},
 'arms':ARMS,'orders':ORDERS,'health':health,'evaluation_target':'User-configured canonical 厉飞宇 at 在下厉飞羽 source occurrence',
 'scope':'One user-selected recording; no full gold transcript, no CER or population accuracy claim. Same PCM for every new real Web/ASR/Refiner session. Baseline temporary service started without entity database; production database untouched.'}
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')

def excerpt(text):
 for key in ('在下厉飞','厉飞'):
  position=text.find(key)
  if position>=0:return text[max(0,position-15):position+30]
 return None

async def run(round_no,arm):
 folder=OUT/f'round_{round_no}'/arm;folder.mkdir(parents=True,exist_ok=True)
 args={'language':'Chinese','mode':'streaming','asr_chunk_seconds':2,**{k:v for k,v in ARMS[arm].items() if k!='label'}}
 events=(folder/'events.jsonl').open('w');counts=Counter();raw='';repair_snapshots=[];started=time.perf_counter();last_report=started
 port=18081 if arm=='no_library' else 8081
 async with websockets.connect(f'ws://127.0.0.1:{port}/ws/stream?'+urlencode(args),max_size=32*1024*1024,open_timeout=20) as ws:
  ready=json.loads(await asyncio.wait_for(ws.recv(),30));assert ready['event']=='ready',ready
  assert ready['entity_asr_context_requested']==bool(args['entity_asr_context'])
  assert ready['entity_audio_repair_requested']==bool(args['entity_audio_repair'])
  if args['entity_asr_context']:assert ready['entity_asr_context_applied']
  (folder/'ready.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n')
  async def receive_until(name):
   nonlocal raw,last_report
   while True:
    result=json.loads(await asyncio.wait_for(ws.recv(),240))
    events.write(json.dumps(result,ensure_ascii=False)+'\n');events.flush()
    counts[result.get('event','')]+=1
    if result.get('event')=='error':raise RuntimeError(result)
    if result.get('raw_text'):raw=result['raw_text']
    if result.get('entity_audio_repair'):repair_snapshots.append(result['entity_audio_repair'])
    if time.perf_counter()-last_report>20:
     print('PROGRESS',round_no,arm,counts['chunk_ack'],'/323',round(time.perf_counter()-started,2),excerpt(raw),flush=True);last_report=time.perf_counter()
    if result.get('event')==name:return result
  for offset in range(0,len(pcm),64000):
   await ws.send(pcm[offset:offset+64000]);await receive_until('chunk_ack')
  await ws.send(json.dumps({'event':'finish'}));final=await receive_until('final')
 events.close()
 (folder/'final.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
 text=final.get('clean_text','');repair=final.get('entity_audio_repair') or (repair_snapshots[-1] if repair_snapshots else None)
 result={'round':round_no,'arm':arm,'label':ARMS[arm]['label'],'trace_id':ready['trace_id'],
  'elapsed_seconds':round(time.perf_counter()-started,3),'asr_context_applied':final.get('entity_asr_context_applied'),
  'raw_excerpt':excerpt(final.get('raw_text','')),'clean_excerpt':excerpt(text),
  'raw_target_count':final.get('raw_text','').count('厉飞宇'),'raw_wrong_count':final.get('raw_text','').count('厉飞羽'),
  'clean_target_count':text.count('厉飞宇'),'clean_wrong_count':text.count('厉飞羽'),
  'target_corrected':text.count('厉飞宇')==1 and '厉飞羽' not in text,'keeps_zai_xia':'在下厉飞宇' in text,
  'normalizations':final.get('entity_normalizations',[]),'audio_repair':repair,'refiner_stats':final.get('refiner_session_stats'),
  'event_counts':dict(counts),'final_sha256':hashlib.sha256(text.encode()).hexdigest(),'raw_sha256':hashlib.sha256(final.get('raw_text','').encode()).hexdigest()}
 (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 with (OUT/'results.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
 print('COMPLETE',json.dumps({k:result[k] for k in ('round','arm','elapsed_seconds','raw_excerpt','clean_excerpt','target_corrected','keeps_zai_xia')},ensure_ascii=False),flush=True)
 return result

async def main():
 done={}
 if (OUT/'results.jsonl').exists():
  done={(r['round'],r['arm']):r for r in map(json.loads,(OUT/'results.jsonl').open())}
 for round_no,order in enumerate(ORDERS,1):
  for arm in order:
   if (round_no,arm) not in done:await run(round_no,arm)
 assert hashlib.sha256(DB.read_bytes()).hexdigest()==plan['entity_db_sha256'],'Formal dictionary changed during comparison'
 print('ALL_COMPLETE',flush=True)

asyncio.run(main())
