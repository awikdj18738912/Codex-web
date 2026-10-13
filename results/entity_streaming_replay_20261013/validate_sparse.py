"""Real ASR old/new regression on three existing referenced long recordings."""
import asyncio,hashlib,importlib.util,json,sys,time
from pathlib import Path
import httpx
import soundfile as sf
from rapidfuzz.distance import Levenshtein
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).parent
from system.entity_store import EntityStore
from system.entity_audio_repair import EntityAudioRepair, normalized
spec=importlib.util.spec_from_file_location('system._stream_sparse_old',OUT/'before_entity_audio_repair.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
DATA=ROOT/'results/entity_local_replay_20261012/v2/sparse_development'
inputs=[json.loads(l) for l in (DATA/'inputs.jsonl').open()];refs=json.loads((DATA/'references.json').read_text())
async def main():
 async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
  for index,row in enumerate(inputs):
   pcm=Path(row['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
   definitions=EntityStore(row['library']).list_entities()
   for arm in (['old','new'] if index%2==0 else ['new','old']):
    timings=[]
    async def request(endpoint,session_id=None,data=b'',params=None):
     query=dict(params or {})
     if session_id:query['session_id']=session_id
     r=(await client.post('http://127.0.0.1:8766'+endpoint,params=query,content=data));r.raise_for_status();return r.json()
    guard=(old.EntityAudioRepair if arm=='old' else EntityAudioRepair)(request,definitions)
    started=time.perf_counter();ready=await guard.request('/stream/start',params={'language':'Chinese','chunk_seconds':2.0})
    for offset in range(0,len(pcm),64000):
     partial=await guard.request('/stream/chunk',ready['session_id'],pcm[offset:offset+64000])
     await asyncio.sleep(.08)
    chunk_end=time.perf_counter();final=await guard.request('/stream/finish',ready['session_id']);end=time.perf_counter()
    ref=refs[row['id']]['reference'];text=final['text']
    result={'id':row['id'],'arm':arm,'total_seconds':round(end-started,3),'chunk_seconds':round(chunk_end-started,3),'tail_seconds':round(end-chunk_end,3),'character_errors':Levenshtein.distance(normalized(ref),normalized(text)),'reference_chars':len(normalized(ref)),'text':text,'repair':final['entity_audio_repair'],'pcm_sha256':row['pcm_sha256']}
    with (OUT/'sparse_results.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    print('SPARSE',row['id'],arm,result['total_seconds'],result['tail_seconds'],result['character_errors'],result['repair'].get('streaming_cache_used'),flush=True)
asyncio.run(main())
