from __future__ import annotations
import asyncio, hashlib, importlib.util, json, os, random, sys, threading, time, fcntl
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
OUT=Path(__file__).resolve().parent
ROOT=OUT/'source_snapshot'
DATA=Path('/home/aim0/data/datasets/ASR/refiner_natural_200_20261017')
SEED=2026101720; TARGET=200; BASE='http://127.0.0.1:8010/v1'; MODEL='refiner-teacher'; CONC=4
LOCK=(OUT/'worker.lock').open('w'); fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
APPEND_LOCK=threading.Lock()
def now(): return datetime.now(timezone.utc).isoformat()
def dump(path,obj):
 p=Path(path); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n'); t.replace(p)
def append(path,obj):
 with APPEND_LOCK:
  with open(path,'a',encoding='utf8') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
def load_script(name,filename):
 spec=importlib.util.spec_from_file_location(name,ROOT/'pipeline/scripts'/filename); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
async def main():
 sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
 os.environ.update(VLLM_BASE_URL=BASE,VLLM_MODEL_NAME=MODEL,OPENROUTER_API_KEY='',OPENROUTER_MODEL='',CLEAN_TARGET_MODE='construction',REFINER_TRAIN_ZH_ITN_LIBRARY=str(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so'))
 from configs import settings
 for key,sub in [('SEEDS_DIR','seeds'),('RAW_DIR','raw'),('PROCESSED_DIR','processed'),('FINAL_DIR','final')]:
  p=DATA/sub;p.mkdir(parents=True,exist_ok=True);setattr(settings,key,str(p))
 from src.generators.oral_text_generator import OralTextGenerator
 from src.generators.clean_text_generator import CleanTextGenerator
 from src.generators.asr_simulator import ASRSimulator
 from src.pipeline.dedup import StreamingDeduplicator
 from src.services.llm_client import AsyncLLMClient
 asm=load_script('frozen_assemble','04_assemble.py'); fin=load_script('frozen_finalize','05_finalize.py'); exp=load_script('frozen_export','export_sft.py')
 # weights are precisely the currently configured nine-scene distribution.
 quotas={s:int(TARGET*w) for s,w in settings.SCENE_DISTRIBUTION.items()}; rem=TARGET-sum(quotas.values())
 for s in sorted(quotas,key=lambda x:TARGET*settings.SCENE_DISTRIBUTION[x]-quotas[x],reverse=True)[:rem]: quotas[s]+=1
 jobs=[s for s,n in quotas.items() for _ in range(n)]; random.Random(SEED).shuffle(jobs)
 api_path=OUT/'api_calls.jsonl'; api_lock=threading.Lock()
 class Recorded(AsyncLLMClient):
  async def generate(self,**kw):
   ctx=CTX.get(); t=time.monotonic(); row={'stage':ctx.get('stage'),'id':ctx.get('id'),'request':kw}
   try:
    result=await super().generate(**kw); row['response']=result; return result
   except Exception as e: row['error']=type(e).__name__+':'+str(e); raise
   finally:
    row['seconds']=round(time.monotonic()-t,3); row['utc']=now()
    with api_lock:
     with api_path.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 from contextvars import ContextVar
 global CTX; CTX=ContextVar('ctx',default={})
 async def tagged(stage,cid,fn):
  tok=CTX.set({'stage':stage,'id':cid})
  try:return await fn
  finally:CTX.reset(tok)
 client=Recorded(base_url=BASE,model=MODEL,max_concurrent=CONC,timeout=600,max_retries=2)
 clean_gen=CleanTextGenerator(client,target_mode='construction'); clean_gen.qc.llm=client
 sem=asyncio.Semaphore(CONC); prog={'stage':'running','target':TARGET,'completed':0,'accepted_before_dedup':0,'retained':0,'scene_quotas':quotas,'updated_utc':now()}
 dump(OUT/'progress.json',prog)
 async def one(i,scene):
  cid=f'natural_{scene}_{i:04d}'; row={'serial':i,'id':cid,'scene':scene}; t=time.monotonic()
  try:
   oral=await tagged('oral',cid,OralTextGenerator(client,rng=random.Random(SEED+i*1009)).generate_one(scene,i))
   if not oral:row['status']='oral_rejected';return row,None
   oral['id']=cid;row['oral']=oral
   cl=await tagged('clean',cid,clean_gen.generate_one(scene,oral))
   if not cl:row['status']='clean_rejected';return row,None
   row['clean']=cl
   ok,reason,_=await clean_gen.qc.validate_keywords_one(cl)
   if not ok:row.update(status='keyword_rejected',reason=reason);return row,None
   asr=await tagged('asr',cid,ASRSimulator(client,mode='llm_sim',rng=random.Random(SEED+i*401+23)).simulate_one(scene,cl))
   if not asr:row['status']='asr_rejected';return row,None
   row['asr']=asr
   records,stats=asm._assemble_from_asr(scene,[asr]);row['assembly_stats']=stats
   if not records:row['status']='assembly_rejected';return row,None
   checked,passed,reason=await tagged('audit',cid,fin.qc_one(records[0],client,sem));row['quality_audit']=checked.get('meta',{}).get('quality_audit')
   if not passed:row.update(status='audit_rejected',reason=reason);return row,None
   if exp.convert_one(checked) is None:row['status']='export_rejected';return row,None
   row['status']='quality_pass';return row,checked
  except Exception as e:row.update(status='exception',reason=type(e).__name__+':'+str(e));return row,None
  finally:row['seconds']=round(time.monotonic()-t,3)
 results=[]; done=0
 async def worker(queue):
  nonlocal done
  while True:
   x=await queue.get()
   if x is None:queue.task_done();return
   i,s=x;res=await one(i,s);results.append(res);done+=1
   append(OUT/'attempts.jsonl',res[0])
   p={'stage':'running','target':TARGET,'completed':done,'accepted_before_dedup':sum(1 for r,_ in results if r['status']=='quality_pass'),'scene_quotas':quotas,'scene_attempted':dict(Counter(r['scene'] for r,_ in results)),'updated_utc':now()};dump(OUT/'progress.json',p)
   queue.task_done()
 q=asyncio.Queue()
 for i,s in enumerate(jobs):q.put_nowait((i,s))
 workers=[asyncio.create_task(worker(q)) for _ in range(CONC)]
 await q.join()
 for _ in workers:q.put_nowait(None)
 await asyncio.gather(*workers)
 # Stable serial-order retention with production-compatible conflict, intent and streaming dedup checks.
 results.sort(key=lambda z:z[0]['serial']); dedup=StreamingDeduplicator(settings.DEDUP_THRESHOLD); kept=[]; intents=Counter(); inputs={}; rejects=Counter(); accepted=[]
 for row,item in results:
  if item is None:rejects[row['status']]+=1;continue
  accepted.append(item); key=exp.intent_group_key(item); ih=hashlib.sha256(item['input'].encode()).hexdigest(); th=hashlib.sha256(item['output']['refined_text'].encode()).hexdigest()
  if ih in inputs and inputs[ih]!=th:row['status']='input_target_conflict'
  elif intents[key]>=3:row['status']='intent_frequency_limit'
  elif not dedup.add(item):row['status']='near_duplicate'
  else:
   row['status']='retained';kept.append(item);intents[key]+=1;inputs[ih]=th
  rejects[row['status']]+=1
  append(OUT/'retention.jsonl',{'id':row['id'],'scene':row['scene'],'status':row['status']})
 with (OUT/'retained.jsonl').open('w',encoding='utf8') as f:
  for item in kept:f.write(json.dumps(item,ensure_ascii=False)+'\n')
 dump(OUT/'summary.json',{'target_denominator':TARGET,'candidate_count':len(results),'retained':len(kept),'pass_rate':len(kept)/TARGET,'quality_pass_before_retention':len(accepted),'reject_reasons':dict(rejects),'retained_by_scene':dict(Counter(x['meta']['scene'] for x in kept)),'candidate_by_scene':dict(Counter(x['scene'] for x,_ in results)),'scene_quotas':quotas,'seed':SEED,'total_api_calls':sum(1 for _ in api_path.open()) if api_path.exists() else 0,'completed_utc':now(),'training_performed':False})
 await client.client.close()
if __name__=='__main__':asyncio.run(main())
