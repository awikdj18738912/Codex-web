from __future__ import annotations
import asyncio, fcntl, hashlib, importlib.util, json, os, random, sys, threading, time, urllib.request
from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT/'source_snapshot'
DATA=Path('/home/aim0/data/datasets/ASR/refiner_original_boundary_200_filterfix_20261018')
TARGET=200; SEED=2026101817; BASE='http://127.0.0.1:8010/v1'; MODEL='refiner-teacher'; CONC=4
CTX=ContextVar('request_ctx',default={}); APPEND_LOCK=threading.Lock()
LOCK=(OUT/'worker.lock').open('w'); fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
def utc():return datetime.now(timezone.utc).isoformat()
def dump(path,obj):
 p=Path(path); t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def append(path,obj):
 with APPEND_LOCK:
  with open(path,'a',encoding='utf8') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def load_script(name,filename):
 spec=importlib.util.spec_from_file_location(name,ROOT/'pipeline/scripts'/filename);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
async def tagged(stage,cid,fn):
 token=CTX.set({'stage':stage,'id':cid})
 try:return await fn
 finally:CTX.reset(token)
async def main():
 random.seed(SEED)
 sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
 os.environ.update(VLLM_BASE_URL=BASE,VLLM_MODEL_NAME=MODEL,OPENROUTER_API_KEY='',OPENROUTER_MODEL='',CLEAN_TARGET_MODE='teacher',REFINER_TRAIN_ZH_ITN_LIBRARY=str(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so'))
 from configs import settings
 for k,sub in [('SEEDS_DIR','seeds'),('RAW_DIR','raw'),('PROCESSED_DIR','processed'),('FINAL_DIR','final')]:
  folder=DATA/sub;folder.mkdir(parents=True,exist_ok=True);setattr(settings,k,str(folder))
 from src.generators.oral_text_generator import OralTextGenerator
 from src.generators.clean_text_generator import CleanTextGenerator
 from src.generators.asr_simulator import ASRSimulator
 from src.pipeline.dedup import StreamingDeduplicator
 from src.services.llm_client import AsyncLLMClient
 assemble=load_script('frozen_assemble','04_assemble.py');finalize=load_script('frozen_finalize','05_finalize.py');exporter=load_script('frozen_export','export_sft.py');audit=load_script('frozen_audit','audit_dataset.py')
 m=json.loads((OUT/'source_manifest.json').read_text())
 for rel,expected in m['sha256'].items():
  actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
  if actual!=expected:raise RuntimeError('Frozen source hash mismatch: '+rel)
 for name,expected in m['seed_sha256'].items():
  if hashlib.sha256((DATA/'seeds'/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Seed hash mismatch: '+name)
 def model_info():
  with urllib.request.urlopen(BASE+'/models',timeout=10) as r:return json.load(r)
 info=model_info();model=next((x for x in info.get('data',[]) if x.get('id')==MODEL),None)
 if model is None:raise RuntimeError('Teacher model not served: '+MODEL)
 if '/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit' not in str(model.get('root','')):raise RuntimeError('Teacher model identity mismatch: '+str(model.get('root')))
 quotas={s:int(TARGET*w) for s,w in settings.SCENE_DISTRIBUTION.items()};left=TARGET-sum(quotas.values())
 for s in sorted(quotas,key=lambda x:TARGET*settings.SCENE_DISTRIBUTION[x]-quotas[x],reverse=True)[:left]:quotas[s]+=1
 jobs=[scene for scene,n in quotas.items() for _ in range(n)];random.Random(SEED).shuffle(jobs)
 api_path=OUT/'api_calls.jsonl';attempt_path=OUT/'attempts.jsonl'
 class Recorded(AsyncLLMClient):
  async def generate(self,**kwargs):
   ctx=CTX.get();row={'stage':ctx.get('stage','unknown'),'id':ctx.get('id'),'request':kwargs};started=time.monotonic()
   try:row['response']=await super().generate(**kwargs);return row['response']
   except Exception as e:row['error']=type(e).__name__+':'+str(e);raise
   finally:row.update(seconds=round(time.monotonic()-started,3),utc=utc());append(api_path,row)
 client=Recorded(base_url=BASE,model=MODEL,max_concurrent=CONC,timeout=600,max_retries=3)
 cleaner=CleanTextGenerator(client,target_mode='teacher');cleaner.qc.llm=client
 # Make cleaner temperature depend on each case's fixed seed rather than task scheduling.
 cleaner._sample_temp=lambda: random.Random(SEED+int(str(CTX.get().get('id','0')).rsplit('_',1)[-1] or 0)*7919).uniform(0.15,0.35)
 sem=asyncio.Semaphore(CONC); results=[]; done=0
 def write_progress(stage='running'):
  rows=[r for r,_ in results]
  p={'stage':stage,'target':TARGET,'completed':done,'quality_pass':sum(r.get('status')=='quality_pass' for r in rows),'scene_quotas':quotas,'candidate_status_counts':dict(Counter(r['status'] for r in rows)),'scene_attempted':dict(Counter(r['scene'] for r in rows)),'api_calls_so_far':sum(1 for _ in api_path.open()) if api_path.exists() else 0,'updated_utc':utc()}
  dump(OUT/'progress.json',p);return p
 write_progress()
 async def candidate(serial,scene):
  cid=f'original_{scene}_{serial:04d}';row={'serial':serial,'id':cid,'scene':scene};start=time.monotonic()
  try:
   oral=await tagged('oral',cid,OralTextGenerator(client,rng=random.Random(SEED+serial*1009)).generate_one(scene,serial))
   if not oral:row.update(status='oral_rejected',reason='OralTextGenerator returned None');return row,None
   oral['id']=cid;row['oral']=oral
   clean=await tagged('clean',cid,cleaner.generate_one(scene,oral))
   if not clean:row.update(status='clean_rejected',reason='CleanTextGenerator returned None');return row,None
   row['clean']=clean
   ok,reason,_=await cleaner.qc.validate_keywords_one(clean)
   if not ok:row.update(status='keyword_rejected',reason=reason);return row,None
   asr=await tagged('asr',cid,ASRSimulator(client,mode='llm_sim',rng=random.Random(SEED+serial*401+23)).simulate_one(scene,clean))
   if not asr:row.update(status='asr_rejected',reason='ASRSimulator returned empty pair');return row,None
   row['asr']=asr
   records,stats=assemble._assemble_from_asr(scene,[asr]);row['assembly_stats']=stats
   if not records:row.update(status='assembly_rejected',reason=str(stats));return row,None
   checked,passed,reason=await tagged('audit',cid,finalize.qc_one(records[0],client,sem));row['quality_audit']=checked.get('meta',{}).get('quality_audit')
   if not passed:row.update(status='audit_rejected',reason=reason);return row,None
   if exporter.convert_one(checked) is None:row.update(status='export_rejected',reason='exporter.convert_one returned None');return row,None
   row['status']='quality_pass';return row,checked
  except Exception as e:row.update(status='exception',reason=type(e).__name__+':'+str(e));return row,None
  finally:row['seconds']=round(time.monotonic()-start,3)
 q=asyncio.Queue()
 for i,scene in enumerate(jobs):q.put_nowait((i,scene))
 async def worker():
  nonlocal done
  while True:
   try:serial,scene=q.get_nowait()
   except asyncio.QueueEmpty:return
   res=await candidate(serial,scene);results.append(res);append(attempt_path,res[0]);done+=1;write_progress();q.task_done()
 await asyncio.gather(*(worker() for _ in range(CONC)))
 results.sort(key=lambda z:z[0]['serial']);dedup=StreamingDeduplicator(settings.DEDUP_THRESHOLD);retained=[];intent_counts=Counter();input_targets={};reject=Counter();decisions=[]
 for row,item in results:
  if item is None:reject[row['status']]+=1;decisions.append({'id':row['id'],'scene':row['scene'],'status':row['status'],'reason':row.get('reason','')});continue
  key=exporter.intent_group_key(item);ih=hashlib.sha256(item['input'].encode()).hexdigest();th=hashlib.sha256(item['output']['refined_text'].encode()).hexdigest()
  if ih in input_targets and input_targets[ih]!=th:status='input_target_conflict'
  elif intent_counts[key]>=3:status='intent_frequency_limit'
  elif not dedup.add(item):status='near_duplicate'
  else:status='retained';retained.append(item);intent_counts[key]+=1;input_targets[ih]=th
  reject[status]+=1;decisions.append({'id':row['id'],'scene':row['scene'],'status':status})
 with (OUT/'retained.jsonl').open('w',encoding='utf8') as f:
  for item in retained:f.write(json.dumps(item,ensure_ascii=False)+'\n')
 with (OUT/'retention.jsonl').open('w',encoding='utf8') as f:
  for item in decisions:f.write(json.dumps(item,ensure_ascii=False)+'\n')
 # Full structural acceptance audit over exactly the retained set.
 report,samples=audit.audit(OUT/'retained.jsonl',OUT/'audit_report.json',OUT/'review_sample.jsonl',min(200,len(retained)),SEED,min_records=len(retained))
 dump(OUT/'audit_report.json',report)
 with (OUT/'review_sample.jsonl').open('w',encoding='utf8') as f:
  for sample in samples:f.write(json.dumps(sample,ensure_ascii=False)+'\n')
 api_rows=[json.loads(x) for x in api_path.read_text().splitlines() if x.strip()] if api_path.exists() else []
 api_stages=Counter(x.get('stage','unknown') for x in api_rows)
 diffs=Counter('different' if x['input']!=x['output']['refined_text'] else 'same' for x in retained)
 summary={'target_denominator':TARGET,'candidate_count':len(results),'quality_pass_before_retention':sum(r.get('status')=='quality_pass' for r,_ in results),'retained':len(retained),'raw_retained_rate':len(retained)/TARGET,'reject_reasons':dict(reject),'candidate_stage_counts':dict(Counter(r['status'] for r,_ in results)),'retained_by_scene':dict(Counter(x['meta']['scene'] for x in retained)),'candidate_by_scene':dict(Counter(r['scene'] for r,_ in results)),'scene_quotas':quotas,'seed':SEED,'seed_source':m['seed_source'],'seed_aggregate_sha256':m['seed_aggregate_sha256'],'teacher_model':model,'teacher_endpoint':BASE,'api_calls_total':len(api_rows),'api_calls_by_stage':dict(api_stages),'api_call_errors':sum('error' in x for x in api_rows),'retained_input_vs_output':{'same':diffs['same'],'different':diffs['different']},'target_origin_counts':dict(Counter((x.get('meta') or {}).get('target_origin','unknown') for x in retained)),'source_manifest_aggregate_sha256':m['aggregate_sha256'],'full_structural_audit':{'passed':report['passed'],'errors':report['errors'],'counts':report['counts'],'input_sha256':report['input_sha256']},'review_sample_records':len(samples),'training_performed':False,'completed_utc':utc()}
 dump(OUT/'summary.json',summary)
 progress={'stage':'complete' if report['passed'] else 'complete_audit_failed','target':TARGET,'completed':len(results),'quality_pass':summary['quality_pass_before_retention'],'retained':len(retained),'raw_retained_rate':len(retained)/TARGET,'candidate_status_counts':summary['candidate_stage_counts'],'retained_status_counts':summary['reject_reasons'],'scene_quotas':quotas,'api_calls':len(api_rows),'api_calls_by_stage':dict(api_stages),'audit_passed':report['passed'],'updated_utc':summary['completed_utc'],'worker_pid':os.getpid()}
 dump(OUT/'progress.json',progress)
 print(json.dumps(summary,ensure_ascii=False),flush=True)
 await client.client.close()
if __name__=='__main__':asyncio.run(main())
