from __future__ import annotations
import asyncio,fcntl,hashlib,importlib.util,json,os,random,sys,threading,time,urllib.request
from collections import Counter
from contextvars import ContextVar
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT/'source_snapshot';DATA=Path('/home/aim0/data/datasets/ASR/refiner_original_boundary_200_20261017')
TARGET=200;SEED=2026101720;BASE='http://127.0.0.1:8010/v1';MODEL='refiner-teacher';CONC=4
CTX=ContextVar('request_ctx',default={});APPEND_LOCK=threading.Lock();LOCK=(OUT/'resume.lock').open('w');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
def utc():return datetime.now(timezone.utc).isoformat()
def dump(path,obj):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
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
 from src.pipeline.training_quality import validate_audit
 assemble=load_script('frozen_assemble_resume','04_assemble.py');finalize=load_script('frozen_finalize_resume','05_finalize.py');exporter=load_script('frozen_export_resume','export_sft.py');audit=load_script('frozen_audit_resume','audit_dataset.py')
 manifest=json.loads((OUT/'source_manifest.json').read_text())
 for rel,expected in manifest['sha256'].items():
  if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen source hash mismatch: '+rel)
 for name,expected in manifest['seed_sha256'].items():
  if hashlib.sha256((DATA/'seeds'/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Seed hash mismatch: '+name)
 quotas={s:int(TARGET*w) for s,w in settings.SCENE_DISTRIBUTION.items()};left=TARGET-sum(quotas.values())
 for s in sorted(quotas,key=lambda x:TARGET*settings.SCENE_DISTRIBUTION[x]-quotas[x],reverse=True)[:left]:quotas[s]+=1
 jobs=[scene for scene,n in quotas.items() for _ in range(n)];random.Random(SEED).shuffle(jobs)
 attempt_path=OUT/'attempts.jsonl';api_path=OUT/'api_calls.jsonl'
 all_rows=[json.loads(x) for x in attempt_path.read_text().splitlines() if x.strip()]
 serials=[r.get('serial') for r in all_rows]
 if len(all_rows)!=len(set(serials)):raise RuntimeError('Unsafe resume: duplicate existing candidate serial')
 if any(type(n)is not int or n<0 or n>=TARGET for n in serials):raise RuntimeError('Unsafe resume: serial outside fixed 0..199 plan')
 mapping_errors=[]
 for r in all_rows:
  n=r['serial'];expected_id=f'original_{jobs[n]}_{n:04d}'
  if r.get('scene')!=jobs[n] or r.get('id')!=expected_id:mapping_errors.append({'serial':n,'stored_scene':r.get('scene'),'expected_scene':jobs[n],'stored_id':r.get('id'),'expected_id':expected_id})
 if mapping_errors:raise RuntimeError('Unsafe resume: persisted candidate differs from fixed job table: '+json.dumps(mapping_errors,ensure_ascii=False))
 completed=set(serials);missing=[(n,scene) for n,scene in enumerate(jobs) if n not in completed]
 # Rebuild and validate every previously passing record offline before teacher calls.
 reconstructed={};rebuild_fail=[]
 for r in all_rows:
  if r.get('status')!='quality_pass':continue
  if not isinstance(r.get('asr'),dict) or not isinstance(r.get('quality_audit'),dict):rebuild_fail.append({'serial':r['serial'],'reason':'missing ASR or stored audit evidence'});continue
  recs,stats=assemble._assemble_from_asr(r['scene'],[r['asr']])
  if len(recs)!=1:rebuild_fail.append({'serial':r['serial'],'reason':'assemble count '+str(len(recs))});continue
  item=recs[0];meta=dict(item.get('meta') or {});meta['quality_audit']=r['quality_audit'];item={**item,'meta':meta}
  passed,reason=validate_audit(r['quality_audit'],item['input'],item['output']['refined_text'])
  if not passed:rebuild_fail.append({'serial':r['serial'],'reason':'stored audit invalid: '+str(reason)});continue
  try:converted=exporter.convert_one(item)
  except Exception as exc:rebuild_fail.append({'serial':r['serial'],'reason':'export exception '+repr(exc)});continue
  if converted is None:rebuild_fail.append({'serial':r['serial'],'reason':'exporter.convert_one returned None'});continue
  reconstructed[r['serial']]=item
 preflight={'existing_rows':len(all_rows),'unique_serials':len(completed),'existing_serials':sorted(completed),'missing_serials':sorted(n for n,_ in missing),'missing_count':len(missing),'old_quality_pass_count':sum(r.get('status')=='quality_pass' for r in all_rows),'reconstructed_quality_pass_count':len(reconstructed),'rebuild_failures':rebuild_fail,'job_mapping_errors':mapping_errors,'model_requests_made':0,'checked_utc':utc()}
 dump(OUT/'recovery_preflight.json',preflight)
 if rebuild_fail:raise RuntimeError('Cannot safely resume: prior quality_pass rebuild/export check failed; see recovery_preflight.json')
 if len(completed)!=72 or not set(range(63)).issubset(completed) or any(n in completed for n in (63,72,73)):
  # Preserve observed state while asserting the known interruption signature.
  if len(completed)!=72:raise RuntimeError('Unexpected prior attempt count; inspect recovery_preflight.json')
  if 63 in completed or 72 in completed or 73 in completed:raise RuntimeError('Unexpected known interruption serial set')
  if 74 not in completed:raise RuntimeError('Expected already completed concurrent serial 74 is absent')
 info=json.load(urllib.request.urlopen(BASE+'/models',timeout=10));model=next((x for x in info.get('data',[]) if x.get('id')==MODEL),None)
 if model is None or '/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit' not in str(model.get('root','')):raise RuntimeError('Teacher identity mismatch; no resume calls made')
 dump(OUT/'recovery_preflight.json',{**preflight,'model_identity':{k:model.get(k) for k in ('id','root','parent','max_model_len')},'model_requests_made':0,'ready_to_resume':True,'checked_utc':utc()})
 class Recorded(AsyncLLMClient):
  async def generate(self,**kwargs):
   ctx=CTX.get();row={'stage':ctx.get('stage','unknown'),'id':ctx.get('id'),'request':kwargs};t=time.monotonic()
   try:row['response']=await super().generate(**kwargs);return row['response']
   except Exception as e:row['error']=type(e).__name__+':'+str(e);raise
   finally:row.update(seconds=round(time.monotonic()-t,3),utc=utc());append(api_path,row)
 client=Recorded(base_url=BASE,model=MODEL,max_concurrent=CONC,timeout=600,max_retries=3)
 cleaner=CleanTextGenerator(client,target_mode='teacher');cleaner.qc.llm=client
 cleaner._sample_temp=lambda:random.Random(SEED+int(str(CTX.get().get('id','0')).rsplit('_',1)[-1] or 0)*7919).uniform(0.15,0.35)
 sem=asyncio.Semaphore(CONC);new_rows=[];done=len(all_rows)
 def write_progress(stage='resume_running'):
  rows=all_rows+new_rows;api_n=sum(1 for _ in api_path.open()) if api_path.exists() else 0
  dump(OUT/'progress.json',{'stage':stage,'target':TARGET,'completed':len(rows),'quality_pass':sum(r.get('status')=='quality_pass' for r in rows),'scene_quotas':quotas,'candidate_status_counts':dict(Counter(r['status'] for r in rows)),'scene_attempted':dict(Counter(r['scene'] for r in rows)),'already_completed_serials':len(completed),'newly_completed_this_resume':len(new_rows),'remaining_jobs':len(missing)-len(new_rows),'api_calls_so_far':api_n,'updated_utc':utc()})
 write_progress()
 async def candidate(serial,scene):
  cid=f'original_{scene}_{serial:04d}';row={'serial':serial,'id':cid,'scene':scene};start=time.monotonic()
  try:
   oral=await tagged('oral',cid,OralTextGenerator(client,rng=random.Random(SEED+serial*1009)).generate_one(scene,serial))
   if not oral:row.update(status='oral_rejected',reason='OralTextGenerator returned None');return row
   oral['id']=cid;row['oral']=oral
   clean=await tagged('clean',cid,cleaner.generate_one(scene,oral))
   if not clean:row.update(status='clean_rejected',reason='CleanTextGenerator returned None');return row
   row['clean']=clean
   ok,reason,_=await cleaner.qc.validate_keywords_one(clean)
   if not ok:row.update(status='keyword_rejected',reason=reason);return row
   asr=await tagged('asr',cid,ASRSimulator(client,mode='llm_sim',rng=random.Random(SEED+serial*401+23)).simulate_one(scene,clean))
   if not asr:row.update(status='asr_rejected',reason='ASRSimulator returned empty pair');return row
   row['asr']=asr;records,stats=assemble._assemble_from_asr(scene,[asr]);row['assembly_stats']=stats
   if not records:row.update(status='assembly_rejected',reason=str(stats));return row
   checked,passed,reason=await tagged('audit',cid,finalize.qc_one(records[0],client,sem));row['quality_audit']=checked.get('meta',{}).get('quality_audit')
   if not passed:row.update(status='audit_rejected',reason=reason);return row
   if exporter.convert_one(checked) is None:row.update(status='export_rejected',reason='exporter.convert_one returned None');return row
   row['status']='quality_pass';return row
  except Exception as e:row.update(status='exception',reason=type(e).__name__+':'+str(e));return row
  finally:row['seconds']=round(time.monotonic()-start,3)
 queue=asyncio.Queue()
 for job in missing:queue.put_nowait(job)
 async def worker():
  while True:
   try:n,scene=queue.get_nowait()
   except asyncio.QueueEmpty:return
   row=await candidate(n,scene);new_rows.append(row);append(attempt_path,row);write_progress();queue.task_done()
 await asyncio.gather(*(worker() for _ in range(CONC)))
 merged=all_rows+new_rows;merged.sort(key=lambda r:r['serial']);merged_serials=[r['serial'] for r in merged]
 if merged_serials!=list(range(TARGET)):raise RuntimeError('Final candidate serial verification failed: '+json.dumps({'count':len(merged_serials),'unique':len(set(merged_serials)),'missing':sorted(set(range(TARGET))-set(merged_serials)),'duplicates':[n for n,c in Counter(merged_serials).items() if c>1]}))
 if any(r['scene']!=jobs[r['serial']] or r['id']!=f"original_{jobs[r['serial']]}_{r['serial']:04d}" for r in merged):raise RuntimeError('Final job scene/id verification failed')
 quality_items=[];quality_item_by_serial={};rebuild_final_fail=[]
 for r in merged:
  if r.get('status')!='quality_pass':continue
  if r['serial'] in reconstructed:item=reconstructed[r['serial']]
  else:
   if not isinstance(r.get('asr'),dict) or not isinstance(r.get('quality_audit'),dict):rebuild_final_fail.append({'serial':r['serial'],'reason':'missing ASR or audit evidence'});continue
   records,stats=assemble._assemble_from_asr(r['scene'],[r['asr']])
   if len(records)!=1:rebuild_final_fail.append({'serial':r['serial'],'reason':'assemble count '+str(len(records))});continue
   item=records[0];meta=dict(item.get('meta') or {});meta['quality_audit']=r['quality_audit'];item={**item,'meta':meta}
  passed,reason=validate_audit(item['meta'].get('quality_audit',{}),item['input'],item['output']['refined_text'])
  if not passed or exporter.convert_one(item) is None:rebuild_final_fail.append({'serial':r['serial'],'reason':'audit/export validation failed: '+str(reason)});continue
  quality_items.append((r,item));quality_item_by_serial[r['serial']]=item
 if rebuild_final_fail:raise RuntimeError('Final deterministic quality_pass reconstruction failed: '+json.dumps(rebuild_final_fail,ensure_ascii=False))
 dedup=StreamingDeduplicator(settings.DEDUP_THRESHOLD);retained=[];intent_counts=Counter();input_targets={};reject=Counter();decisions=[]
 for r in merged:
  item=quality_item_by_serial.get(r['serial']) if r['status']=='quality_pass' else None
  if r['status']!='quality_pass':reject[r['status']]+=1;decisions.append({'id':r['id'],'serial':r['serial'],'scene':r['scene'],'status':r['status'],'reason':r.get('reason','')});continue
  key=exporter.intent_group_key(item);ih=hashlib.sha256(item['input'].encode()).hexdigest();th=hashlib.sha256(item['output']['refined_text'].encode()).hexdigest()
  if ih in input_targets and input_targets[ih]!=th:status='input_target_conflict'
  elif intent_counts[key]>=3:status='intent_frequency_limit'
  elif not dedup.add(item):status='near_duplicate'
  else:status='retained';retained.append(item);intent_counts[key]+=1;input_targets[ih]=th
  reject[status]+=1;decisions.append({'id':r['id'],'serial':r['serial'],'scene':r['scene'],'status':status})
 with (OUT/'retained.jsonl').open('w',encoding='utf8') as f:
  for item in retained:f.write(json.dumps(item,ensure_ascii=False)+'\n')
 with (OUT/'retention.jsonl').open('w',encoding='utf8') as f:
  for d in decisions:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 report,samples=audit.audit(OUT/'retained.jsonl',OUT/'audit_report.json',OUT/'review_sample.jsonl',min(TARGET,len(retained)),SEED,min_records=len(retained));dump(OUT/'audit_report.json',report)
 with (OUT/'review_sample.jsonl').open('w',encoding='utf8') as f:
  for x in samples:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 api_rows=[json.loads(x) for x in api_path.read_text().splitlines() if x.strip()] if api_path.exists() else [];api_stages=Counter(x.get('stage','unknown') for x in api_rows)
 diffs=Counter('different' if x['input']!=x['output']['refined_text'] else 'same' for x in retained)
 stop_diagnostics={'scope':'parent-provided diagnostic of the original 72 attempts; it does not alter candidate outcomes','oral_rejected':3,'clean_rejected_diagnostic':{'numeric_guard':13,'english_guard':8,'both_numeric_and_english_guards':7,'surface_issue':2,'other':1},'audit_rejected_diagnostic':{'unsupported_content_edits':23,'model_semantic_rejects':3}}
 summary={'target_denominator':TARGET,'candidate_count':len(merged),'unique_serials':len(set(merged_serials)),'serial_min':min(merged_serials),'serial_max':max(merged_serials),'serials_missing':sorted(set(range(TARGET))-set(merged_serials)),'serial_duplicates':sorted(n for n,c in Counter(merged_serials).items() if c>1),'preserved_interrupted_rows':len(all_rows),'newly_generated_missing_jobs':len(new_rows),'interrupted_serials_preserved':sorted(completed),'quality_pass_before_retention':sum(r.get('status')=='quality_pass' for r in merged),'retained':len(retained),'raw_retained_rate':len(retained)/TARGET,'reject_reasons':dict(reject),'candidate_stage_counts':dict(Counter(r['status'] for r in merged)),'retained_by_scene':dict(Counter(x['meta']['scene'] for x in retained)),'candidate_by_scene':dict(Counter(r['scene'] for r in merged)),'scene_quotas':quotas,'seed':SEED,'seed_source':manifest['seed_source'],'seed_aggregate_sha256':manifest['seed_aggregate_sha256'],'teacher_model':model,'teacher_endpoint':BASE,'api_calls_total':len(api_rows),'api_calls_by_stage':dict(api_stages),'api_call_errors':sum('error' in x for x in api_rows),'retained_input_vs_output':{'same':diffs['same'],'different':diffs['different']},'target_origin_counts':dict(Counter((x.get('meta') or {}).get('target_origin','unknown') for x in retained)),'source_manifest_aggregate_sha256':manifest['aggregate_sha256'],'full_structural_audit':{'passed':report['passed'],'errors':report['errors'],'counts':report['counts'],'input_sha256':report['input_sha256']},'review_sample_records':len(samples),'training_performed':False,'interruption_diagnostics':stop_diagnostics,'recovery_preflight':preflight,'completed_utc':utc()}
 dump(OUT/'summary.json',summary)
 progress={'stage':'complete' if report['passed'] else 'complete_audit_failed','target':TARGET,'completed':len(merged),'unique_serials':len(set(merged_serials)),'missing_serials':[],'duplicates':[],'quality_pass':summary['quality_pass_before_retention'],'retained':len(retained),'raw_retained_rate':len(retained)/TARGET,'candidate_status_counts':summary['candidate_stage_counts'],'retained_status_counts':summary['reject_reasons'],'scene_quotas':quotas,'api_calls':len(api_rows),'api_calls_by_stage':dict(api_stages),'audit_passed':report['passed'],'updated_utc':summary['completed_utc'],'worker_pid':os.getpid()}
 dump(OUT/'progress.json',progress);print(json.dumps(summary,ensure_ascii=False),flush=True);await client.client.close()
if __name__=='__main__':asyncio.run(main())
