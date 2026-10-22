from __future__ import annotations
import asyncio,fcntl,hashlib,importlib.util,json,os,random,sys,threading,time,urllib.request
from collections import Counter
from contextvars import ContextVar
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT/'source_snapshot';DATA=Path('/home/aim0/data/datasets/ASR/refiner_original_qc_replay_200_20261018')
TARGET=200;CONC=4;BASE='http://127.0.0.1:8010/v1';MODEL='refiner-teacher';CTX=ContextVar('qc_replay_context',default={});APPEND_LOCK=threading.Lock()
LOCK=(OUT/'worker.lock').open('w');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
def utc():return datetime.now(timezone.utc).isoformat()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode('utf8')).hexdigest()
def dump(path,obj):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def append(path,row):
 with APPEND_LOCK:
  with open(path,'a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def load_script(name,filename):
 sp=importlib.util.spec_from_file_location(name,ROOT/'pipeline/scripts'/filename);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
async def tagged(stage,cid,coro):
 tok=CTX.set({'stage':stage,'id':cid})
 try:return await coro
 finally:CTX.reset(tok)
async def main():
 sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
 os.environ.update(VLLM_BASE_URL=BASE,VLLM_MODEL_NAME=MODEL,OPENROUTER_API_KEY='',OPENROUTER_MODEL='',CLEAN_TARGET_MODE='teacher',REFINER_TRAIN_ZH_ITN_LIBRARY=str(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so'))
 from configs import settings
 for key,sub in [('SEEDS_DIR','seeds'),('RAW_DIR','raw'),('PROCESSED_DIR','processed'),('FINAL_DIR','final')]:
  folder=DATA/sub;folder.mkdir(parents=True,exist_ok=True);setattr(settings,key,str(folder))
 from src.pipeline.dedup import StreamingDeduplicator
 from src.services.llm_client import AsyncLLMClient
 assemble=load_script('replay_assemble','04_assemble.py');finalize=load_script('replay_finalize','05_finalize.py');exporter=load_script('replay_export','export_sft.py');audit=load_script('replay_audit','audit_dataset.py')
 manifest=json.loads((OUT/'source_manifest.json').read_text())
 for rel,h in manifest['source_sha256'].items():
  if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source hash mismatch: '+rel)
 for name,h in manifest['seed_sha256'].items():
  if hashlib.sha256((DATA/'seeds'/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen seed hash mismatch: '+name)
 source=OUT/'filterfix_attempts.jsonl';source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
 if source_sha!=manifest['input_attempts_sha256']:raise RuntimeError('Source attempt file changed')
 input_rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
 if len(input_rows)!=TARGET:raise RuntimeError(f'Expected exactly 200 source candidates, got {len(input_rows)}')
 filterfix_summary=json.loads((OUT/'filterfix_summary.json').read_text());seed=int(filterfix_summary['seed'])
 quotas={s:int(TARGET*w) for s,w in settings.SCENE_DISTRIBUTION.items()};rem=TARGET-sum(quotas.values())
 for s in sorted(quotas,key=lambda x:TARGET*settings.SCENE_DISTRIBUTION[x]-quotas[x],reverse=True)[:rem]:quotas[s]+=1
 jobs=[s for s,n in quotas.items() for _ in range(n)];random.Random(seed).shuffle(jobs)
 serials=[r.get('serial') for r in input_rows]
 if sorted(serials)!=list(range(TARGET)) or len(set(serials))!=TARGET:raise RuntimeError('Source serials are not a unique 0..199 sequence')
 candidates=[];missing_asr=[];mapping_errors=[]
 for r in sorted(input_rows,key=lambda x:x['serial']):
  n=r['serial'];expected=f'original_{jobs[n]}_{n:04d}'
  if r.get('id')!=expected or r.get('scene')!=jobs[n]:mapping_errors.append({'serial':n,'id':r.get('id'),'scene':r.get('scene'),'expected_id':expected,'expected_scene':jobs[n]})
  if isinstance(r.get('asr'),dict) and r['asr']:candidates.append(r)
  else:missing_asr.append(r)
 if mapping_errors:raise RuntimeError('Source candidate scene/id mismatch: '+json.dumps(mapping_errors,ensure_ascii=False))
 if len(candidates)!=186 or len(missing_asr)!=14 or any(r.get('status')!='oral_rejected' for r in missing_asr):
  raise RuntimeError(f'Expected 186 ASR-backed plus 14 oral_rejected; got {len(candidates)}+{len(missing_asr)}')
 assembled=[];assembly_rejects=[]
 for r in candidates:
  records,stats=assemble._assemble_from_asr(r['scene'],[r['asr']])
  if len(records)!=1:assembly_rejects.append({'id':r['id'],'serial':r['serial'],'stats':stats});continue
  assembled.append((r,records[0]))
 if assembly_rejects or len(assembled)!=186:raise RuntimeError('Source ASR candidates failed frozen assembly reconstruction: '+json.dumps(assembly_rejects,ensure_ascii=False))
 # Check the actual teacher deployment identity before any replay calls.
 model_info=json.load(urllib.request.urlopen(BASE+'/models',timeout=10));model=next((x for x in model_info.get('data',[]) if x.get('id')==MODEL),None)
 if not model or '/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit' not in str(model.get('root','')):raise RuntimeError('Unexpected teacher model identity')
 api_path=OUT/'api_calls.jsonl';checkpoint=OUT/'candidate_results.jsonl'
 api_cache={}
 if api_path.exists():
  for line in api_path.read_text().splitlines():
   if not line.strip():continue
   old=json.loads(line)
   if old.get('response') is not None:api_cache[old['request_key']]=old['response']
 done_rows=[json.loads(x) for x in checkpoint.read_text().splitlines() if x.strip()] if checkpoint.exists() else []
 done_ids=[x.get('id') for x in done_rows]
 if len(done_ids)!=len(set(done_ids)):raise RuntimeError('Duplicate IDs in candidate checkpoint')
 allowed={r['id'] for r,_ in assembled}
 if not set(done_ids).issubset(allowed):raise RuntimeError('Unknown ID in candidate checkpoint')
 class Recorded(AsyncLLMClient):
  async def generate(self,**kwargs):
   ctx=CTX.get();reqkey=digest(canonical({'stage':ctx.get('stage'),'id':ctx.get('id'),'request':kwargs}))
   if reqkey in api_cache:return api_cache[reqkey]
   row={'stage':ctx.get('stage','unknown'),'id':ctx.get('id'),'request_key':reqkey,'request':kwargs};t=time.monotonic()
   try:
    response=await super().generate(**kwargs);row['response']=response;api_cache[reqkey]=response;return response
   except Exception as e:row['error']=type(e).__name__+':'+str(e);raise
   finally:row.update(seconds=round(time.monotonic()-t,3),utc=utc());append(api_path,row)
 client=Recorded(base_url=BASE,model=MODEL,max_concurrent=CONC,timeout=600,max_retries=3)
 completed={x['id'] for x in done_rows};queue=asyncio.Queue()
 for original,item in assembled:
  if original['id'] not in completed:queue.put_nowait((original,item))
 results=list(done_rows);done=len(results);sem=asyncio.Semaphore(CONC)
 def update_progress(stage='running'):
  by=Counter(x['status'] for x in results);api_n=sum(1 for _ in api_path.open()) if api_path.exists() else 0
  dump(OUT/'progress.json',{'stage':stage,'target_denominator':TARGET,'source_candidates':TARGET,'validated_oral':len(candidates),'assembled_pairs':len(assembled),'replay_completed':done,'binary_qc_pass':by['qc_pass'],'binary_qc_reject':by['qc_reject'],'precheck_reject':by['precheck_reject'],'qc_errors':by['qc_error'],'api_calls_logged':api_n,'source_serials_unique':len(set(serials)),'source_serials_missing':sorted(set(range(TARGET))-set(serials)),'updated_utc':utc()})
 update_progress()
 async def worker():
  nonlocal done
  while True:
   try:original,item=queue.get_nowait()
   except asyncio.QueueEmpty:return
   t=time.monotonic();cid=original['id'];row={'serial':original['serial'],'id':cid,'scene':original['scene'],'status':'precheck_reject','reason':'not_run'}
   try:
    checked,passed,reason=await tagged('qc_replay',cid,finalize.qc_one(item,client,sem))
    evidence=(checked.get('meta') or {}).get('quality_audit')
    if evidence is None:
     row.update(status='precheck_reject',reason=reason,quality_audit=None)
    else:
     row.update(status='qc_pass' if passed else 'qc_reject',reason=reason,quality_audit=evidence,record=checked)
   except Exception as e:row.update(status='qc_error',reason=type(e).__name__+':'+str(e))
   row['seconds']=round(time.monotonic()-t,3);append(checkpoint,row);results.append(row);done+=1;update_progress();queue.task_done()
 await asyncio.gather(*(worker() for _ in range(CONC)))
 # Ensure all 186 source candidates have one and only one replay outcome.
 if len(results)!=len(assembled) or {x['id'] for x in results}!={r['id'] for r,_ in assembled}:raise RuntimeError('Replay checkpoint set does not cover the 186 assembled candidates exactly')
 result_by_id={x['id']:x for x in results}
 if len(result_by_id)!=len(results):raise RuntimeError('Duplicate replay candidate result IDs')
 # Export then apply exactly the filterfix retention gates to QC-passing pairs.
 dedup=StreamingDeduplicator(settings.DEDUP_THRESHOLD);retained=[];intents=Counter();inputs={};outcomes=Counter();decisions=[];export_errors=[]
 for original,item in assembled:
  outcome=result_by_id[original['id']];
  if outcome['status']!='qc_pass':outcomes[outcome['status']]+=1;decisions.append({'serial':original['serial'],'id':original['id'],'scene':original['scene'],'status':outcome['status'],'reason':outcome.get('reason','')});continue
  checked=outcome['record'];
  if exporter.convert_one(checked) is None:outcomes['export_rejected']+=1;export_errors.append(original['id']);decisions.append({'serial':original['serial'],'id':original['id'],'scene':original['scene'],'status':'export_rejected'});continue
  key=exporter.intent_group_key(checked);ih=hashlib.sha256(checked['input'].encode()).hexdigest();th=hashlib.sha256(checked['output']['refined_text'].encode()).hexdigest()
  if ih in inputs and inputs[ih]!=th:status='input_target_conflict'
  elif intents[key]>=3:status='intent_frequency_limit'
  elif not dedup.add(checked):status='near_duplicate'
  else:status='retained';retained.append(checked);intents[key]+=1;inputs[ih]=th
  outcomes[status]+=1;decisions.append({'serial':original['serial'],'id':original['id'],'scene':original['scene'],'status':status})
 with (OUT/'retained.jsonl').open('w',encoding='utf8') as f:
  for x in retained:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 with (OUT/'retention.jsonl').open('w',encoding='utf8') as f:
  for x in decisions:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 report,samples=audit.audit(OUT/'retained.jsonl',OUT/'audit_report.json',OUT/'review_sample.jsonl',min(TARGET,len(retained)),seed,min_records=len(retained));dump(OUT/'audit_report.json',report)
 with (OUT/'review_sample.jsonl').open('w',encoding='utf8') as f:
  for x in samples:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 old_retained=[json.loads(x) for x in (OUT/'filterfix_retained.jsonl').read_text().splitlines() if x.strip()]
 old_ids={x.get('id') for x in old_retained};new_ids={x.get('id') for x in retained};api_rows=[json.loads(x) for x in api_path.read_text().splitlines() if x.strip()]
 api_stages=Counter(x.get('stage','unknown') for x in api_rows)
 qc_counts=Counter(x['status'] for x in results)
 old_quality_ids={r['id'] for r in input_rows if r.get('status')=='quality_pass'}
 summary={'denominator':TARGET,'source_candidate_count':len(input_rows),'validated_oral':len(input_rows)-len(missing_asr),'assembled_pairs':len(assembled),'source_serial_count':len(set(serials)),'source_serial_min':min(serials),'source_serial_max':max(serials),'source_serial_missing':sorted(set(range(TARGET))-set(serials)),'source_serial_duplicates':sorted(n for n,c in Counter(serials).items() if c>1),'binary_qc_pass':qc_counts['qc_pass'],'binary_qc_reject':qc_counts['qc_reject'],'precheck_reject':qc_counts['precheck_reject'],'qc_errors':qc_counts['qc_error'],'export_rejected':outcomes['export_rejected'],'retained':len(retained),'retained_rate_fixed_denominator':len(retained)/TARGET,'retention_outcomes':dict(outcomes),'retained_by_scene':dict(Counter(x['meta']['scene'] for x in retained)),'qc_status_by_scene':{scene:dict(Counter(res['status'] for res in results if res['scene']==scene)) for scene in sorted({x['scene'] for x in results})},'api_call_count':len(api_rows),'api_calls_by_stage':dict(api_stages),'api_call_errors':sum('error' in x for x in api_rows),'same_filterfix_accepted_count':len(old_quality_ids),'qc_pass_overlap_with_filterfix':len(old_quality_ids & {x['id'] for x in results if x['status']=='qc_pass'}),'qc_pass_new_vs_filterfix':len({x['id'] for x in results if x['status']=='qc_pass'}-old_quality_ids),'qc_pass_lost_vs_filterfix':len(old_quality_ids-{x['id'] for x in results if x['status']=='qc_pass'}),'retained_overlap_with_filterfix':len(old_ids & new_ids),'retained_only_filterfix':len(old_ids-new_ids),'retained_only_qc_replay':len(new_ids-old_ids),'source_manifest_aggregate_sha256':manifest['source_aggregate_sha256'],'teacher_model':model,'teacher_endpoint':BASE,'full_structural_audit':{'passed':report['passed'],'errors':report['errors'],'counts':report['counts'],'input_sha256':report['input_sha256']},'training_performed':False,'completed_utc':utc()}
 dump(OUT/'summary.json',summary)
 final_progress={'stage':'complete' if report['passed'] and len(results)==186 and len(set(serials))==200 else 'complete_audit_failed','target_denominator':TARGET,'source_candidates':len(input_rows),'validated_oral':len(candidates),'assembled_pairs':len(assembled),'replay_completed':len(results),'unique_source_serials':len(set(serials)),'source_serial_missing':summary['source_serial_missing'],'binary_qc_pass':qc_counts['qc_pass'],'binary_qc_reject':qc_counts['qc_reject'],'retained':len(retained),'retained_rate_fixed_denominator':len(retained)/TARGET,'api_calls':len(api_rows),'api_calls_by_stage':dict(api_stages),'audit_passed':report['passed'],'updated_utc':summary['completed_utc']}
 dump(OUT/'progress.json',final_progress);print(json.dumps(summary,ensure_ascii=False),flush=True);await client.client.close()
if __name__=='__main__':asyncio.run(main())
