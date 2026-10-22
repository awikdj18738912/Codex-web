"""Offline postprocessing fix for the interrupted original-boundary experiment.
No teacher API calls are made here.
"""
import hashlib,importlib.util,json,os,random,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT/'source_snapshot';DATA=Path('/home/aim0/data/datasets/ASR/refiner_original_boundary_200_20261017')
TARGET=200;SEED=2026101720
def utc():return datetime.now(timezone.utc).isoformat()
def dump(path,obj):
 p=Path(path);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
os.environ.update(CLEAN_TARGET_MODE='teacher',REFINER_TRAIN_ZH_ITN_LIBRARY=str(ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so'))
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT)]
from configs import settings
from src.pipeline.training_quality import validate_audit
from src.pipeline.dedup import StreamingDeduplicator
manifest=json.loads((OUT/'source_manifest.json').read_text())
for rel,expected in manifest['sha256'].items():
 if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen source changed: '+rel)
for name,expected in manifest['seed_sha256'].items():
 if hashlib.sha256((DATA/'seeds'/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Seed changed: '+name)
asm=loadmod('frozen_assemble_final','pipeline/scripts/04_assemble.py')
exp=loadmod('frozen_export_final','pipeline/scripts/export_sft.py')
aud=loadmod('frozen_audit_final','pipeline/scripts/audit_dataset.py')
rows=[json.loads(x) for x in (OUT/'attempts.jsonl').read_text().splitlines() if x.strip()]
if len(rows)!=TARGET:raise RuntimeError(f'Expected {TARGET} attempt rows, found {len(rows)}')
serials=[x.get('serial') for x in rows]
if len(set(serials))!=TARGET or sorted(serials)!=list(range(TARGET)):raise RuntimeError('Serial verification failed')
quotas={s:int(TARGET*w) for s,w in settings.SCENE_DISTRIBUTION.items()};rem=TARGET-sum(quotas.values())
for s in sorted(quotas,key=lambda x:TARGET*settings.SCENE_DISTRIBUTION[x]-quotas[x],reverse=True)[:rem]:quotas[s]+=1
jobs=[s for s,n in quotas.items() for _ in range(n)];random.Random(SEED).shuffle(jobs)
rows.sort(key=lambda r:r['serial'])
for r in rows:
 n=r['serial'];expected=f'original_{jobs[n]}_{n:04d}'
 if r.get('scene')!=jobs[n] or r.get('id')!=expected:raise RuntimeError(f'Serial/job mismatch: {n}')
quality_items={};failures=[]
for r in rows:
 if r.get('status')!='quality_pass':continue
 if not isinstance(r.get('asr'),dict) or not isinstance(r.get('quality_audit'),dict):failures.append({'serial':r['serial'],'reason':'missing ASR/audit'});continue
 records,stats=asm._assemble_from_asr(r['scene'],[r['asr']])
 if len(records)!=1:failures.append({'serial':r['serial'],'reason':'assembly count '+str(len(records))});continue
 item=records[0];meta=dict(item.get('meta') or {});meta['quality_audit']=r['quality_audit'];item={**item,'meta':meta}
 ok,reason=validate_audit(r['quality_audit'],item['input'],item['output']['refined_text'])
 if not ok:failures.append({'serial':r['serial'],'reason':'invalid stored audit '+str(reason)});continue
 try:converted=exp.convert_one(item)
 except Exception as e:failures.append({'serial':r['serial'],'reason':'export exception '+repr(e)});continue
 if converted is None:failures.append({'serial':r['serial'],'reason':'export returned None'});continue
 quality_items[r['serial']]=item
if failures:raise RuntimeError('Quality-pass offline reconstruction failed: '+json.dumps(failures,ensure_ascii=False))
if len(quality_items)!=sum(r['status']=='quality_pass' for r in rows):raise RuntimeError('Quality-pass rebuild count mismatch')
# Exact production sequence: input/target conflict, per-intent cap, then streaming Jaccard dedup.
dedup=StreamingDeduplicator(settings.DEDUP_THRESHOLD);retained=[];intents=Counter();targets={};decisions=[];outcomes=Counter()
for r in rows:
 status=r['status']
 if status!='quality_pass':
  outcomes[status]+=1;decisions.append({'serial':r['serial'],'id':r['id'],'scene':r['scene'],'status':status,'reason':r.get('reason','')});continue
 item=quality_items[r['serial']];key=exp.intent_group_key(item);ih=hashlib.sha256(item['input'].encode()).hexdigest();th=hashlib.sha256(item['output']['refined_text'].encode()).hexdigest()
 if ih in targets and targets[ih]!=th:final='input_target_conflict'
 elif intents[key]>=3:final='intent_frequency_limit'
 elif not dedup.add(item):final='near_duplicate'
 else:final='retained';retained.append(item);intents[key]+=1;targets[ih]=th
 outcomes[final]+=1;decisions.append({'serial':r['serial'],'id':r['id'],'scene':r['scene'],'status':final})
with (OUT/'retained.jsonl').open('w',encoding='utf8') as f:
 for item in retained:f.write(json.dumps(item,ensure_ascii=False)+'\n')
with (OUT/'retention.jsonl').open('w',encoding='utf8') as f:
 for x in decisions:f.write(json.dumps(x,ensure_ascii=False)+'\n')
report,samples=aud.audit(OUT/'retained.jsonl',OUT/'audit_report.json',OUT/'review_sample.jsonl',min(TARGET,len(retained)),SEED,min_records=len(retained));dump(OUT/'audit_report.json',report)
with (OUT/'review_sample.jsonl').open('w',encoding='utf8') as f:
 for x in samples:f.write(json.dumps(x,ensure_ascii=False)+'\n')
api=[json.loads(x) for x in (OUT/'api_calls.jsonl').read_text().splitlines() if x.strip()]
api_stage=Counter(x.get('stage','unknown') for x in api);api_errors=Counter(x.get('stage','unknown') for x in api if 'error' in x)
status=Counter(r['status'] for r in rows);scene_status=defaultdict(Counter);category_status=defaultdict(Counter);boundary_scene=defaultdict(Counter);boundary_status=Counter();category_stats=Counter()
for r in rows:
 scene_status[r['scene']][r['status']]+=1
 oral=r.get('oral') or {};phen=oral.get('speech_phenomena',[]);phen=phen if isinstance(phen,list) else [phen]
 for label in phen:
  category_status[str(label)][r['status']]+=1;category_stats[str(label)]+=1
 if 'boundary_punctuation' in phen:
  boundary_status[r['status']]+=1;boundary_scene[r['scene']][r['status']]+=1
final_scene=defaultdict(Counter)
for d in decisions:final_scene[d['scene']][d['status']]+=1
retained_change=Counter('different' if x['input']!=x['output']['refined_text'] else 'same' for x in retained)
interruption_diagnostics={'scope':'parent-provided 72-candidate diagnosis before recovery; descriptive only','oral_rejected':3,'clean_rejected':{'numeric_guard':13,'english_guard':8,'both_numeric_and_english_guards':7,'surface_issue':2,'other':1},'audit_rejected':{'unsupported_content_edits':23,'model_semantic_rejects':3}}
summary={'target_denominator':TARGET,'candidate_count':len(rows),'unique_serial_count':len(set(serials)),'serial_range':[min(serials),max(serials)],'serials_missing':sorted(set(range(TARGET))-set(serials)),'duplicate_serials':sorted(n for n,c in Counter(serials).items() if c>1),'preserved_interrupted_attempts':72,'newly_generated_missing_jobs':128,'quality_pass_before_dedup':len(quality_items),'retained':len(retained),'raw_retained_rate':len(retained)/TARGET,'candidate_stage_counts':dict(status),'final_retention_outcomes':dict(outcomes),'retained_by_scene':dict(Counter(x['meta']['scene'] for x in retained)),'candidate_by_scene':dict(Counter(r['scene'] for r in rows)),'scene_quotas':quotas,'candidate_outcomes_by_scene':{s:dict(c) for s,c in scene_status.items()},'final_retention_by_scene':{s:dict(c) for s,c in final_scene.items()},'candidate_categories_by_final_status':{k:dict(v) for k,v in category_status.items()},'boundary_punctuation_candidates':{'count':sum(boundary_status.values()),'stage_outcomes':dict(boundary_status),'by_scene':{s:dict(c) for s,c in boundary_scene.items()},'quality_pass':sum(c.get('quality_pass',0) for c in boundary_scene.values()),'retained':sum(1 for x in retained if 'boundary_punctuation' in ((x.get('meta') or {}).get('speech_phenomena') or []))},'seed':SEED,'seed_source':manifest['seed_source'],'seed_aggregate_sha256':manifest['seed_aggregate_sha256'],'teacher_identity':{'id':'refiner-teacher','root':'/home/aim0/data/models/ASR/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit'},'api_call_count_cumulative':len(api),'api_calls_by_stage':dict(api_stage),'api_call_errors_by_stage':dict(api_errors),'retained_input_vs_output':{'same':retained_change['same'],'different':retained_change['different']},'target_origin_counts':dict(Counter((x.get('meta') or {}).get('target_origin','unknown') for x in retained)),'source_manifest_aggregate_sha256':manifest['aggregate_sha256'],'full_structural_audit':{'passed':report['passed'],'errors':report['errors'],'counts':report['counts'],'input_sha256':report['input_sha256']},'review_sample_records':len(samples),'training_performed':False,'interruption_diagnostics':interruption_diagnostics,'postprocess_fix':'Rows are dicts; recovery finalizer iterates rows and resolves quality-passed records from quality_item_by_serial. No new teacher requests were made during finalization.','completed_utc':utc()}
dump(OUT/'summary.json',summary)
progress={'stage':'complete' if report['passed'] and not summary['serials_missing'] and not summary['duplicate_serials'] else 'complete_audit_failed','target':TARGET,'completed':len(rows),'unique_serials':len(set(serials)),'missing_serials':summary['serials_missing'],'duplicates':summary['duplicate_serials'],'quality_pass':len(quality_items),'retained':len(retained),'raw_retained_rate':len(retained)/TARGET,'candidate_status_counts':dict(status),'retained_status_counts':dict(outcomes),'scene_quotas':quotas,'api_calls':len(api),'api_calls_by_stage':dict(api_stage),'audit_passed':report['passed'],'updated_utc':summary['completed_utc'],'worker_pid':None}
dump(OUT/'progress.json',progress)
print(json.dumps({'candidate_status':dict(status),'retention_outcomes':dict(outcomes),'boundary':summary['boundary_punctuation_candidates'],'serials':{'count':len(serials),'unique':len(set(serials)),'missing':summary['serials_missing'],'duplicates':summary['duplicate_serials']},'quality_pass':len(quality_items),'retained':len(retained),'rate':len(retained)/TARGET,'audit':summary['full_structural_audit'],'api_calls':{'total':len(api),'by_stage':dict(api_stage)},'progress':progress['stage']},ensure_ascii=False,indent=2))
