"""Create separate versioned candidate pools from the live audit ledger."""
from pathlib import Path
from collections import Counter
import json,hashlib,time,shutil
P=Path(__file__).resolve().parent
STORE=Path('/home/aim0/data/ASR/dataset_repair_20261023')
META=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/final/train.jsonl')
def read(path):
 for line in path.open():
  if line.strip():
   try:yield json.loads(line)
   except json.JSONDecodeError:continue
rows={}
for f in [P/'reused_pilot_results.jsonl',*sorted((P/'outputs').glob('batch_*.jsonl'))]:
 for r in read(f):
  if r['record_id']in rows:raise ValueError('duplicate record '+r['record_id'])
  rows[r['record_id']]=r
meta={}
for r in read(META):
 if r.get('id')in rows:
  m=r.get('meta',{});meta[r['id']]={'weak_speech_phenomena':m.get('speech_phenomena',[]),'weak_is_fragment':m.get('is_fragment'),'weak_language':m.get('language'),'annotation_scope':'original generation metadata, not verified phenotype labels'}
version=time.strftime('%Y%m%d_%H%M%S');dest=STORE/'pools'/version;dest.mkdir(parents=True)
names=['core_candidates','repaired_candidates','review_queue','replacement_candidates'];streams={n:(dest/f'{n}.jsonl').open('w')for n in names};regen=(dest/'fresh_seed_generation_requests.jsonl').open('w')
counts=Counter();scenes={n:Counter()for n in names};labels={n:Counter()for n in names};base_labels=Counter();base_scenes=Counter();noop=Counter();original_by_split=Counter()
for r in rows.values():
 d=r['decision'];source_status=r.get('checks',{}).get('source',{}).get('result',{}).get('status');m=meta.get(r['record_id'],{});tags=m.get('weak_speech_phenomena',[]);tags=tags if isinstance(tags,list)else[tags]
 scene=r.get('scene','unknown');base_scenes[scene]+=1;base_labels.update(set(map(str,tags)));split=r['original_split'];original_by_split[split]+=1
 if d=='accept_candidate':pool='repaired_candidates'if r.get('approved_prior_patch')else'core_candidates'
 elif d=='repaired_candidate':pool='repaired_candidates'
 elif d=='quarantine':pool='replacement_candidates'
 else:pool='review_queue'
 target=r['repair']['result']['repaired_target']if d=='repaired_candidate'else r['target']
 row={**r,**m,'candidate_target':target,'candidate_target_sha256':hashlib.sha256(target.encode()).hexdigest(),'pool':pool,'quality_status':'model-screened candidate; independent sample and student evaluation pending'}
 streams[pool].write(json.dumps(row,ensure_ascii=False)+'\n');counts[pool]+=1;scenes[pool][scene]+=1;labels[pool].update(set(map(str,tags)));noop[pool]+=r['input']==target
 if pool=='replacement_candidates':
  # Retired wording is deliberately excluded from generation briefs.
  brief={'replacement_id':'replacement_'+r['record_id'],'retired_record_id':r['record_id'],'original_split':split,'scene':scene,'desired_phenomena_unverified':tags,'seed_policy':'generate a fresh coherent seed; do not reuse retired input or target','status':'await source-rejection confirmation and category coverage allocation','source_audit_status':source_status}
  regen.write(json.dumps(brief,ensure_ascii=False)+'\n')
for f in streams.values():f.close()
regen.close()
report={'version':version,'processed_including_reused':len(rows),'pool_counts':dict(counts),'by_scene':{k:dict(v)for k,v in scenes.items()},'weak_phenomena_counts':{k:dict(v)for k,v in labels.items()},'processed_baseline':{'scene':dict(base_scenes),'weak_phenomena':dict(base_labels),'split':dict(original_by_split)},'exact_noop_counts':dict(noop),'generation_started':False,'quality_scope':'teacher and model review, not human gold; original category labels are weak metadata. Pending API/quote errors are not regeneration requests. Source rejection is a replacement candidate, subject to confirmation. Coverage relates only to processed rows, not the unprocessed full corpus.','original_training_exports_modified':False}
(dest/'coverage_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(STORE/'pools/latest.json').write_text(json.dumps({'path':str(dest),'version':version},indent=2))
print(json.dumps({'path':str(dest),'processed':len(rows),'counts':dict(counts),'fresh_seed_briefs':counts['replacement_candidates']},ensure_ascii=False))
