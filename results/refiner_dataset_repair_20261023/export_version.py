from pathlib import Path
from collections import Counter
import json,hashlib,random
if (Path(__file__).resolve().parent/'error_catalog_reaudit_required.json').exists():
 raise RuntimeError('Legacy export superseded: run refiner_dataset_repair_20261024_error_loop/export_candidates.py after its full re-audit')
P=Path(__file__).resolve().parent;M=json.loads((P/'manifest.json').read_text());D=Path(M['original_data']);OUT=P/'candidate_sft';OUT.mkdir(exist_ok=True)
read=lambda p:[json.loads(x)for x in p.read_text().splitlines()if x.strip()]
results=read(P/'reused_pilot_results.jsonl')
for f in sorted((P/'outputs').glob('batch_*.jsonl')):results.extend(read(f))
assert len(results)==M['total'] and len({r['record_id']for r in results})==M['total'],'incomplete or duplicate result set'
original={'train':json.loads((D/'train_sft.json').read_text()),'val':json.loads((D/'val_sft.json').read_text())};candidates=[];pending=[]
for r in results:
 if r['decision']not in ['accept_candidate','repaired_candidate']:pending.append(r);continue
 split=r['original_split'];split='val'if split in ['val','validation']else'train';row=json.loads(json.dumps(original[split][r['output_index']]))
 target=r['repair']['result']['repaired_target']if r['decision']=='repaired_candidate'else r['target']
 actual=next(t['value']for t in row['conversations']if t['from']=='human');old=next(t['value']for t in row['conversations']if t['from']=='gpt')
 assert hashlib.sha256(actual.encode()).hexdigest()==r['input_sha256'] and hashlib.sha256(old.encode()).hexdigest()==r['target_sha256']
 for t in row['conversations']:
  if t['from']=='gpt':t['value']=target
 candidates.append((split,row,{'record_id':r['record_id'],'original_split':split,'original_index':r['output_index'],'input_sha256':r['input_sha256'],'old_target_sha256':r['target_sha256'],'new_target_sha256':hashlib.sha256(target.encode()).hexdigest(),'decision':r['decision']}))
val_targets={m['new_target_sha256']for s,r,m in candidates if s=='val'};kept=[];leakage=[]
for s,r,m in candidates:
 if s=='train'and m['new_target_sha256']in val_targets:leakage.append(m)
 else:kept.append((s,r,m))
for split,filename in [('train','train_sft.json'),('val','val_sft.json')]:
 (OUT/filename).write_text(json.dumps([r for s,r,m in kept if s==split],ensure_ascii=False))
(OUT/'lineage.jsonl').write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n'for s,r,m in kept))
(OUT/'pending_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in pending))
(OUT/'cross_split_target_collisions.json').write_text(json.dumps(leakage,ensure_ascii=False,indent=2))
info=json.loads((D/'dataset_info.json').read_text());(OUT/'dataset_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
rng=random.Random(20261023);sample=[]
for split in ['train','val']:
 group=[(r,m)for s,r,m in kept if s==split]
 for r,m in rng.sample(group,min(150,len(group))):sample.append({'lineage':m,'conversations':r['conversations']})
(OUT/'post_repair_audit_sample300.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in sample))
report={'status':'candidate_version_exported_pending_independent_sample_and_training_ablation','original_records':M['total'],'kept':len(kept),'by_split':dict(Counter(s for s,r,m in kept)),'pending':len(pending),'cross_split_collisions_removed_from_train':len(leakage),'formal_quality_guarantee':False,'original_data_overwritten':False}
(OUT/'quality_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report)
