from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;OLD=P.parent/'refiner_specialist_pilot500_20261022';D=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/sft_reviewed_v5_20261021')
read=lambda p:[json.loads(x)for x in p.read_text().splitlines()if x.strip()]
previous=read(OLD/'pilot500_results.jsonl');lookup={r['audit_id']:r['record_id']for r in previous};issues={}
for r in read(OLD/'confirmed_or_uncertain_followup_queue.jsonl'):
 issues[lookup[r['audit_id']]]={'reason':r['review']['note'],'source_uncertain':r['review']['source_status']!='valid'}
C=P.parent/'refiner_reaudit_calibration_20261022'
for name in ['known_error_queue.jsonl','known_ambiguous_queue.jsonl']:
 for r in read(C/name):issues.setdefault(r['record_id'],{'reason':r['reason'],'source_uncertain':'ambiguous'in name})
reused=[r for r in previous if r['record_id']not in issues];reused_ids={r['record_id']for r in reused}
(P/'reused_pilot_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in reused))
patches={r['record_id']:r for r in read(C/'reviewed_patch_overlay.jsonl')}
sets={'train':json.loads((D/'train_sft.json').read_text()),'val':json.loads((D/'val_sft.json').read_text())};rows=[]
for m in read(D/'split_manifest.jsonl'):
 if m['source_id']in reused_ids:continue
 split='val'if m['split']in ['val','validation']else'train';turns=sets[split][m['output_index']]['conversations'];source=next(t['value']for t in turns if t['from']=='human');target=next(t['value']for t in turns if t['from']=='gpt')
 assert hashlib.sha256(source.encode()).hexdigest()==m['input_sha256'] and hashlib.sha256(target.encode()).hexdigest()==m['target_sha256']
 r={'audit_id':m['source_id'],'record_id':m['source_id'],'original_split':split,'output_index':m['output_index'],'input':source,'target':target,'original_target':target,'input_sha256':m['input_sha256'],'target_sha256':m['target_sha256'],'split_group_sha256':m['split_group_sha256'],'scene':m['scene']}
 if m['source_id']in patches:
  patch=patches[m['source_id']];assert patch['original_target_sha256']==m['target_sha256'];r['target']=patch['proposed_target'];r['approved_prior_patch']=True
 elif m['source_id']in issues:r['known_issue']=issues[m['source_id']]['reason'];r['force_quarantine']=issues[m['source_id']]['source_uncertain']
 rows.append(r)
rows.sort(key=lambda r:(not bool(r.get('known_issue')or r.get('approved_prior_patch')),r['original_split'],r['output_index']))
(P/'inputs').mkdir(exist_ok=True);(P/'outputs').mkdir(exist_ok=True)
for i in range(0,len(rows),500):(P/'inputs'/f'batch_{i//500:04d}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rows[i:i+500]))
report={'total':99852,'reused':len(reused),'queued':len(rows),'batches':(len(rows)+499)//500,'priority_known_issues':len(issues),'prior_reviewed_patches':len(patches),'original_data':str(D),'source_hashes':{n:hashlib.sha256((D/n).read_bytes()).hexdigest()for n in ['train_sft.json','val_sft.json']},'policy':'Residual error allowed; candidate dataset is evaluated by independent stratified sample and student metrics, not required to be error free. Preserve original split and all decisions. API/quote failures remain unresolved, not silently treated as bad data.'}
(P/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
f=P/'run_specialists.py';s=f.read_text();s=s.replace("out={**row,'checks':checks", "if row.get('force_quarantine'):decision='quarantine'\n  elif row.get('known_issue') and status('source')=='pass' and not any(v.get('error')for v in checks.values()):decision='repair_candidate'\n  out={**row,'checks':checks")
s=s.replace("fix=await call(row,'repair',index,checks)","fix=await call(row,'repair',index,{'checks':checks,'prior_confirmed_issue':row.get('known_issue')})")
f.write_text(s);compile(s,str(f),'exec')
