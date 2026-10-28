import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/sft_reviewed_v5_20261021')
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
manifest={}
for line in (DATA/'split_manifest.jsonl').open():
 r=json.loads(line);manifest[(r['split'],r['output_index'])]=r
report={'scope':'read-only inventory of exact SFT files; flags are not semantic verdicts','files':{},'counts':{},'input_conflicts':0,'cross_split_exact_inputs':0,'missing_or_inconsistent_manifest':0,'flags':{},'training_data_changed':False};seen=defaultdict(list);flagcounts=Counter();flags=[]
for split,filename in [('train','train_sft.json'),('validation','val_sft.json')]:
 f=DATA/filename;raw=f.read_bytes();rows=json.loads(raw);report['files'][filename]={'path':str(f),'sha256':hashlib.sha256(raw).hexdigest()};report['counts'][split]=len(rows)
 for i,row in enumerate(rows):
  turns=row['conversations'];human=[x['value'] for x in turns if x['from']=='human'];target=[x['value'] for x in turns if x['from']=='gpt']
  if len(human)!=1 or len(target)!=1:raise ValueError(f'multi-turn record at {split}:{i}')
  source,gold=human[0],target[0];m=manifest.get(('val' if split=='validation' else split,i)) or manifest.get((split,i));hs,ht=sha(source),sha(gold)
  if not m or m['input_sha256']!=hs or m['target_sha256']!=ht:report['missing_or_inconsistent_manifest']+=1
  seen[hs].append((split,ht,m.get('source_id') if m else None))
  reasons=[]
  if source==gold:reasons.append('exact_noop')
  if any(c in source or c in gold for c in '（）【】'):reasons.append('bracket_needs_context')
  if any(w in source for w in ['不对','说错','不是','改成','等等']):reasons.append('correction_or_negation_needs_context')
  if len(source)>400:reasons.append('long_input')
  flagcounts.update(reasons)
  if reasons:flags.append({'split':split,'index':i,'source_id':m.get('source_id') if m else None,'flags':reasons})
for group in seen.values():
 if len({x[1] for x in group})>1:report['input_conflicts']+=1
 if len({x[0] for x in group})>1:report['cross_split_exact_inputs']+=1
report['flags']=dict(flagcounts)
(ROOT/'export_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(ROOT/'structural_risk_index.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in flags))
print(json.dumps(report,ensure_ascii=False,indent=2))
