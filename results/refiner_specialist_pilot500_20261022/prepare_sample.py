import json,random,re,hashlib
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent
D=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/sft_reviewed_v5_20261021')
OLD=P.parent/'refiner_reaudit_calibration_20261022'
excluded={r['record_id'] for r in map(json.loads,(OLD/'reference_key.jsonl').read_text().splitlines())}
for r in map(json.loads,(P.parent/'refiner_validation_blind_audit_256_20261022/final_c30a1d4cf9a9/audited_comparison_256.jsonl').read_text().splitlines()):
 for key in ['id','source_id','record_id']:
  if isinstance(r.get(key),str):excluded.add(r[key])
manifest=[json.loads(x) for x in (D/'split_manifest.jsonl').read_text().splitlines()]
sets={'train':json.loads((D/'train_sft.json').read_text()),'val':json.loads((D/'val_sft.json').read_text())}
rows=[]
for m in manifest:
 if m['source_id'] in excluded:continue
 split='val' if m['split'] in ['val','validation'] else 'train';turns=sets[split][m['output_index']]['conversations'];source=next(t['value'] for t in turns if t['from']=='human');target=next(t['value'] for t in turns if t['from']=='gpt')
 assert hashlib.sha256(source.encode()).hexdigest()==m['input_sha256']
 assert hashlib.sha256(target.encode()).hexdigest()==m['target_sha256']
 flags=[]
 if any(w in source for w in ['不对','说错','不是','改成','等等']):flags.append('correction_or_negation')
 if re.search('[A-Za-z]',source):flags.append('mixed_code')
 if len(source)>400:flags.append('long_input')
 if any(c in source or c in target for c in '（）【】'):flags.append('brackets')
 rows.append(dict(record_id=m['source_id'],original_split=split,output_index=m['output_index'],scene=m['scene'],input=source,target=target,input_sha256=m['input_sha256'],target_sha256=m['target_sha256'],split_group_sha256=m['split_group_sha256'],risk_flags=flags))
rng=random.Random(20261023);random_rows=rng.sample(rows,250);selected={r['record_id'] for r in random_rows};out=[dict(r,sampling_arm='uniform_random',sampling_stratum='uniform') for r in random_rows]
for flag,n in [('brackets',25),('long_input',50),('mixed_code',75),('correction_or_negation',100)]:
 candidates=[r for r in rows if r['record_id'] not in selected and flag in r['risk_flags']]
 chosen=rng.sample(candidates,n);selected.update(r['record_id'] for r in chosen);out.extend(dict(r,sampling_arm='risk_enriched',sampling_stratum=flag) for r in chosen)
rng.shuffle(out)
for i,r in enumerate(out,1):r['audit_id']=f'pilot500_{i:03d}'
(P/'sample500.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
oldrows=[json.loads(x) for x in (OLD/'development_probe20.jsonl').read_text().splitlines()]
(P/'calibration20.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in oldrows))
report={'eligible_population':len(rows),'excluded_previously_reviewed_ids':len(excluded),'samples':500,'sampling_arms':dict(Counter(r['sampling_arm'] for r in out)),'risk_strata':dict(Counter(r['sampling_stratum'] for r in out if r['sampling_arm']=='risk_enriched')),'source_files_sha256':{name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in ['train_sft.json','val_sft.json','split_manifest.jsonl']},'sample_sha256':hashlib.sha256((P/'sample500.jsonl').read_bytes()).hexdigest(),'scope':'random arm estimates only the eligible previously unreviewed population; risk arm is diagnostic, never combine as full-dataset error rate','formal_dataset_modified':False}
(P/'sample_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
