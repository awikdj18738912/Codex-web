import hashlib,json,random
from pathlib import Path
BASE=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018')
MANIFEST=BASE/'sft_reviewed_v5_20261021/split_manifest.jsonl'
TRAIN=BASE/'final/train.jsonl'
OUT=Path(__file__).parent/'sample_500.jsonl'
train_target_hashes=set()
for line in MANIFEST.open():
 r=json.loads(line)
 if r.get('split')=='train': train_target_hashes.add(r['target_sha256'])
rows=[]
for line in TRAIN.open():
 r=json.loads(line); target=r.get('output',{}).get('refined_text',''); scene=r.get('meta',{}).get('scene','')
 if not target or r.get('meta',{}).get('target_sha256') in train_target_hashes:
  pass
 digest=hashlib.sha256(target.encode()).hexdigest()
 if digest not in train_target_hashes: continue
 if not (20<=len(target)<=240) or '__ENTITY_' in target: continue
 if scene not in {'daily_chat','vibe_coding','explanation','meeting','customer_service','academic','navigation','dictation_memo','voice_search'}:continue
 rows.append({'source_id':r.get('source_record_id'),'scene':scene,'clean_target':target,
              'target_sha256':digest})
# Stable selection across scenes, shuffled after allocation.
quotas={'daily_chat':75,'vibe_coding':85,'explanation':70,'meeting':55,'customer_service':55,'academic':50,'navigation':35,'dictation_memo':45,'voice_search':30}
rng=random.Random(20261022);picked=[]
for scene,n in quotas.items():
 pool=[r for r in rows if r['scene']==scene];rng.shuffle(pool)
 if len(pool)<n: raise SystemExit(f'{scene}: only {len(pool)} eligible targets for quota {n}')
 picked.extend(pool[:n])
rng.shuffle(picked)
phenomena=['clean_passthrough']*75+['fillers']*125+['repetition_false_start']*100+['self_correction']*125+['boundary_punctuation']*50+['natural_mixed_code_switch']*25
rng.shuffle(phenomena)
records=[]
for i,(r,phen) in enumerate(zip(picked,phenomena)):
 records.append({'case_id':f'target_first_{i:04d}','source_id':r['source_id'],'scene':r['scene'],
                 'clean_target':r['clean_target'],'target_sha256':r['target_sha256'],'phenomenon':phen})
OUT.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
print(json.dumps({'eligible_verified_train_targets':len(rows),'records':len(records),'scene_quotas':quotas,
 'phenomena':{p:sum(x['phenomenon']==p for x in records) for p in sorted(set(phenomena))},
 'source_manifest':str(MANIFEST),'sample_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
