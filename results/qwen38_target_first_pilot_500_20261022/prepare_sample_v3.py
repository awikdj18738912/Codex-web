import hashlib,json,random,re,sys
from pathlib import Path
BASE=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018')
MANIFEST=BASE/'sft_reviewed_v5_20261021/split_manifest.jsonl'
TRAIN=BASE/'final/train.jsonl'
OUT=Path(__file__).parent/'sample_500_v3.jsonl'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pipeline'))
from src.pipeline.training_quality import surface_issues
train_target_hashes=set()
for line in MANIFEST.open():
 r=json.loads(line)
 if r.get('split')=='train':train_target_hashes.add(r['target_sha256'])
editor_gloss=re.compile(r'[\u3400-\u9fff]{1,12}[（(]\s*[A-Za-z][A-Za-z0-9 ._+/#&-]{1,60}[）)]')
scene_quotas={'daily_chat':75,'vibe_coding':85,'explanation':70,'meeting':55,'customer_service':55,'academic':50,'navigation':35,'dictation_memo':45,'voice_search':30}
rows=[];seen=set();exclude=0
for line in TRAIN.open():
 r=json.loads(line);target=r.get('output',{}).get('refined_text','');meta=r.get('meta',{});scene=meta.get('scene','')
 if not target or hashlib.sha256(target.encode()).hexdigest() not in train_target_hashes:continue
 if editor_gloss.search(target) or surface_issues(target,target):exclude+=1;continue
 if not (20<=len(target)<=180) or '__ENTITY_' in target:continue
 if scene not in scene_quotas:continue
 h=hashlib.sha256(target.encode()).hexdigest()
 if h in seen:continue
 seen.add(h)
 rows.append({'source_id':meta.get('source_record_id',r.get('source_record_id')),'scene':scene,'clean_target':target,'target_sha256':h})
rng=random.Random(20261023);picked=[]
for scene,n in scene_quotas.items():
 pool=[r for r in rows if r['scene']==scene];rng.shuffle(pool)
 if len(pool)<n:raise SystemExit(f'{scene}: only {len(pool)} unique eligible train targets for quota {n}')
 picked.extend(pool[:n])
rng.shuffle(picked)
phenomena=['clean_passthrough']*75+['fillers']*125+['repetition_false_start']*100+['self_correction']*125+['boundary_punctuation']*50+['natural_mixed_code_switch']*25
rng.shuffle(phenomena)
records=[]
for i,(r,phen) in enumerate(zip(picked,phenomena)):
 records.append({'case_id':f'target_first_v2_{i:04d}',**r,'phenomenon':phen})
OUT.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
print(json.dumps({'eligible_unique_verified_train_targets':len(rows),'excluded_editor_gloss_targets_seen':exclude,'records':len(records),'scene_quotas':scene_quotas,'phenomena':{p:sum(r['phenomenon']==p for r in records) for p in sorted(set(phenomena))},'sample_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
