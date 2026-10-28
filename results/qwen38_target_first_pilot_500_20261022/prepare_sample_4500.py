import hashlib,json,random,sys
from pathlib import Path
BASE=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018')
MANIFEST=BASE/'sft_reviewed_v5_20261021/split_manifest.jsonl';TRAIN=BASE/'final/train.jsonl'
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'sample_4500.jsonl'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pipeline'))
from src.pipeline.training_quality import surface_issues
train_hashes=set()
for l in MANIFEST.open():
 r=json.loads(l)
 if r.get('split')=='train':train_hashes.add(r['target_sha256'])
used={json.loads(l)['target_sha256'] for l in (ROOT/'sample_500_v3.jsonl').read_text().splitlines()}
scene_quota={'daily_chat':675,'vibe_coding':765,'explanation':630,'meeting':495,'customer_service':495,'academic':450,'navigation':315,'dictation_memo':405,'voice_search':270}
rows=[];seen=set();excluded=0
for l in TRAIN.open():
 r=json.loads(l);target=r.get('output',{}).get('refined_text','');meta=r.get('meta',{});scene=meta.get('scene','')
 if not target or hashlib.sha256(target.encode()).hexdigest() not in train_hashes:continue
 h=hashlib.sha256(target.encode()).hexdigest()
 if h in used or h in seen:continue
 if surface_issues(target,target):excluded+=1;continue
 if not 20<=len(target)<=180 or '__ENTITY_' in target:continue
 if scene not in scene_quota:continue
 seen.add(h);rows.append({'source_id':meta.get('source_record_id',r.get('source_record_id')),'scene':scene,'clean_target':target,'target_sha256':h})
rng=random.Random(20261025);picked=[]
for scene,n in scene_quota.items():
 pool=[r for r in rows if r['scene']==scene];rng.shuffle(pool)
 if len(pool)<n:raise SystemExit(f'{scene}: need {n}, available {len(pool)}')
 picked+=pool[:n]
rng.shuffle(picked)
phen=['clean_passthrough']*675+['fillers']*1125+['repetition_false_start']*900+['self_correction']*1125+['boundary_punctuation']*450+['natural_mixed_code_switch']*225
rng.shuffle(phen)
records=[]
for i,(r,p) in enumerate(zip(picked,phen)):
 records.append({'case_id':f'target_first_bulk_{i:04d}',**r,'phenomenon':p})
OUT.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
print(json.dumps({'new_targets':len(records),'preexisting_500_overlap':len(used & {r['target_sha256'] for r in records}),'scene_quotas':scene_quota,'phenomena':{p:sum(r['phenomenon']==p for r in records) for p in sorted(set(phen))},'sample_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'editorial_artifacts_excluded':excluded},ensure_ascii=False,indent=2))
