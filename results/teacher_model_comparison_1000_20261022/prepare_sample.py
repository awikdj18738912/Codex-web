"""Prepare a fixed, scene-balanced, risk-stratified teacher comparison sample."""
import hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/sft_reviewed_v5_20261021')
RAW=Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/final/train.jsonl')
SEED=2026102201
QUOTAS={'boundary_punctuation':100,'multi_correction':200,'english_entity_or_code':150,
        'numeric_surface':100,'single_correction':150,'clean_or_passthrough':100,'other_cleanup':200}
correction=re.compile(r'不对|不是|改成|改为|说错|记错|等等|还是|刚才.{0,8}不是|最后定|应该是')
identifier=re.compile(r'(?<![A-Za-z])[A-Z][A-Za-z0-9]*(?:[._:/-][A-Za-z0-9_-]+)*|\b[a-z][a-z0-9_]*(?:\.[a-zA-Z0-9_-]+)+\b|\b[A-Za-z]+_[A-Za-z0-9_]+\b|\b(?:[A-Z]{2,}[A-Za-z0-9_-]*)\b')
number=re.compile(r'(?<![A-Za-z])\d+(?:[.,:/-]\d+)*(?:%|毫秒|秒|元|块|公里|米|GB|MB|TB|号|日|月|年)?')
boundary=re.compile(r'[。！？；，、…][\s“”‘’]*?(?:让我|我觉得|这个|就是|结果|特别|真的|有点|还挺|不过|不对|最后)')

sft=json.loads((DATA/'train_sft.json').read_text())
manifest=[json.loads(l) for l in (DATA/'split_manifest.jsonl').read_text().splitlines() if l.strip()]
manifest=[x for x in manifest if x['split']=='train']
assert len(sft)==len(manifest)==84875
raw={}
for line in RAW.open():
 row=json.loads(line);raw[str(row['id'])]=row
pools=defaultdict(lambda:defaultdict(list));all_rows=[]
for i,(row,m) in enumerate(zip(sft,manifest)):
 conv=row['conversations'];source=conv[1]['value'];target=conv[2]['value'];rid=str(m['source_id'])
 phenomena=(raw.get(rid,{}).get('meta') or {}).get('speech_phenomena') or []
 phenomena_text=' '.join(map(str,phenomena)).lower();n=len(correction.findall(source))
 if 'boundary_punctuation' in phenomena_text or boundary.search(source):stratum='boundary_punctuation'
 elif n>=2 or 'rollback' in phenomena_text or 'multiple' in phenomena_text:stratum='multi_correction'
 elif identifier.search(source):stratum='english_entity_or_code'
 elif number.search(source):stratum='numeric_surface'
 elif n==1:stratum='single_correction'
 elif source==target:stratum='clean_or_passthrough'
 else:stratum='other_cleanup'
 item={'case_no':None,'id':rid,'scene':m['scene'],'source_decision':m['decision'],'stratum':stratum,
       'input':source,'existing_target':target,'spoken_source':raw.get(rid,{}).get('oral_text',source),
       'speech_phenomena':phenomena}
 pools[stratum][m['scene']].append(item);all_rows.append(item)

rng=random.Random(SEED)
scenes=sorted({r['scene'] for r in all_rows})
scene_targets={s:(112 if i==len(scenes)-1 else 111) for i,s in enumerate(scenes)}
selected=[];scene_counts=Counter();selected_ids=set();shortfalls={}
for category,quota in QUOTAS.items():
 by_scene=pools[category]
 for values in by_scene.values():rng.shuffle(values)
 remaining=quota
 while remaining:
  eligible=[s for s in scenes if by_scene.get(s) and by_scene[s] and scene_counts[s]<scene_targets[s]]
  if not eligible:
   eligible=[s for s in scenes if by_scene.get(s) and by_scene[s]]
  if not eligible:
   shortfalls[category]=remaining;break
  min_ratio=min(scene_counts[s]/scene_targets[s] for s in eligible)
  choices=[s for s in eligible if scene_counts[s]/scene_targets[s]==min_ratio]
  scene=rng.choice(choices);item=by_scene[scene].pop()
  assert item['id'] not in selected_ids
  selected_ids.add(item['id']);selected.append(item);scene_counts[scene]+=1;remaining-=1
# Use remaining candidates to complete exact scene targets if a risk stratum was scarce.
if len(selected)<1000:
 remaining_by_scene=defaultdict(list)
 for item in all_rows:
  if item['id'] not in selected_ids:remaining_by_scene[item['scene']].append(item)
 for scene in scenes:
  rng.shuffle(remaining_by_scene[scene])
 while len(selected)<1000:
  eligible=[s for s in scenes if scene_counts[s]<scene_targets[s] and remaining_by_scene[s]]
  if not eligible:eligible=[s for s in scenes if remaining_by_scene[s]]
  scene=min(eligible,key=lambda s:scene_counts[s]/scene_targets[s])
  item=remaining_by_scene[scene].pop();item={**item,'stratum':'scene_balanced_random'}
  selected_ids.add(item['id']);selected.append(item);scene_counts[scene]+=1
rng.shuffle(selected)
for i,item in enumerate(selected,1):item['case_no']=i
out=ROOT/'sample_1000.jsonl'
out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected))
report={'records':len(selected),'seed':SEED,'source_split':'train only; not validation','source_records':len(sft),
 'requested_strata':QUOTAS,'actual_strata':dict(Counter(x['stratum'] for x in selected)),
 'by_scene':dict(Counter(x['scene'] for x in selected)),'by_source_decision':dict(Counter(x['source_decision'] for x in selected)),
 'scene_targets':scene_targets,'stratum_shortfalls':shortfalls,
 'sample_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'candidate_models':['Qwen3-30B-A3B-Instruct-2507-AWQ-4bit','Qwen3-4B-Instruct-2507'],
 'note':'Risk-stratified and scene-balanced diagnostic sample, not a prevalence sample. Generation requests will contain input only; existing_target is kept for later comparison.'}
(ROOT/'sample_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
