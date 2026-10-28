import json,random,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
rows=[]
for variant in ['baseline','revised']:
 for line in (p/f'{variant}.jsonl').read_text().splitlines():
  r=json.loads(line)
  if r.get('oral_text') and r['phenomenon']!='clean_passthrough':
   rows.append(dict(variant=variant,**r))
random.Random(732901).shuffle(rows)
key=[];blind=[]
for i,r in enumerate(rows,1):
 bid=f'pairblind_{i:03d}'
 key.append(dict(blind_id=bid,variant=r['variant'],case_id=r['case_id'],phenomenon=r['phenomenon']))
 blind.append(dict(blind_id=bid,target=r['clean_target'],oral=r['oral_text']))
for name,data in [('blind_key.jsonl',key),('blind_inputs.jsonl',blind)]:
 (p/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
print(json.dumps(dict(blind_count=len(blind),input_sha256=hashlib.sha256((p/'blind_inputs.jsonl').read_bytes()).hexdigest())))
