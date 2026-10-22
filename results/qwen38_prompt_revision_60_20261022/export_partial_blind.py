import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows={r['case_no']:r for r in map(json.loads,(ROOT/'predictions_qwen38.jsonl').read_text().splitlines())}
sample=[json.loads(l) for l in (ROOT/'sample_60.jsonl').read_text().splitlines()]
for label,part in [('01',sample[:30]),('02',sample[30:])]:
 missing={r['case_no'] for r in part}-set(rows)
 if missing:
  print(label,'pending',len(missing));continue
 cases=[]
 for source in part:
  row=rows[source['case_no']]
  if row['error']:continue
  flip=int(hashlib.sha256((row['id']+'qwen38comparison').encode()).hexdigest(),16)%2
  a,b=(row['clean_text'],row['baseline_target']) if flip else (row['baseline_target'],row['clean_text'])
  cases.append({'case_no':row['case_no'],'input':row['input'],'candidate_a':a,'candidate_b':b})
 (ROOT/f'blind_cases_{label}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
 print(label,'exported',len(cases))
