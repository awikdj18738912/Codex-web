"""Export stable numbered chunks once all their requests are present."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows={r['case_no']:r for r in map(json.loads,(ROOT/'predictions_qwen38.jsonl').read_text().splitlines())}
for lo,hi in [(1,67),(68,134),(135,200)]:
    missing=set(range(lo,hi+1))-set(rows)
    if missing:
        print(lo,hi,'pending',len(missing));continue
    cases=[]
    for n in range(lo,hi+1):
        row=rows[n]
        if row['error']:continue
        flip=int(hashlib.sha256((row['id']+'qwen38comparison').encode()).hexdigest(),16)%2
        old,new=row['baseline_target'],row['clean_text'];a,b=(new,old) if flip else (old,new)
        cases.append({'case_no':n,'input':row['input'],'candidate_a':a,'candidate_b':b})
    (ROOT/f'blind_cases_{lo:03d}_{hi:03d}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    print(lo,hi,'exported',len(cases))
