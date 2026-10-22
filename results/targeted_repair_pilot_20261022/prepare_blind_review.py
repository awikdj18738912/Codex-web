"""Build deterministic blinded candidate comparisons, hiding teacher decisions."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[json.loads(l) for l in (ROOT/'teacher_proposals.jsonl').read_text().splitlines()]
cases=[];keys=[]
for row in sorted(rows,key=lambda r:r['case_no']):
    if row['error'] or row['proposal']['decision']=='UNRESOLVED':continue
    old=row['existing_target'];new=row['proposal']['clean_text']
    reverse=int(hashlib.sha256((row['id']+'targeted20261022').encode()).hexdigest(),16)%2
    a,b=(new,old) if reverse else (old,new)
    cases.append({'case_no':row['case_no'],'input':row['input'],'candidate_a':a,'candidate_b':b})
    keys.append({'case_no':row['case_no'],'id':row['id'],'a_is':'proposal' if reverse else 'old',
                 'b_is':'old' if reverse else 'proposal','is_control':row['is_control'],
                 'teacher_decision':row['proposal']['decision'],'identical':old==new})
for name,data in [('blind_cases.jsonl',cases),('blind_key.jsonl',keys)]:
    (ROOT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
for suffix,part in [('001_050',cases[:50]),('051_100',cases[50:])]:
    (ROOT/f'blind_cases_{suffix}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in part))
for lo,hi in [(21,60),(61,100)]:
    part=[r for r in cases if lo<=int(r['case_no'][1:])<=hi]
    (ROOT/f'blind_remaining_{lo:03d}_{hi:03d}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in part))
print(json.dumps({'blinded_cases':len(cases),'unresolved_or_error':len(rows)-len(cases)}))
