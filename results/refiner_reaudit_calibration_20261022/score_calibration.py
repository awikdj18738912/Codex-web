import argparse,json
from pathlib import Path
from collections import Counter
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('results',type=Path);a=parser.parse_args()
read=lambda f:[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
key={r['audit_id']:r for r in read(p/'reference_key.jsonl')};rows=read(a.results);matrix=Counter();errors=[];disagreements=[]
for r in rows:
 k=key[r['audit_id']];act=r.get('audit',{}).get('action','error');matrix[(k['reference_verdict'],act)]+=1
 if r.get('error'):errors.append(r)
 if (k['reference_verdict']=='PASS' and act!='accept') or (k['reference_verdict']=='FAIL' and act=='accept') or (k['reference_verdict']=='REVIEW' and act!='quarantine'):
  disagreements.append(dict(**r,reference=k))
report=dict(completed=len(rows),errors=len(errors),confusion={k:{a:n for (v,a),n in matrix.items() if v==k} for k in ['PASS','MINOR','FAIL','REVIEW']},disagreement_count=len(disagreements),limitations='Diagnostic references are earlier GPT reviews, not human gold. Resolve disagreements before treating the confusion matrix as accuracy. Five FAIL references per split cannot establish full-corpus false-accept rate.')
a.results.with_suffix('.score.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
a.results.with_suffix('.disagreements.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in disagreements))
print(json.dumps(report,ensure_ascii=False,indent=2))
