"""Export frozen anonymous comparison chunks with no model or prompt identity."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
p=argparse.ArgumentParser();p.add_argument('--round',required=True);p.add_argument('--limit',type=int,default=40);p.add_argument('--chunk-size',type=int,default=20);p.add_argument('--prefix',default='screen');a=p.parse_args()
folder=ROOT/a.round;sample=read(ROOT/'development_regression_160.jsonl')[:a.limit]
pred={r['id']:r for r in read(folder/'predictions.jsonl')}
cases=[];keys=[]
for r in sample:
    if r['id'] not in pred:continue
    n=pred[r['id']]
    assert n['input']==r['input'] and n['baseline_target']==r['baseline_target']
    if n['error']:continue
    flip=int(hashlib.sha256((r['id']+a.round+'blind').encode()).hexdigest(),16)%2
    old,new=r['baseline_target'],n['clean_text'];x,y=(new,old) if flip else (old,new)
    cases.append({'case_no':r['case_no'],'input':r['input'],'candidate_a':x,'candidate_b':y})
    keys.append({'case_no':r['case_no'],'id':r['id'],'a_is':'candidate' if flip else 'baseline','b_is':'baseline' if flip else 'candidate','identical':old==new,'group':r['group']})
for name,rows in [(a.prefix+'_cases.jsonl',cases),(a.prefix+'_key.jsonl',keys)]:
    (folder/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
for i in range(0,len(sample),a.chunk_size):
    subset=sample[i:i+a.chunk_size];ids={r['id'] for r in subset}
    if not ids<=set(pred):print('chunk',i//a.chunk_size+1,'pending');continue
    numbers={r['case_no'] for r in subset};out=[r for r in cases if r['case_no'] in numbers]
    (folder/f'{a.prefix}_cases_{i//a.chunk_size+1:02d}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
    print('chunk',i//a.chunk_size+1,'exported',len(out))
