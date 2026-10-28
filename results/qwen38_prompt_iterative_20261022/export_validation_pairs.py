"""Freeze anonymous baseline-vs-second-pass comparisons for one-use validation."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'validation_second_pass'
def read(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
sample=read(ROOT/'validation_100_with_draft.jsonl')
pred=read(FOLDER/'predictions.jsonl')
by_id={r['id']:r for r in pred}
assert len(sample)==100 and len(by_id)==len(pred)==100
assert set(by_id)=={r['id'] for r in sample}
cases=[];keys=[]
for no,row in enumerate(sample,1):
    out=by_id[row['id']]
    assert out['input']==row['input']
    assert out['baseline_target']==row['baseline_target']
    assert not out.get('error') and isinstance(out.get('clean_text'),str) and out['clean_text'].strip()
    base,candidate=row['baseline_target'],out['clean_text']
    flip=int(hashlib.sha256((row['id']+'qwen38-two-pass-sealed-validation-v1').encode()).hexdigest(),16)%2
    a,b=(candidate,base) if flip else (base,candidate)
    cases.append({'case_no':no,'input':row['input'],'candidate_a':a,'candidate_b':b})
    keys.append({'case_no':no,'id':row['id'],'a_is':'candidate' if flip else 'baseline','b_is':'baseline' if flip else 'candidate','identical':a==b})
for name,rows in [('blind_cases.jsonl',cases),('blind_key.jsonl',keys)]:
    (FOLDER/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
for idx,start in enumerate(range(0,100,50),1):
    (FOLDER/f'blind_cases_{idx:02d}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases[start:start+50]))
manifest={'items':len(cases),'candidate_prompt_sha256':json.loads((FOLDER/'freeze_record.json').read_text())['prompt_sha256'],
          'candidate_output_sha256':hashlib.sha256((FOLDER/'predictions.jsonl').read_bytes()).hexdigest(),
          'validation_sample_sha256':hashlib.sha256((ROOT/'validation_100_with_draft.jsonl').read_bytes()).hexdigest(),
          'key_sha256':hashlib.sha256((FOLDER/'blind_key.jsonl').read_bytes()).hexdigest(),
          'ordering':'A/B randomized per item by SHA-256 id+fixed salt; key kept separate from reviews.'}
(FOLDER/'blind_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'exported':len(cases),'identical':sum(k['identical'] for k in keys),'split':[50,50]},ensure_ascii=False))
