"""Validate paired reviews and retain explicit adjudication records."""
import argparse,csv,json,re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def norm_evidence(s):return ''.join(ch for ch in s if not (ch.isspace() or ch in "，。；：？！、‘’“”\"'（）()《》<>"))
p=argparse.ArgumentParser();p.add_argument('--round',required=True);p.add_argument('--prefix',default='screen');a=p.parse_args();folder=ROOT/a.round
cases=read(folder/(a.prefix+'_cases.jsonl'));keys={r['case_no']:r for r in read(folder/(a.prefix+'_key.jsonl'))}
files=sorted(folder.glob(a.prefix+'_review_*.jsonl'));raw=[r for f in files for r in read(f)]
assert len(raw)==len({r['case_no'] for r in raw})==len(cases),'review coverage count'
reviews={r['case_no']:r for r in raw};assert set(reviews)=={r['case_no'] for r in cases}
leads_path=folder/(a.prefix+'_leads.jsonl');leads={r['case_no']:r for r in read(leads_path)} if leads_path.exists() else {}
out=[];counts={'baseline':Counter(),'candidate':Counter()};wins=Counter()
frozen_path=ROOT/'screen_baseline_frozen.json'
frozen=json.loads(frozen_path.read_text()) if frozen_path.exists() else None
for case in cases:
    n=case['case_no'];r=reviews[n];key=keys[n]
    for field,source in [('source_evidence',case['input']),('a_evidence',case['candidate_a']),('b_evidence',case['candidate_b'])]:
        assert isinstance(r[field],str),(n,field)
    assert r['a_verdict'] in {'PASS','MINOR','FAIL','AMBIGUOUS'} and r['b_verdict'] in {'PASS','MINOR','FAIL','AMBIGUOUS'}
    assert r['winner'] in {'A','B','TIE','UNRESOLVED'}
    if key['identical']:assert r['a_verdict']==r['b_verdict'] and r['winner'] in {'TIE','UNRESOLVED'},n
    grade={key['a_is']:r['a_verdict'],key['b_is']:r['b_verdict']}
    win={'A':key['a_is'],'B':key['b_is']}.get(r['winner'],r['winner']);note=r['reason']
    if n in leads:
        lead=leads[n];grade={'baseline':lead['baseline_verdict'],'candidate':lead['candidate_verdict']};win=lead['winner_identity'];note=lead['reason']
    if n==84:
        grade={'baseline':'AMBIGUOUS','candidate':'AMBIGUOUS'};win='UNRESOLVED';note='Pre-frozen source ambiguity:20000 vs2000capacity without an explicit final correction; isolate rather than infer a gold value.'
    if frozen and a.prefix=='screen':
        expected=frozen['grades'].get(str(n))
        if expected is not None:grade['baseline']=expected
        if n in frozen['ambiguous_source_case_numbers']:
            grade={'baseline':'AMBIGUOUS','candidate':'AMBIGUOUS'};win='UNRESOLVED'
        elif key['identical']:
            grade['candidate']=grade['baseline'];win='TIE'
    if key['identical']:assert grade['baseline']==grade['candidate'] and win in {'TIE','UNRESOLVED'}
    for model in counts:counts[model][grade[model]]+=1
    wins[win]+=1
    out.append({**case,**r,'id':key['id'],'group':key['group'],'baseline_verdict':grade['baseline'],'candidate_verdict':grade['candidate'],
                'winner_identity':win,'lead_note':note,'a_is':key['a_is'],'b_is':key['b_is']})
preds=read(folder/'predictions.jsonl')
summary={'compared_records':len(out),'generation_records_so_far':len(preds),'generation_errors':sum(bool(r['error']) for r in preds),
 'ratings':{k:dict(v) for k,v in counts.items()},'paired_wins':dict(wins),'semantic_regression_cases':[r['case_no'] for r in out if r['baseline_verdict'] not in {'FAIL','AMBIGUOUS'} and r['candidate_verdict']=='FAIL'],
 'semantic_improvement_cases':[r['case_no'] for r in out if r['baseline_verdict']=='FAIL' and r['candidate_verdict'] not in {'FAIL','AMBIGUOUS'}],
 'limitation':'Development/risk-enriched comparison with model-assisted judgments and explicit lead adjudications; not population accuracy.'}
(folder/(a.prefix+'_final.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
(folder/(a.prefix+'_report.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (folder/(a.prefix+'_final.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(out[0]) if out else []);writer.writeheader()
    for r in out:writer.writerow({k:("'"+v if isinstance(v,str) and v.lstrip()[:1] in ['=','+','-','@'] else v) for k,v in r.items()})
print(json.dumps(summary,ensure_ascii=False,indent=2))
