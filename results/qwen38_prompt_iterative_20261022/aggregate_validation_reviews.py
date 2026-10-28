"""Aggregate two independent blinded audits; preserve raw opinions separately."""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'validation_second_pass'
def read(name):
    p=ROOT/name
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
cases={r['case_no']:r for r in read('blind_cases.jsonl')}
keys={r['case_no']:r for r in read('blind_key.jsonl')}
review_names=['review_luna1.jsonl','review_luna2.jsonl']
reviews=[]
for name in review_names:
    rows={r['case_no']:r for r in read(name)}
    assert set(rows)==set(cases),(name,len(rows),len(cases))
    reviews.append(rows)
leads={r['case_no']:r for r in read('lead_adjudications.jsonl')} if (ROOT/'lead_adjudications.jsonl').exists() else {}
valid={'PASS','MINOR','FAIL','AMBIGUOUS'}
raw_pair_counts=[Counter(),Counter()]
ratings={'baseline':Counter(),'candidate':Counter()};wins=Counter();agreement=Counter();out=[]
for n,case in sorted(cases.items()):
    key=keys[n];a_id=key['a_is'];b_id=key['b_is']
    assert {a_id,b_id}=={'baseline','candidate'}
    pair=[]
    for j,rmap in enumerate(reviews):
        r=rmap[n]
        assert r['a_verdict'] in valid and r['b_verdict'] in valid
        assert r['winner'] in {'A','B','TIE','UNRESOLVED'}
        if key['identical']:
            assert r['a_verdict']==r['b_verdict'] and r['winner'] in {'TIE','UNRESOLVED'}
        g={a_id:r['a_verdict'],b_id:r['b_verdict']}
        w={'A':a_id,'B':b_id}.get(r['winner'],r['winner'])
        pair.append((g,w))
        raw_pair_counts[j][w]+=1
    same=pair[0]==pair[1]
    agreement['same_grades_and_winner' if same else 'needs_lead_review']+=1
    if n in leads:
        lead=leads[n]
        grade={'baseline':lead['baseline_verdict'],'candidate':lead['candidate_verdict']}
        winner=lead['winner_identity'];note=lead['reason']
    else:
        # Do not silently resolve a reviewer disagreement: retain UNRESOLVED in the aggregate.
        grade={m:(pair[0][0][m] if pair[0][0][m]==pair[1][0][m] else 'UNRESOLVED') for m in ('baseline','candidate')}
        winner=pair[0][1] if pair[0][1]==pair[1][1] else 'UNRESOLVED'
        note=''
    for model in ratings:
        if grade[model] in valid: ratings[model][grade[model]]+=1
    wins[winner]+=1
    out.append({**case,'id':key['id'],'baseline_verdict':grade['baseline'],'candidate_verdict':grade['candidate'],
                'winner_identity':winner,'review1_winner':pair[0][1],'review2_winner':pair[1][1],
                'review1_baseline':pair[0][0]['baseline'],'review1_candidate':pair[0][0]['candidate'],
                'review2_baseline':pair[1][0]['baseline'],'review2_candidate':pair[1][0]['candidate'],
                'lead_adjudicated':n in leads,'lead_note':note})
(ROOT/'comparison_final.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
report={'items':len(out),'reviews_per_item':2,'rating_distributions':{k:dict(v) for k,v in ratings.items()},
        'paired_wins':dict(wins),'reviewer_agreement':dict(agreement),'raw_winners':{f'reviewer_{i+1}':dict(c) for i,c in enumerate(raw_pair_counts)},
        'lead_adjudications':len(leads),'limitation':'One-use held-out comparison for this local Qwen3.8 source-guided second-pass variant; model-assisted reviewers; not population accuracy.'}
(ROOT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
