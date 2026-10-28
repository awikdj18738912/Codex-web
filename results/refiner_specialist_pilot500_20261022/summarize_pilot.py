import json
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent
read=lambda p:[json.loads(x)for x in p.read_text().splitlines() if x.strip()]
rows=read(P/'pilot500_results.jsonl');report={'completed':len(rows),'target':500,'formal_training_approved':False,'arms':{},'limitations':'Teacher candidate counts are not independent quality pass rates. All specialist/recheck calls use the same local model. Risk-enriched and uniform-random arms must be reported separately.'}
for arm in ['uniform_random','risk_enriched']:
 rs=[r for r in rows if r['sampling_arm']==arm]
 report['arms'][arm]={'completed':len(rs),'target':250,'decisions':dict(Counter(r['decision']for r in rs)),'specialists':{stage:dict(Counter(r['checks'][stage].get('result',{}).get('status','error')for r in rs))for stage in ['source','fidelity','cleanup']},'repaired_candidates':sum(r['decision']=='repaired_candidate'for r in rs),'repair_attempts':sum('repair'in r for r in rs)}
if (P/'lead_prereview20_verdicts.jsonl').exists():
 verdicts={r['blind_id']:r for r in read(P/'lead_prereview20_verdicts.jsonl')};key=read(P/'lead_prereview_key.jsonl');outputs={r['audit_id']:r for r in rows};cross=[]
 for k in key:
  if k['audit_id']not in outputs:continue
  model=outputs[k['audit_id']];lead=verdicts[k['blind_id']]
  cross.append({**k,'teacher_decision':model['decision'],'teacher_statuses':{s:v.get('result',{}).get('status','error')for s,v in model['checks'].items()},'lead':lead})
 report['lead_prereview_comparison']={'completed_available':len(cross),'prereview_total':20,'accepted_with_lead_fail_or_uncertainty':sum(r['teacher_decision']=='accept_candidate'and(r['lead']['source_status']!='valid'or r['lead']['fidelity']!='pass'or r['lead']['cleanup']!='pass')for r in cross),'scope':'primary-model blind review performed before reading teacher scores, not human gold or guaranteed quality'}
 (P/'lead_teacher_comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in cross))
(P/'pilot_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
