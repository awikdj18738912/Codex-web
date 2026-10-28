import json,random
from pathlib import Path
P=Path(__file__).resolve().parent
read=lambda p:[json.loads(x)for x in p.read_text().splitlines() if x.strip()]
rows=read(P/'pilot500_results.jsonl');rng=random.Random(20261023);selected=[]
for arm in ['uniform_random','risk_enriched']:
 subset=[r for r in rows if r['sampling_arm']==arm and r['decision']=='accept_candidate'];selected.extend((r,'accepted_sample')for r in rng.sample(subset,min(25,len(subset))))
repaired=[r for r in rows if r['decision']=='repaired_candidate'];selected.extend((r,'repaired_check')for r in rng.sample(repaired,min(100,len(repaired))))
failed=[r for r in rows if r['decision']in ['quarantine','needs_review','repair_needs_review','repair_candidate']];selected.extend((r,'rejected_or_pending_sample')for r in rng.sample(failed,min(25,len(failed))))
rng.shuffle(selected);blind=[];key=[]
for i,(r,kind)in enumerate(selected,1):
 bid=f'specialistblind_{i:03d}';target=r['repair']['result']['repaired_target']if kind=='repaired_check'else r['target'];blind.append({'blind_id':bid,'input':r['input'],'target':target});key.append({'blind_id':bid,'audit_id':r['audit_id'],'record_id':r['record_id'],'sampling_arm':r['sampling_arm'],'kind':kind,'decision':r['decision']})
for name,data in [('spotcheck_blind.jsonl',blind),('spotcheck_key.jsonl',key)]: (P/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in data))
print(json.dumps({'accepted_checks':sum(k['kind']=='accepted_sample'for k in key),'repaired_checks':sum(k['kind']=='repaired_check'for k in key),'pending_checks':sum(k['kind']=='rejected_or_pending_sample'for k in key),'total':len(blind)}))
