"""Compare integrated code with frozen experiment outputs and old regressions."""
from collections import Counter
import json
import sqlite3
from evaluate_entity_library import ROOT
OUT=ROOT/'results/entity_procedure_stem_20261008'
from evaluate_entity_library import read_jsonl, write_json, ROOT
from evaluate_medical_entities import lexicon_occurrences
from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.protection import EntityProtector


def verify():
    ds=EntityStore(OUT/'live_before.db').list_entities(domain='medical')
    p=EntityProtector(ds,protect_numeric_spans=False);m=EntityCandidateMatcher(ds,selected_domain='medical',mode='auto')
    session=json.loads((OUT/'user_session_before.json').read_text())
    source=session['output']['raw_text'];result=prepare_entity_segment(source,p,m,allow_auto=True)
    write_json(OUT/'user_session_rule_after.json',dict(source=source,output=result.baseline_text,normalizations=result.normalizations,candidates=[x.public_dict() for x in result.report.matches]))
    print('User case',result.baseline_text,result.normalizations,flush=True)
    mismatches=[]
    from evaluate_entity_omission_rule import suites
    historical={(r['suite'],r['id'],r['variant']):r for r in read_jsonl(ROOT/'results/entity_rule_optimization_20261007/final_opt_in.jsonl')}
    totals={};unexpected=[]
    for suite,defs,domain,rows in suites():
        protector=EntityProtector(defs,protect_numeric_spans=False)
        matcher=EntityCandidateMatcher(defs,selected_domain=domain,mode='auto');stats=Counter()
        for row in rows:
            old=historical[(suite,row['id'],row['variant'])]['output']
            new=prepare_entity_segment(row['raw_text'],protector,matcher,allow_auto=True).baseline_text
            stats.update(samples=1,changed=int(new!=old))
            if new!=old:unexpected.append(dict(suite=suite,id=row['id'],variant=row['variant'],before=old,after=new))
        totals[suite]=dict(stats);print(suite,dict(stats),flush=True)
    write_json(OUT/'historical_regression.json',dict(summaries=totals,unexpected_changes=unexpected))
    if mismatches or unexpected:raise AssertionError('Integrated rule differs from expected frozen outputs or historical results')


if __name__=='__main__':verify()
