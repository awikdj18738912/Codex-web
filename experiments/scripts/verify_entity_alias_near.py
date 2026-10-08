"""Compare integrated code with frozen experiment outputs and old regressions."""
from collections import Counter
from dataclasses import replace
import json
import sqlite3
from evaluate_entity_alias_near import OUT, defs as new_definitions
from evaluate_entity_library import read_jsonl, write_json, ROOT
OLD = ROOT/'results/entity_rule_sweep_20261008'
from evaluate_medical_entities import lexicon_occurrences
from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.protection import EntityProtector


def verify():
    ds=tuple(replace(d,spelling_context=replace(d.spelling_context,
        keywords=tuple((w,v) for w,v in d.spelling_context.keywords if w not in d.canonical_text)))
        if d.canonical_text in {'ECMO','ECMO手术'} else d for d in new_definitions())
    p=EntityProtector(ds,protect_numeric_spans=False)
    m=EntityCandidateMatcher(ds,selected_domain='medical',mode='auto')
    summaries={};changes=[];mismatches=[]
    names={d.canonical_text for d in ds}
    for label in ['development','holdout']:
        stats=Counter()
        for row in read_jsonl(OUT/f'{label}_outputs.jsonl'):
            result=prepare_entity_segment(row['raw_text'],p,m,allow_auto=True)
            wanted=row['outputs']['near_actions'];baseline=row['outputs']['current']
            stats.update(samples=1,changed=int(result.baseline_text!=baseline),prototype_mismatches=int(result.baseline_text!=wanted))
            if result.baseline_text!=wanted:mismatches.append(dict(label=label,**row,actual=result.baseline_text))
            if row['variant'] in {'clean','noise'}:
                expected=Counter(e['text'] for e in lexicon_occurrences(row['reference'],names))
                for key,text in [('baseline',baseline),('new',result.baseline_text)]:
                    have=Counter(e['text'] for e in lexicon_occurrences(text,names))
                    stats[key+'_recovered']+=sum(min(n,have[w]) for w,n in expected.items())
                stats['expected_mentions']+=sum(expected.values());stats['audio_versions']+=1
            if result.baseline_text!=baseline:changes.append(dict(label=label,id=row['id'],variant=row['variant'],reference=row['reference'],baseline=baseline,output=result.baseline_text,normalizations=result.normalizations))
        summaries[label]=dict(stats)
    write_json(OUT/'integrated_verification.json',dict(summaries=summaries,changes=changes,prototype_mismatches=mismatches))
    print(json.dumps(summaries,ensure_ascii=False),flush=True)

    # A reusable configuration contains two equivalent aliases, no ASR errors.
    config=json.loads((OLD/'medical_entities.json').read_text())
    for row in config:
        if row['canonical_text'] in {'ECMO','ECMO手术'}:
            row['aliases']=['艾克膜' if row['canonical_text']=='ECMO' else '艾克膜手术']
            policy=next(d.spelling_context for d in ds if d.canonical_text==row['canonical_text'])
            row['spelling_context']=dict(keywords=dict(policy.keywords),minimum_score=policy.minimum_score,source=policy.source,allow_internal_omission=False,allow_near_spelling=True)
    write_json(OUT/'medical_entities.json',config)
    db=OUT/'official_alias_terms.db'
    if not db.exists():
        source=sqlite3.connect(f'file:{OLD / "official_alias_terms.db"}?mode=ro',uri=True);dest=sqlite3.connect(db)
        source.backup(dest);dest.close();source.close()
    store=EntityStore(db)
    for d in ds:
        if d.canonical_text in {'ECMO','ECMO手术'}:
            store.upsert_entity(d.canonical_text,entity_type=d.entity_type,domain=d.domain,normalization_policy=d.normalization_policy,
                priority=d.priority,aliases=d.aliases,source=d.source+'; explicit near alias permission')
            policy=d.spelling_context
            store.set_spelling_context(d.entity_id,keywords=dict(policy.keywords),minimum_score=policy.minimum_score,source=policy.source,allow_near_spelling=True)

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
