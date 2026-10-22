#!/usr/bin/env python3
"""Offline runtime-monkeypatch guard ablation; all source data are read-only."""
import json, sys, hashlib, importlib.util, collections
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/refiner_guard_ablation_20261017'
sys.path.insert(0,str(ROOT))
def rows(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def module_for(snapshot):
    sys.path[:] = [p for p in sys.path if '/source_snapshot/pipeline' not in p]
    for key in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
        del sys.modules[key]
    sys.path.insert(0,str(ROOT/'results'/snapshot/'source_snapshot'/'pipeline'))
    from src.utils import numeric_policy as np
    return np

def analyze300():
    baseline=rows(ROOT/'results/refiner_auto_acceptance_300_20261017/attempts.jsonl')
    calls=rows(ROOT/'results/refiner_auto_acceptance_300_20261017/api_calls.jsonl')
    asr_candidates={x['id']:x for x in calls if x.get('stage')=='asr'}
    mod=module_for('refiner_auto_acceptance_300_20261017')
    original=mod.lock_numeric_target
    def valid_partial_pair(full_input, full_target, partial_input, partial_target):
        compact=lambda t: re.sub(r"[\s，,。.!！?？:：;；、‘’“”\"'…—]+", "", t)
        source,target,inp,out=map(compact,(full_input,full_target,partial_input,partial_target))
        prefixes=[]
        for m in re.finditer(r"[,，;；。！？!?]",full_target):
            end=m.end()
            if end < 4 or end >= len(full_target): continue
            if m.start() and full_target[m.start()-1].isdigit() and full_target[end].isdigit(): continue
            prefixes.append(full_target[:end])
        return bool(len(inp)>=4 and 4<=len(out)<len(target) and len(inp)<len(source)
                    and target.startswith(out) and out in {compact(x) for x in prefixes})
    # For each row invoke the real checker with four runtime monkeypatch conditions.
    groups={f'numeric_{n}_english_{e}':[] for n,e in [(1,1),(0,1),(1,0),(0,0)]}
    details=[]
    for stub in baseline:
        # auto_acceptance is the matched completed replay; its observed responses
        # are authoritative. The earlier repair pass had 33 clean-model failures.
        r=stub
        oral=r.get('oral') or {}; clean=r.get('clean') or {}
        source=oral.get('oral_text',''); target=clean.get('clean_text','')
        plan=(oral.get('meta') or {}).get('generation_plan')
        if not source or not target or not isinstance(plan,dict):
            details.append({'id':stub['id'],'status':stub.get('status'),'evaluated':False,'reason':'no saved oral/target/plan in baseline or repair attempts'})
            continue
        expected=plan.get('expected_clean')
        row={'id':stub['id'],'status':r.get('status'),'source':source,'target':target,'expected_clean':expected,
             'construction':plan.get('construction_version'),'exact_expected':target==expected,
             'source_valid':(r.get('audit') or {}).get('source_valid'),'audit_pass':r.get('status')=='accepted',
             'asr_selected':r.get('asr_selected'),'natural_fragment_eligible':r.get('natural_fragment_eligible')}
        if r.get('status')=='asr_rejected':
            call=asr_candidates.get(r['id']); response=(call or {}).get('response')
            try: candidate=json.loads(response) if isinstance(response,str) else response
            except (ValueError,TypeError): candidate=None
            row['saved_asr_guard_candidate']=candidate
            row['saved_asr_candidate_exact_expected']=bool(candidate and candidate.get('target')==expected)
            row['saved_asr_candidate_status']='saved_api_response_reconstructed; status remains asr_rejected'
            if candidate:
                pi=candidate.get('input',''); pt=candidate.get('target','')
                row['partial_pair_valid']=valid_partial_pair(source,expected or '',pi,pt)
                row['partial_candidate_input']=pi; row['partial_candidate_target']=pt
                row['partial_candidate_expected_prefix']=bool(expected and pt and expected.startswith(pt.rstrip('，,。.!！?？')))
                # The six saved candidates are ASR fragments and follow the production
                # fragment path (no construction plan); every other gate remains live.
                row['partial_candidate_ordinary_min_length']=len(pi.strip())>=2 and len(pt.strip())>=2
                if row['partial_pair_valid']:
                    for n,e in [(1,1),(0,1),(1,0),(0,0)]:
                        from src.utils import numeric_policy as np
                        from src.utils import english_entity_policy as ep
                        sn,se=np.lock_numeric_target, np.english_entities_preserved
                        sm=mod.lock_numeric_target; sme=mod.english_entities_preserved
                        if not n: np.lock_numeric_target=lambda s,t:t; mod.lock_numeric_target=lambda s,t:t
                        if not e: np.english_entities_preserved=lambda s,t:True; mod.english_entities_preserved=lambda s,t:True
                        try: result=mod.lock_training_target(pi,pt,None)
                        finally:
                            np.lock_numeric_target=sn; np.english_entities_preserved=se
                            mod.lock_numeric_target=sm; mod.english_entities_preserved=sme
                        row[f'asr_candidate_boundary_n{n}_e{e}']=result is not None
                else:
                    row['asr_candidate_boundary_groups']='not_reached; valid_partial_pair rejected candidate'
        for n,e in [(1,1),(0,1),(1,0),(0,0)]:
            # Construction branch is unchanged and returns based on exact proof; legacy branch only gets selected monkeypatches.
            from src.utils import numeric_policy as np
            from src.utils import english_entity_policy as ep
            saved_num=np.lock_numeric_target; saved_eng=np.english_entities_preserved
            if not n: np.lock_numeric_target=lambda s,t: t
            if not e: np.english_entities_preserved=lambda s,t: True
            # Also patch imported symbol in numeric_policy module.
            if not e: mod.english_entities_preserved=lambda s,t: True
            try: result=mod.lock_training_target(source,target,plan)
            finally:
                np.lock_numeric_target=saved_num; np.english_entities_preserved=saved_eng
                mod.lock_numeric_target=saved_num; mod.english_entities_preserved=ep.english_entities_preserved
            groups[f'numeric_{n}_english_{e}'].append(result is not None)
        details.append(row)
    mod.lock_numeric_target=original
    out={k:{'evaluated':len(v),'passes':sum(v),'rate':sum(v)/len(v) if v else None} for k,v in groups.items()}
    # Analyze lexical guard outcomes separately on those same candidates; construction bypass means this is diagnostic only.
    from src.utils.english_entity_policy import english_entities_preserved
    legacy=[]
    for r in details:
        if r.get('evaluated') is False: continue
        n=original(r['source'],r['target'])
        en=english_entities_preserved(r['source'],r['target'])
        r['numeric_guard_pass']=n is not None
        r['english_guard_pass']=en
        r['lexical_both_pass']=n is not None and en
        r['construction_gate_result']=r.get('exact_expected') and r.get('construction')=='synthetic-oral-v1'
        if r['construction_gate_result'] and not r['lexical_both_pass']:
            legacy.append(r['id'])
    lexical=collections.Counter()
    for r in details:
        if r.get('evaluated') is False: continue
        lexical[f"numeric_guard_{'pass' if r.get('numeric_guard_pass') else 'fail'}"]+=1
        lexical[f"english_guard_{'pass' if r.get('english_guard_pass') else 'fail'}"]+=1
        lexical[f"both_lexical_{'pass' if r.get('lexical_both_pass') else 'fail'}"]+=1
    return {'groups':out,'rows':details,'construction_bypass_rows_with_lexical_guard_failures':legacy,
            'lexical_guard_diagnostic_counts':dict(lexical),
            'count_by_status':dict(collections.Counter(x.get('status') for x in details)),
            'saved_asr_rejected_responses':sum(bool(x.get('saved_asr_guard_candidate')) for x in details),
            'newly_guard_eligible_candidates_requiring_judge':sum(bool(x.get('partial_pair_valid')) for x in details if x.get('status')=='asr_rejected'),
            'source_files': ['results/refiner_auto_acceptance_300_20261017/attempts.jsonl','results/refiner_auto_acceptance_300_20261017/api_calls.jsonl']}

def analyze1000():
    rs=rows(ROOT/'results/refiner_teacher_scale1000_20261017/attempts.jsonl')
    mod=module_for('refiner_teacher_scale1000_20261017')
    from src.utils.english_entity_policy import english_entities_preserved
    out=[]
    for r in rs:
        oral=r.get('oral') or {}; clean=r.get('clean') or {}
        source=oral.get('oral_text',''); target=clean.get('clean_text',''); plan=(oral.get('meta') or {}).get('generation_plan')
        if not source or not target: continue
        npass=mod.lock_numeric_target(source,target) is not None
        epass=english_entities_preserved(source,target)
        construction=isinstance(plan,dict) and plan.get('construction_version')=='synthetic-oral-v1'
        exact=not construction or target==plan.get('expected_clean')
        both=npass and epass
        out.append({'id':r['id'],'status':r.get('status'),'numeric_guard_pass':npass,'english_guard_pass':epass,
                    'both_pass':both,'numeric_only_fail':not npass and epass,'english_only_fail':npass and not epass,
                    'overlap_fail':not npass and not epass,'construction_versioned':construction,'exact_expected':exact,
                    'audit_pass':r.get('status')=='accepted','source_valid':(r.get('audit') or {}).get('source_valid'),
                    'assembly_stats':r.get('assembly_stats')})
    c=collections.Counter()
    for x in out:
        if x['numeric_only_fail']: c['numeric_only_fail']+=1
        if x['english_only_fail']: c['english_only_fail']+=1
        if x['overlap_fail']: c['overlap_fail']+=1
        if not x['numeric_guard_pass'] and not x['english_guard_pass']: c['numeric_english_overlap_fail']+=1
        if x['both_pass']: c['both_pass']+=1
        if not x['numeric_guard_pass']: c['numeric_guard_fail']+=1
        if not x['english_guard_pass']: c['english_guard_fail']+=1
        c['status:'+str(x['status'])]+=1
    return {'n':len(out),'module_file':str(mod.__file__),'source_snapshot_numeric_policy_sha256':hashlib.sha256((ROOT/'results/refiner_teacher_scale1000_20261017/source_snapshot/pipeline/src/utils/numeric_policy.py').read_bytes()).hexdigest(),
            'current_system_refinement_guard_sha256':hashlib.sha256((ROOT/'system/refinement_guard.py').read_bytes()).hexdigest(),
            'counts':dict(c),'rows':out,'interpretation':'Historical 2026-10-17 snapshot; diagnostic applies its saved numeric policy against the currently present system dependencies, whose historical hashes were not captured. Saved attempt statuses are original run observations; this is not full old-environment reproduction.'}

if __name__=='__main__':
    a=analyze300(); dump('offline_300.json',a)
    b=analyze1000(); dump('historical_1000.json',b)
    print(json.dumps({'300':a['groups'],'legacy_counts':a['count_by_status'],'lexical':a['lexical_guard_diagnostic_counts'],
       'saved_asr_rejections':[(x['id'],x.get('partial_pair_valid'),x.get('asr_candidate_boundary_groups')) for x in a['rows'] if x.get('status')=='asr_rejected'],
       '1000':{'n_with_saved_source_target':b['n'],'module_file':b['module_file'],'counts':b['counts']}},ensure_ascii=False,indent=2))
