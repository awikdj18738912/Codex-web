"""Cheap window-only prototypes, selected using executed development arms.

Every action combination used here must have an actually executed arm. No sum
of isolated gains, learned local veto or unseen-test threshold selection.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import pickle
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d
from experiments.scripts.gate_terminal_gain_pilot import view_id
from system.weak_refinement_gate import input_text


def feature_text(value):
    description,target,context=input_text(value)
    return description+context+'\n'+target


def executed_arm(row,mask):
    if not mask:return row['native']
    if mask==set(row['counterfactuals']):return row['skip_all']
    if len(mask)==1:return row['counterfactuals'][next(iter(mask))]
    raise ValueError('This joint intervention was not executed')


def evaluate(rows,scores,threshold,gate_ms):
    refined=skipped=benefit_kept=benefit_total=quality_losses=0
    chars=distance=exact=calls=ppl=forward=tokens=0;model_time=0;per_cohort=defaultdict(lambda:dict(benefit=0,kept=0))
    for row in rows:
        score=scores[view_id(row)]
        mask={key for key,p in score.items() if p>=threshold}
        chosen=executed_arm(row,mask);cost=chosen['cost']
        skipped+=len(mask);refined+=len(score)-len(mask)
        chars+=row['intended_chars'];distance+=chosen['intended_distance'];exact+=chosen['intended_distance']==0
        quality_losses+=chosen['intended_distance']>row['native']['intended_distance']
        calls+=cost['refiner_calls'];ppl+=cost['plausibility_calls'];tokens+=cost['refiner_token_positions'];forward+=cost['forward_passes']
        model_time+=cost['refiner_wall_ms']+cost['plausibility_wall_ms']
        for effect in row['effects']:
            if effect['label']=='benefit':
                benefit_total+=1;benefit_kept+=effect['window_key'] not in mask
                cohort=row.get('cohort','unknown')
                per_cohort[cohort]['benefit']+=1;per_cohort[cohort]['kept']+=effect['window_key'] not in mask
    gate_time=gate_ms*(refined+skipped) if threshold<=1 else 0
    return dict(recording_views=len(rows),exact_script=exact,intended_micro_cer=distance/max(1,chars),
        intended_distance=distance,degraded_views=quality_losses,beneficial_windows=benefit_total,
        beneficial_windows_kept=benefit_kept,by_cohort=dict(per_cohort),
        window_vetoes=skipped,window_retained=refined,refiner_calls=calls,plausibility_calls=ppl,
        token_positions=tokens,forward_passes=forward,recorded_model_ms=model_time,
        estimated_gate_ms=gate_time,estimated_serial_model_plus_gate_ms=model_time+gate_time)


def train(root):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    folder=root/'dataset';train_rows=d.read_rows(folder/'train.jsonl');dev_rows=d.read_rows(folder/'dev.jsonl')
    recordings=json.loads((folder/'recordings.json').read_text());source_lookup={}
    for split in ('train','dev'):
        for row in d.read_rows(root/f'{split}_collection/sources.jsonl'):source_lookup[view_id(row)]=row
        for row in recordings[split]:row['cohort']=source_lookup[view_id(row)]['round3_cohort']
    output=root/'linear_training'
    if output.exists():raise FileExistsError('Never overwrite a previous training run')
    output.mkdir();(output/'executed_code').mkdir()
    (output/'executed_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if len({r['recording_group'] for r in train_rows if r['label']=='benefit'})<10:raise ValueError('Too few beneficial training groups')
    if not any(r['label']=='benefit' for r in dev_rows):raise ValueError('No beneficial development windows')
    groups=Counter(r['recording_group'] for r in train_rows)
    template_groups=defaultdict(set)
    for r in train_rows:template_groups[r['split_group']].add(r['recording_group'])
    weights=[1/(groups[r['recording_group']]*len(template_groups[r['split_group']])) for r in train_rows]
    mean=sum(weights)/len(weights);weights=[w/mean for w in weights]
    configs={};all_predictions={}
    native_scores={view_id(r):{e['window_key']:0.0 for e in r['effects']} for r in recordings['dev']}
    native=evaluate(recordings['dev'],native_scores,1.01,0)
    for objective in ('terminal_quality','conservative_cost'):
        targets=[('refine_required' if r['label']=='benefit' else 'skip_allowed')
                 if objective=='terminal_quality' else r['action_target'] for r in train_rows]
        model=Pipeline([('text',TfidfVectorizer(analyzer='char',ngram_range=(1,3),sublinear_tf=True,min_df=1)),
                        ('classifier',LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=20261012))])
        started=time.perf_counter();model.fit([feature_text(r['input']) for r in train_rows],targets,
                                            classifier__sample_weight=weights)
        column=list(model.classes_).index('skip_allowed');scores={};lat=[]
        for row in recordings['dev']:
            local={}
            for effect in row['effects']:
                value=feature_text(effect['input']);start=time.perf_counter()
                local[effect['window_key']]=float(model.predict_proba([value])[0,column])
                lat.append((time.perf_counter()-start)*1000)
            scores[view_id(row)]=local
        # Include one cold forward in this small-sample timing estimate.
        average=statistics.mean(lat) if lat else 0;grid=[];best=None
        for threshold in [0.,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99,1.01]:
            metrics=evaluate(recordings['dev'],scores,threshold,average);grid.append(dict(threshold=threshold,metrics=metrics))
            valid=(metrics['degraded_views']==0 and metrics['beneficial_windows_kept']==metrics['beneficial_windows'])
            if valid and (best is None or metrics['estimated_serial_model_plus_gate_ms']<best['metrics']['estimated_serial_model_plus_gate_ms']):
                best=dict(threshold=threshold,metrics=metrics)
        if best is None:raise RuntimeError('Even native fallback failed retention')
        name=output/objective;name.mkdir()
        with (name/'model.pkl').open('wb') as stream:pickle.dump(model,stream)
        record=dict(objective=objective,classes=list(model.classes_),training_examples=len(train_rows),
            target_counts=dict(Counter(targets)),features='pre-action window text only, char TF-IDF 1-3grams',
            group_and_template_weights=True,fit_seconds=time.perf_counter()-started,
            cpu_score_mean_ms=average,cpu_score_p95_ms=sorted(lat)[max(0,int(.95*len(lat))-1)] if lat else 0,
            threshold=best['threshold'],development=best['metrics'],threshold_grid=grid,
            model_sha256=c.file_hash(name/'model.pkl'),calibrated=False,online_enabled=False,
            source_dataset_manifest_sha256=c.file_hash(folder/'manifest.json'))
        c.write_json(name/'metadata.json',record);configs[objective]=record;all_predictions[objective]=scores
    feasible=[(name,record) for name,record in configs.items()
              if record['development']['estimated_serial_model_plus_gate_ms']<native['estimated_serial_model_plus_gate_ms']]
    primary=min(feasible,key=lambda item:item[1]['development']['estimated_serial_model_plus_gate_ms'])[0] if feasible else 'native'
    selection=dict(event='terminal_gain_development_selection_no_test_opened',primary=primary,variants=configs,native=native,
        criterion='zero development record degradation, all beneficial windows retained, minimum recorded model plus CPU scoring time',
        data_role='reused_development_unverified_scripts_not_independent_test',test_opened=False,online_enabled=False,
        runtime_protocol_sha256=json.loads((root/'manifest.json').read_text())['runtime_protocol_sha256'])
    c.write_json(output/'development_predictions.json',all_predictions)
    c.write_json(output/'selection.json',selection)
    print(json.dumps(selection,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();train(a.root.resolve())
