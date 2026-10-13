"""Second weak gate trial: train augmentation and validation-constrained native veto.

fit never opens the new test pack; evaluate requires the exact frozen selection
that authorized test audio generation. Previously read internal test is excluded.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import json
from importlib.metadata import version
import math
from pathlib import Path
import sys
import time
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.scripts import synthetic_gate_data as data, collect_gate_pairs as collector
from experiments.scripts.train_weak_refinement_gate import (fit_encoder,fit_linear,choose_thresholds,
    classification,model_probabilities,policy_metrics,benchmark,write_json,write_rows)
from system.weak_refinement_gate import (LABELS,TASKS,WeakRefinementGate,input_text,
    select_action,select_veto_action)

NEUTRAL_GRID=(0.6,0.7,0.8,0.9,0.95,0.99,1.01)
HARM_GRID=(0.2,0.4,0.6,0.8,0.95,1.01)


def signature(row):
    strings=input_text(row['input'])
    return data.digest([''.join(unicodedata.normalize('NFKC',s).casefold().split()) for s in strings])


def check_partition_links(partitions):
    """Reject shared template, recording or actual encoder-input links."""
    for kind, key in [('template','split_group'),('recording','recording_group'),('case','case_id')]:
        seen=defaultdict(set)
        for partition,rows in partitions.items():
            for row in rows: seen[row[key]].add(partition)
        if any(len(v)>1 for v in seen.values()):
            raise ValueError('Cross-partition '+kind+' leakage')
    seen=defaultdict(set)
    for partition,rows in partitions.items():
        for row in rows: seen[signature(row)].add(partition)
    if any(len(v)>1 for v in seen.values()): raise ValueError('Cross-partition encoder-input leakage')


def teacher_signature(path):
    protocol=collector.verify_frozen(path)
    cfg=json.loads((path/'run_configuration.json').read_text())
    tree=ast.parse((path/'frozen/system/web_app.py').read_text())
    bodies={n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    required=['refinement_gate.py','refinement_guard.py','refinement_protocol.py','protection.py',
              'numeric_normalizer.py','zh_itn_bridge.py','window_refinement.py','entity_pipeline.py',
              'entity_store.py','entity_matcher.py']
    return dict(refiner_weights=cfg['weights_sha256'],refiner_config=cfg['config_sha256'],
                dtype=cfg['dtype'],operation_settings=protocol['settings'],
                entity_snapshot=data.file_hash(path/'entities_snapshot.db'),
                itn=data.file_hash(path/'native/libzh_itn_bridge.so'),
                closures={k:data.digest(bodies[k]) for k in ['refine_update','review_final_repetition_text']},
                dependencies={k:protocol['frozen_hashes']['frozen/system/'+k] for k in required})


def load_records(source, split, allowed=None):
    from system.zh_itn_bridge import ZhITNNormalizer
    path=source/f'weak_gate_labels_{split}_all.jsonl'
    rows=data.read_rows(path)
    if allowed is not None: rows=[r for r in rows if r['case_id'] in allowed]
    if not rows or len({r['case_id'] for r in rows})!=len(rows): raise ValueError('Missing/duplicate labels')
    if allowed is not None and {r['case_id'] for r in rows}!=allowed: raise ValueError('Incomplete selected labels')
    if any(r['split']!=split or r['human_gold'] or r['audio_script_verified'] for r in rows):
        raise ValueError('Unexpected supervision provenance')
    sources={}; outcomes={}; fingerprints={}
    for row in rows:
        name=row['pair_set']
        if name not in sources:
            folder=source/('pairs_full' if name=='window_full' else 'pairs_review_full')
            sources[name]=(folder,{r['case_id']:r for r in data.read_rows(folder/'cases.jsonl')},
                           {r['case_id']:r for r in data.read_rows(folder/'pairs.jsonl')},
                           ZhITNNormalizer(folder/'native/libzh_itn_bridge.so'))
            fingerprints[name]=teacher_signature(folder)
        folder,cases,pairs,norm=sources[name]
        case,pair=cases[row['case_id']],pairs[row['case_id']]
        meta=case['synthetic_metadata']
        if meta['synthetic_split']!=split or meta['template_group']!=row['split_group']:
            raise ValueError('Case metadata mismatch')
        expected=dict(task_kind=case['task_kind'],domain=case['domain'],data_kind=case['data_kind'])
        if case['task_kind']=='window_refine':
            expected.update(raw_text=case['raw_text'],baseline_text=case['baseline_text'],
                            available_context=case['context_available_before'],window_chunk_count=case['window_chunk_count'])
        else:
            expected.update(baseline_text=case['candidate']['context'],available_context=case['candidate']['context'],
                            candidate_local_start=case['candidate_local_start'],
                            candidate={k:case['candidate'][k] for k in ['source','target','kind']})
        if row['input']!=expected: raise ValueError('Model inputs differ from frozen pre-operation state')
        baseline=pair['arms']['rules']['comparison_text']; always=pair['arms']['always']['comparison_text']
        weak,_=data.weak_label(baseline,always,meta['intended_text'],lambda s:norm.normalize(s).text)
        if weak!=row['label']: raise ValueError('Weak label no longer reproducible')
        row['native_action']=case['original_gate']['action'] if row['input']['task_kind']=='window_refine' else 'refine'
        actual=bool(pair['arms']['current']['logical_calls'] or pair['arms']['current']['plausibility_calls'])
        if actual!=(row['native_action']=='refine'):
            raise ValueError('Native pre-operation action does not explain recorded execution')
        if not actual and pair['arms']['current']['comparison_text']!=baseline:
            raise ValueError('Native KEEP output differs from shared rules baseline')
        normalize=lambda s:data.lexical(norm.normalize(s).text)
        target,spoken=normalize(meta['intended_text']),normalize(meta['spoken_text'])
        arms={}
        for arm,a in pair['arms'].items():
            text=a['comparison_text']; normalized=normalize(text); calls=a['logical_calls']+a['plausibility_calls']
            if any(c.get('error_type') for c in calls) or a['session_stats'].get('failed_call_count'):
                raise ValueError('Failed teacher operation')
            effect,_=data.weak_label(baseline,text,meta['intended_text'],lambda s:norm.normalize(s).text)
            arms[arm]=dict(intended_distance=collector.edit_distance(target,normalized),
                spoken_distance=collector.edit_distance(spoken,normalized),exact_intended=target==normalized,
                effect=effect,logical_refiner_calls=len(a['logical_calls']),
                logical_plausibility_calls=len(a['plausibility_calls']),
                operation_cost_ms=sum(c['wall_ms'] for c in calls),executed=bool(calls),changed=text!=baseline)
        outcomes[row['case_id']]=dict(intended_chars=len(target),spoken_chars=len(spoken),arms=arms)
    return rows,outcomes,fingerprints


def counts(rows):
    return dict(cases=len(rows),eligible=sum(r['eligible_for_weak_pretraining'] for r in rows),
                labels=dict(Counter(r['label'] for r in rows)),templates=len({r['split_group'] for r in rows}),
                recordings=len({r['recording_group'] for r in rows}),
                by_task={t:dict(labels=dict(Counter(r['label'] for r in rows if r['input']['task_kind']==t)),
                    harm_templates=len({r['split_group'] for r in rows if r['input']['task_kind']==t and r['label']=='harm'})) for t in TASKS})


def choose_veto(rows,probabilities,outcomes,minimum_retention=0.95):
    thresholds,evidence={},{}
    for task in TASKS:
        subset=[(r,p) for r,p in zip(rows,probabilities) if r['input']['task_kind']==task]
        native_b=sum(outcomes[r['case_id']]['arms']['current']['effect']=='benefit' for r,_ in subset)
        native_h=sum(outcomes[r['case_id']]['arms']['current']['effect']=='harm' for r,_ in subset)
        required=math.ceil(minimum_retention*native_b)
        best=None
        for neutral in NEUTRAL_GRID:
            for harm in HARM_GRID:
                limits=dict(neutral=neutral,harm=harm)
                arms=[outcomes[r['case_id']]['arms']['current' if select_veto_action(p,limits,r['native_action'])=='refine' else 'rules']
                      for r,p in subset]
                benefit=sum(a['effect']=='benefit' for a in arms); risk=sum(a['effect']=='harm' for a in arms)
                if benefit<required or risk>native_h: continue
                cost=sum(a['operation_cost_ms'] for a in arms)
                rank=(cost,-benefit,risk,-neutral,-harm)
                if best is None or rank<best[0]:
                    best=rank,limits,dict(native_benefit=native_b,required_benefit=required,retained_benefit=benefit,
                        native_harm=native_h,selected_harm=risk,recorded_operation_cost_ms=cost,
                        harm_examples=sum(r['label']=='harm' for r,_ in subset),
                        weak_benefit_retention=benefit/max(1,native_b))
        if best is None: raise ValueError('Native fallback itself must be a feasible candidate')
        thresholds[task],evidence[task]=best[1],best[2]
    return thresholds,dict(by_task=evidence,minimum_retention=minimum_retention,
        constraint='per_task_retain_at_least_95pct_current_weak_benefit_and_no_added_weak_harm',
        selection='minimum_recorded_operation_cost_after_constraints',real_risk_calibrated=False)


def fit(args):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    out=args.output
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a fresh round-two output')
    if json.loads((args.dev_data/'job_state.json').read_text())['status']!='complete':
        raise ValueError('Development audio/labels must be complete before fitting')
    old_train_ids={r['case_id'] for r in data.read_rows(args.old_experiment/'train.jsonl')}
    old_val_ids={r['case_id'] for r in data.read_rows(args.old_experiment/'validation.jsonl')}
    old_test_ids={r['case_id'] for r in data.read_rows(args.old_experiment/'test.jsonl')}
    train,ot,fp1=load_records(args.old_source,'train',old_train_ids)
    validation,ov,fp2=load_records(args.old_source,'train',old_val_ids)
    augmented,oa,fp3=load_records(args.new_train,'train')
    development,od,fp4=load_records(args.dev_data,'dev')
    risk_dev,orisk,fp5=load_records(args.risk_dev_data,'dev')
    train+=augmented; validation+=development+risk_dev
    if any(r['case_id'] in old_test_ids for r in train+validation): raise ValueError('Old internal test leaked')
    check_partition_links(dict(train=train,validation=validation))
    teachers=[v for fp in [fp1,fp2,fp3,fp4,fp5] for v in fp.values()]
    if any(v!=teachers[0] for v in teachers): raise ValueError('Teacher configuration changed across sources')
    outcomes={**ot,**ov,**oa,**od,**orisk}
    out.mkdir(parents=True)
    for name,rows in [('train',train),('validation',validation)]: write_rows(out/f'{name}.jsonl',rows)
    write_json(out/'validation_outcomes.json',ov|od|orisk)
    manifest=dict(train=counts(train),validation=counts(validation),old_internal_test_excluded=True,
        original_source_dev_test_used=False,new_source_dev_used_for_validation=True,new_source_test_generated=False,
        teacher=teachers[0],source_paths=dict(old_experiment=str(args.old_experiment),old_source=str(args.old_source),
                                           new_train=str(args.new_train),dev_data=str(args.dev_data),risk_dev_data=str(args.risk_dev_data)),
        source_hashes={str(p):data.file_hash(p) for p in [args.old_experiment/'train.jsonl',args.old_experiment/'validation.jsonl',
                      args.new_train/'weak_gate_labels_train_all.jsonl',args.dev_data/'weak_gate_labels_dev_all.jsonl',args.risk_dev_data/'weak_gate_labels_dev_all.jsonl']},
        fit_script_sha256=data.file_hash(Path(__file__)),runtime_sha256=data.file_hash(ROOT/'system/weak_refinement_gate.py'),
        encoder_weights_sha256=data.file_hash(args.encoder/'pytorch_model.bin'),
        encoder_config_sha256=data.file_hash(args.encoder/'config.json'),torch_version=torch.__version__,
        transformers_version=version('transformers'),device=args.device,batch_size=args.batch_size,
        baseline_weights_sha256=data.file_hash(args.old_experiment/'encoder_weighted/gate.pt'),
        baseline_metadata_sha256=data.file_hash(args.old_experiment/'encoder_weighted/metadata.json'),
        epochs=args.epochs,train_only_class_weighting=True,neutral_grid=NEUTRAL_GRID,harm_grid=HARM_GRID,
        minimum_benefit_retention=0.95,declared_models=['linear_tfidf','encoder_unweighted','encoder_weighted'],
        declared_routes=['direct','native_veto'],human_gold=0,uncertain_training='excluded',uncertain_evaluation='included')
    write_json(out/'dataset_manifest.json',manifest)
    for filename in ['experiments/scripts/train_gate_round2.py','experiments/scripts/train_weak_refinement_gate.py','system/weak_refinement_gate.py']:
        snapshot=out/'executed_code'/filename
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes((ROOT/filename).read_bytes())
    print('ROUND2_DATA '+json.dumps(dict(train=manifest['train'],validation=manifest['validation']),ensure_ascii=False),flush=True)
    fit_rows=[r for r in train if r['eligible_for_weak_pretraining']]
    val_eligible=[r for r in validation if r['eligible_for_weak_pretraining']]
    pipeline,linear_prob,text=fit_linear(fit_rows,validation,out)
    variants={}
    for weighted in [None,False,True]:
        if weighted is None:
            name='linear_tfidf'; probs=linear_prob; training_info={}; runtime=None
        else:
            name,best_probs,training_info=fit_encoder(fit_rows,val_eligible,out,args,weighted)
            write_json(out/name/'metadata.json',dict(**training_info,thresholds={t:dict(benefit=1.01,harm=0) for t in TASKS}))
            runtime=WeakRefinementGate(out/name)
            probs=model_probabilities(runtime.model,runtime.tokenizer,validation,'cpu',args.max_tokens)
            reloaded=[p for r,p in zip(validation,probs) if r['eligible_for_weak_pretraining']]
            if not np.allclose(reloaded,best_probs,atol=1e-4): raise ValueError('GPU/CPU checkpoint mismatch')
        direct,direct_evidence=choose_thresholds(validation,probs)
        veto,veto_evidence=choose_veto(validation,probs,outcomes)
        if runtime is not None:
            metadata=dict(**training_info,thresholds=direct,veto_thresholds=veto,recommended_route='native_veto')
            write_json(out/name/'metadata.json',metadata); runtime.metadata=metadata
            score=lambda r:runtime.decide_with_fallback(r['input'],r['native_action'])
        else:
            score=lambda r:None if r['native_action']!='refine' else pipeline.predict_proba([text(r)])
        latency=benchmark(score,validation)
        actions={r['case_id']:select_veto_action(p,veto[r['input']['task_kind']],r['native_action']) for r,p in zip(validation,probs)}
        overhead=sum(latency['by_task'][r['input']['task_kind']]['mean_ms'] for r in validation)
        metrics=policy_metrics(validation,outcomes,lambda r:'current' if actions[r['case_id']]=='refine' else 'rules',overhead)
        variants[name]=dict(direct_thresholds=direct,direct_evidence=direct_evidence,veto_thresholds=veto,
            veto_evidence=veto_evidence,validation_native_veto=metrics,validation_classification=classification(validation,probs),
            validation_latency=latency,validation_loss=training_info.get('validation_loss'))
        if runtime is not None: del runtime
    primary=min(['encoder_unweighted','encoder_weighted'],key=lambda n:variants[n]['validation_native_veto']['estimated_serial_cost_ms'])
    selection=dict(event='round2_policy_frozen_before_test_generation',policy_freeze_unix=time.time(),primary_model=primary,
        primary_route='native_veto',variants=variants,selection_partition='old_validation_plus_new_dev',
        test_predictions_generated_at_selection=False,old_internal_test_used_for_selection=False,
        real_risk_calibrated=False,checkpoint_hashes={n:data.file_hash(out/n/'gate.pt') for n in ['encoder_unweighted','encoder_weighted']},
        metadata_hashes={n:data.file_hash(out/n/'metadata.json') for n in ['encoder_unweighted','encoder_weighted']},
        linear_hash=data.file_hash(out/'linear_tfidf/model.joblib'))
    write_json(out/'selection.json',selection)
    write_json(out/'fit_completed.json',dict(selection_sha256=data.file_hash(out/'selection.json'),new_test_not_opened=True))
    print('ROUND2_FROZEN '+json.dumps(dict(primary_model=primary,variants=variants),ensure_ascii=False),flush=True)


def evaluate(args):
    import joblib
    import numpy as np
    import torch
    torch.set_num_threads(2)
    out=args.output
    if (out/'test_report.json').exists(): raise FileExistsError('Test already evaluated; no threshold reselection')
    selected=json.loads((out/'selection.json').read_text())
    pack=json.loads((args.test_data/'manifest.json').read_text())
    if pack['policy_sha256_before_test_generation']!=data.file_hash(out/'selection.json'):
        raise ValueError('Test was not staged under this frozen policy')
    job=json.loads((args.test_data/'job_state.json').read_text())
    if job['status']!='complete' or job['started_unix']<=selected['policy_freeze_unix']:
        raise ValueError('Test generation preceded selection freeze or is incomplete')
    for n,expected in selected['checkpoint_hashes'].items():
        if data.file_hash(out/n/'gate.pt')!=expected or data.file_hash(out/n/'metadata.json')!=selected['metadata_hashes'][n]:
            raise ValueError('Frozen checkpoint changed')
    if data.file_hash(out/'linear_tfidf/model.joblib')!=selected['linear_hash']: raise ValueError('Linear checkpoint changed')
    rows,outcomes,fps=load_records(args.test_data,'test')
    train=data.read_rows(out/'train.jsonl'); validation=data.read_rows(out/'validation.jsonl')
    check_partition_links(dict(train=train,validation=validation,test=rows))
    manifest=json.loads((out/'dataset_manifest.json').read_text())
    if (data.file_hash(args.old_experiment/'encoder_weighted/gate.pt')!=manifest['baseline_weights_sha256']
            or data.file_hash(args.old_experiment/'encoder_weighted/metadata.json')!=manifest['baseline_metadata_sha256']):
        raise ValueError('Previous frozen baseline changed')
    if any(fp!=manifest['teacher'] for fp in fps.values()): raise ValueError('Test teacher changed')
    policies={n:policy_metrics(rows,outcomes,lambda r,n=n:n) for n in ['rules','current','always']}
    predictions=[]; classify={}; latency={}
    for name in ['linear_tfidf','encoder_unweighted','encoder_weighted','previous_frozen']:
        if name=='linear_tfidf':
            pipeline=joblib.load(out/name/'model.joblib'); classes=list(pipeline.named_steps['classifier'].classes_)
            order=[classes.index(k) for k in LABELS]
            # Same representation as fit_linear: description+context+[SEP]+target.
            def text(r):
                first,target,context=input_text(r['input']);return first+context+'[SEP]'+target
            probs=pipeline.predict_proba([text(r) for r in rows])[:,order].tolist(); runtime=None
        else:
            path=args.old_experiment/'encoder_weighted' if name=='previous_frozen' else out/name
            runtime=WeakRefinementGate(path)
            probs=model_probabilities(runtime.model,runtime.tokenizer,rows,'cpu',args.max_tokens)
        settings=(dict(direct_thresholds=runtime.metadata['thresholds']) if name=='previous_frozen' else selected['variants'][name])
        classify[name]=classification(rows,probs)
        for route in ['direct'] if name=='previous_frozen' else ['direct','native_veto']:
            key=name+'_'+route
            if route=='direct':
                actions={r['case_id']:select_action(p,settings['direct_thresholds'][r['input']['task_kind']]) for r,p in zip(rows,probs)}
                arm=lambda r:'always' if actions[r['case_id']]=='refine' else 'rules'
                score=(lambda r:pipeline.predict_proba([text(r)])) if runtime is None else (lambda r:runtime.decide(r['input']))
            else:
                actions={r['case_id']:select_veto_action(p,settings['veto_thresholds'][r['input']['task_kind']],r['native_action']) for r,p in zip(rows,probs)}
                arm=lambda r:'current' if actions[r['case_id']]=='refine' else 'rules'
                score=(lambda r:None if r['native_action']!='refine' else pipeline.predict_proba([text(r)])) if runtime is None else (lambda r:runtime.decide_with_fallback(r['input'],r['native_action']))
            timing=benchmark(score,rows);latency[key]=timing
            overhead=sum(timing['by_task'][r['input']['task_kind']]['mean_ms'] for r in rows)
            metrics=policy_metrics(rows,outcomes,arm,overhead)
            metrics['by_task']={t:policy_metrics([r for r in rows if r['input']['task_kind']==t],outcomes,arm) for t in TASKS}
            policies[key]=metrics
            predictions.extend(dict(case_id=r['case_id'],model=name,route=route,action=actions[r['case_id']],
                                probabilities=dict(zip(LABELS,p)),weak_label=r['label']) for r,p in zip(rows,probs))
        if runtime is not None: del runtime
    report=dict(primary_policy=selected['primary_model']+'_native_veto',counts=counts(rows),policies=policies,
        classification=classify,latency=latency,validation_selection=selected,old_internal_test_excluded=True,
        new_test_generated_after_freeze=True,policy_retuned_after_test=False,human_gold=0,online_enabled=False,
        limits=['Unverified synthetic weak targets; no real-speech risk guarantee',
                'Per-decision counterfactual replay, not sequential full-session quality',
                'Operation costs are saved timings plus CPU estimate, not streaming latency'])
    write_rows(out/'test_predictions.jsonl',predictions);write_json(out/'test_report.json',report)
    lines=['# 第二轮：保守否决门控试验','','旧内部测试不参与本轮选择；新扩展 test 在模型和阈值冻结之后才生成、解码、配对。模型只否决原门控准备执行的操作，KEEP/DEFER 不评分，未知输入保留原动作。','',
           f"主策略：{report['primary_policy']}。新测试配对数 {len(rows)}，有效标签 {report['counts']['eligible']}。",'',
           '| 策略 | 精修调用 | 捕获 benefit | weak harm | 目标精确率 | 目标 CER | 估算串行 ms |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in policies.items(): lines.append(f"| {name} | {m['logical_refiner_calls']} | {m['captured_benefit']} | {m['weak_harm']} | {m['exact_intended_fraction']:.3f} | {m['intended_micro_cer']:.4f} | {m['estimated_serial_cost_ms']:.1f} |")
    lines+=['','验证约束：每任务至少保留现有门控95%的 exact weak benefit，weak harm 不增加；满足约束后最小化验证操作成本。测试表现不保证继续满足该约束，不能据测试重选阈值。uncertain 保留在评估分母，不参与拟合。','',
            '训练/开发/测试每任务标签与独立 harm 模板数见 dataset_manifest.json、test_report.json。所有音频未经人工核验，模型概率未校准，线上继续使用原门控。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'completed_test.json',dict(report_sha256=data.file_hash(out/'test_report.json'),selection_sha256=data.file_hash(out/'selection.json')))
    print('ROUND2_TEST '+json.dumps(dict(primary=report['primary_policy'],counts=report['counts'],policies=policies),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','evaluate'])
    p.add_argument('--output',type=Path,default=ROOT/'results/gate_round2_20261010/training')
    p.add_argument('--old-experiment',type=Path,default=ROOT/'results/learned_gate_weak_prototype_20261010')
    p.add_argument('--old-source',type=Path,default=ROOT/'results/qwen_tts_asr_gpu_pilot_20261010')
    p.add_argument('--new-train',type=Path,default=ROOT/'results/gate_template_expansion_20261010')
    p.add_argument('--dev-data',type=Path,default=ROOT/'results/gate_round2_20261010/dev_data')
    p.add_argument('--risk-dev-data',type=Path,default=ROOT/'results/gate_round2_20261010/risk_dev_data')
    p.add_argument('--test-data',type=Path,default=ROOT/'results/gate_round2_20261010/test_data')
    p.add_argument('--encoder',type=Path,default=ROOT/'models/entity_ranker/encoder')
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--max-tokens',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261012)
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or not 32<=a.max_tokens<=512:p.error('Invalid training settings')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    (fit if a.mode=='fit' else evaluate)(a)


if __name__=='__main__': main()
