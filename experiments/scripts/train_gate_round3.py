"""Third trial: strict/operation supervision and high-recall binary ablation.

No test source is opened. All policies retain native KEEP/DEFER, and selection
uses reused development only. Script-derived effects are not calibrated risk.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c
from experiments.scripts.train_gate_round2 import choose_veto,signature
from experiments.scripts.train_weak_refinement_gate import policy_metrics,benchmark
from system.weak_refinement_gate import (LABELS,GAIN_LABELS,TASKS,INPUT_VERSION,WeakRefinementGate,
    build_network,make_tokenizer,encode_input,collate_inputs,select_veto_action,select_gain_action)


def gain_thresholds(rows,probabilities,outcomes):
    policies={};evidence={}
    for task in TASKS:
        subset=[(r,p) for r,p in zip(rows,probabilities) if r['input']['task_kind']==task]
        if not subset:
            policies[task]=0;evidence[task]=dict(by_cohort={},cost_ms=0,selected=0,scope='no_validation_task_native_preserved');continue
        best=None
        for threshold in [0,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]:
            selected=[r for r,p in subset if select_gain_action(p,threshold,r['native_action'])=='refine']
            cohorts={r['validation_cohort'] for r,_ in subset};retention={};valid=True
            for cohort in cohorts:
                native=defaultdict(set)
                for r,_ in subset:
                    if r['validation_cohort']==cohort and outcomes[r['case_id']]['arms']['current']['effect']=='benefit':
                        native[r.get('recording_group',r['case_id'])].add(r['case_id'])
                chosen={r['case_id'] for r in selected}
                retained=sum(ids.issubset(chosen) for ids in native.values())
                required=math.ceil(0.95*len(native));retention[cohort]=dict(native=len(native),retained=retained,required=required,unit='recording_group_all_beneficial_views')
                valid &= retained>=required
            if not valid:continue
            cost=sum(outcomes[r['case_id']]['arms']['current']['operation_cost_ms'] for r in selected)
            rank=(-cost,threshold)
            if best is None or rank>best[0]:best=(rank,threshold,dict(by_cohort=retention,cost_ms=cost,selected=len(selected)))
        if best is None:raise ValueError('No admissible gain threshold')
        policies[task]=best[1];evidence[task]=best[2]
    return policies,evidence


def ternary_thresholds(rows,probabilities,outcomes):
    policies={};evidence={}
    for task in TASKS:
        subset=[(r,p) for r,p in zip(rows,probabilities) if r['input']['task_kind']==task];best=None
        for neutral in [0.6,0.7,0.8,0.9,0.95,0.99,1.01]:
            for harm in [0.2,0.4,0.6,0.8,0.95,1.01]:
                limit=dict(neutral=neutral,harm=harm)
                selected=[r for r,p in subset if select_veto_action(p,limit,r['native_action'])=='refine']
                chosen={r['case_id'] for r in selected};retention={};valid=True
                for cohort in {r['validation_cohort'] for r,_ in subset}:
                    native=defaultdict(set)
                    for r,_ in subset:
                        if r['validation_cohort']==cohort and outcomes[r['case_id']]['arms']['current']['effect']=='benefit':
                            native[r.get('recording_group',r['case_id'])].add(r['case_id'])
                    retained=sum(ids.issubset(chosen) for ids in native.values());required=math.ceil(.95*len(native))
                    retention[cohort]=dict(native=len(native),retained=retained,required=required,unit='recording_group_all_beneficial_views')
                    valid &= retained>=required
                if not valid:continue
                cost=sum(outcomes[r['case_id']]['arms']['current']['operation_cost_ms'] for r in selected)
                rank=(-cost,neutral,harm)
                if best is None or rank>best[0]:best=(rank,limit,dict(by_cohort=retention,cost_ms=cost,selected=len(selected)))
        if best is None:raise ValueError('No admissible ternary policy')
        policies[task],evidence[task]=best[1],best[2]
    return policies,evidence


def probabilities(model,tokenizer,rows,device,max_tokens):
    import torch
    values=[None]*len(rows);supported=[]
    for i,r in enumerate(rows):
        try:supported.append((i,encode_input(tokenizer,r['input'],max_tokens)))
        except ValueError:continue
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(supported),32):
            items=supported[start:start+32];batch={k:v.to(device) for k,v in collate_inputs(tokenizer,[e for _,e in items]).items()}
            for (i,_),p in zip(items,model(**batch).softmax(-1).cpu().tolist()):values[i]=p
    return values


def fit_variant(train,dev,outcomes,output,args,name):
    import numpy as np
    import torch
    binary=name=='encoder_binary_operation';strict=name=='encoder_ternary_strict'
    labels=GAIN_LABELS if binary else LABELS;directory=output/name;directory.mkdir()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer=make_tokenizer(args.encoder);model=build_network(args.encoder,num_labels=len(labels)).to(args.device)
    input_labels=defaultdict(set)
    for r in train:
        label=r['strict_label'] if strict else r['label']
        if label!='uncertain' and r['native_action']=='refine':input_labels[signature(r)].add('benefit' if label=='benefit' else 'not_benefit' if binary else label)
    conflicting={key for key,values in input_labels.items() if len(values)>1}
    fit=[];encoded=[];targets=[];unsupported=[]
    for row in train:
        label=row['strict_label'] if strict else row['label']
        if label=='uncertain' or row['native_action']!='refine' or signature(row) in conflicting:continue
        try:encoded.append(encode_input(tokenizer,row['input'],args.max_tokens))
        except ValueError:unsupported.append(row['case_id']);continue
        fit.append(row);targets.append(0 if binary and label=='benefit' else 1 if binary else labels.index(label))
    if not fit or len(set(targets))<2:raise ValueError('Insufficient learnable classes')
    counts=Counter(targets);class_weights=torch.tensor([(len(fit)/(len(labels)*max(1,counts[i])))**0.5 for i in range(len(labels))],device=args.device)
    class_weights[0]*=1.5;class_weights/=class_weights.mean()
    groups=Counter((r['recording_group'],r['input']['task_kind']) for r in fit)
    sample_weights=torch.tensor([1/groups[(r['recording_group'],r['input']['task_kind'])] for r in fit],device=args.device)
    sample_weights/=sample_weights.mean();targets=torch.tensor(targets,dtype=torch.long)
    criterion=torch.nn.CrossEntropyLoss(weight=class_weights,reduction='none')
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=2e-5),
        dict(params=list(model.head.parameters())+list(model.task_embedding.parameters()),lr=5e-4)],weight_decay=0.01)
    val=[]
    for row in dev:
        label=row['strict_label'] if strict else row['label']
        if label=='uncertain' or row['native_action']!='refine':continue
        try:val.append((encode_input(tokenizer,row['input'],args.max_tokens),0 if binary and label=='benefit' else 1 if binary else labels.index(label)))
        except ValueError:continue
    history=[];best=None;started=time.perf_counter();rng=random.Random(args.seed)
    for epoch in range(1,args.epochs+1):
        order=list(range(len(fit)));rng.shuffle(order);model.train();loss_sum=0
        for start in range(0,len(order),args.batch_size):
            ids=order[start:start+args.batch_size];batch={k:v.to(args.device) for k,v in collate_inputs(tokenizer,[encoded[i] for i in ids]).items()}
            optimizer.zero_grad(set_to_none=True)
            loss=(criterion(model(**batch),targets[ids].to(args.device))*sample_weights[ids]).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();loss_sum+=loss.item()*len(ids)
        model.eval();val_loss=0
        with torch.inference_mode():
            for start in range(0,len(val),32):
                items=val[start:start+32];batch={k:v.to(args.device) for k,v in collate_inputs(tokenizer,[e for e,_ in items]).items()}
                val_loss+=criterion(model(**batch),torch.tensor([t for _,t in items],device=args.device)).sum().item()
        val_loss/=max(1,len(val));record=dict(epoch=epoch,train_loss=loss_sum/len(fit),validation_loss=val_loss)
        history.append(record);print(name+' '+json.dumps(record),flush=True)
        if best is None or val_loss<best[0]:
            best=(val_loss,epoch);torch.save({k:v.cpu() for k,v in model.state_dict().items()},directory/'gate.pt')
    model.load_state_dict(torch.load(directory/'gate.pt',map_location=args.device,weights_only=True))
    probs=probabilities(model,tokenizer,dev,args.device,args.max_tokens)
    if binary:thresholds,evidence=gain_thresholds(dev,probs,outcomes);policy=dict(gain_thresholds=thresholds)
    else:
        # Unsupported states preserve native scheduling rather than being
        # optimistically counted as zero-cost successful vetoes.
        thresholds,evidence=ternary_thresholds(dev,probs,outcomes);policy=dict(veto_thresholds=thresholds)
    def action(row,p):
        if binary:return select_gain_action(p,thresholds[row['input']['task_kind']],row['native_action'])
        return select_veto_action(p,thresholds[row['input']['task_kind']],row['native_action'])
    actions={r['case_id']:action(r,p) for r,p in zip(dev,probs)}
    metrics=policy_metrics(dev,outcomes,lambda r:'current' if actions[r['case_id']]=='refine' else 'rules')
    (directory/'encoder').mkdir();model.encoder.config.save_pretrained(directory/'encoder');tokenizer.save_pretrained(directory/'encoder')
    metadata=dict(input_version=INPUT_VERSION,labels=list(labels),tasks=list(TASKS),max_tokens=args.max_tokens,
        **policy,variant=name,recommended_route='native_veto',calibrated=False,human_gold=0,online_enabled=False,
        scope='round3_synthetic_operation_weak_prototype',window_context='empty_matches_Web_adapter',
        training_cases=len(fit),class_counts=dict(counts),class_weights=class_weights.cpu().tolist(),
        source_group_task_weighting=True,native_keep_not_fitted=True,unsupported_train=unsupported,
        conflicting_training_inputs_excluded=len(conflicting),
        best_epoch=best[1],epochs=args.epochs,history=history,train_seconds=time.perf_counter()-started,
        validation_selection=evidence,validation_metrics=metrics,parameters=sum(p.numel() for p in model.parameters()))
    c.write_json(directory/'metadata.json',metadata)
    (directory/'validation_predictions.jsonl').write_text(''.join(json.dumps(dict(case_id=r['case_id'],action=actions[r['case_id']],probabilities=p),ensure_ascii=False)+'\n' for r,p in zip(dev,probs)))
    del model,optimizer;torch.cuda.empty_cache()
    gate=WeakRefinementGate(directory,device='cpu');latency=benchmark(lambda r:gate.decide_with_fallback(r['input'],r['native_action']),dev)
    metrics['gate_cpu_estimated_ms']=latency['overall']['mean_ms']*len(dev)
    metrics['estimated_serial_cost_ms']=metrics['recorded_operation_cost_ms']+metrics['gate_cpu_estimated_ms']
    c.write_json(directory/'validation_metrics.json',metrics);c.write_json(directory/'latency.json',latency)
    return dict(metrics=metrics,thresholds=policy,weights_sha256=c.file_hash(directory/'gate.pt'),metadata_sha256=c.file_hash(directory/'metadata.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--encoder',type=Path,default=ROOT/'models/entity_ranker/encoder')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--max-tokens',type=int,default=256)
    parser.add_argument('--seed',type=int,default=20261011);args=parser.parse_args()
    import torch
    torch.set_num_threads(2);out=args.root/'training'
    if out.exists():raise FileExistsError('Training never overwrites an earlier trial')
    out.mkdir();dataset=args.root/'dataset';train=c.read_rows(dataset/'train.jsonl');dev=c.read_rows(dataset/'dev.jsonl')
    outcomes=json.loads((dataset/'outcomes.json').read_text());variants={}
    for name in ['encoder_ternary_strict','encoder_ternary_operation','encoder_binary_operation']:
        variants[name]=fit_variant(train,dev,outcomes,out,args,name)
    primary=min(variants,key=lambda name:variants[name]['metrics']['estimated_serial_cost_ms'])
    selection=dict(event='round3_policy_frozen_before_test_audio',primary_model=primary,primary_route='native_veto',
        variants=variants,freeze_unix=time.time(),dataset_manifest_sha256=c.file_hash(dataset/'manifest.json'),
        test_not_opened=True,cohort_task_minimum_retention=0.95,old_dev_is_reused=True,
        retention_unit='recording_group_all_beneficial_views_per_cohort_task',
        criterion='minimum_development_recorded_model_cost_plus_CPU_gate_cost_after_retention_constraints')
    c.write_json(out/'selection.json',selection);print('FROZEN '+json.dumps(selection,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
