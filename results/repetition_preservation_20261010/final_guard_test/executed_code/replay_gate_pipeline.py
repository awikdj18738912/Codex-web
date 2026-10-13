"""Actual frozen Web final-snapshot replay; no streaming history or training."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as collector, synthetic_gate_data as data
# Keep this class before loading the isolated frozen system package.
from system.weak_refinement_gate import WeakRefinementGate


def review_after_windows(web, review, text, policy, student, domain, stats):
    """Rebuild candidates on the actual chosen post-window text, not saved pairs."""
    decisions=[]
    if policy=='rules':return text,[],decisions
    original=web.find_repetition_review_candidates
    def filtered(value, **kwargs):
        candidates=original(value,**kwargs)
        selected=[]
        for candidate in candidates:
            item=dict(task_kind='local_edit_review',domain=domain,baseline_text=candidate.context,
                available_context=candidate.context,
                candidate_local_start=web._repetition_candidate_context_offset(value,candidate),
                candidate={k:getattr(candidate,k) for k in ['source','target','kind']})
            started=time.perf_counter()
            decision=student.decide_with_fallback(item,'refine')
            decisions.append(dict(candidate=candidate.public_dict(),input=item,decision=decision,
                                  gate_ms=(time.perf_counter()-started)*1000))
            if decision['action']=='refine':selected.append(candidate)
        return tuple(selected)
    if policy=='student':web.find_repetition_review_candidates=filtered
    try:
        # Fresh caches per arm: cold final snapshot, not actual streamed finish.
        result=review(text,final=True,call_stats=stats,review_cache={},response_cache={})
        return web.clean_transcript_deterministically(result[0]),result[1],decisions
    finally:
        web.find_repetition_review_candidates=original


def summarize(rows, normalize):
    result={}
    for policy in rows[0]['arms']:
        distances=[];exact=improved=degraded=calls=ppl=0;chars=0
        for row in rows:
            target=normalize(row['intended_script']);chars+=len(target)
            baseline=normalize(row['arms']['rules']['final_text'])
            arm=row['arms'][policy];text=normalize(arm['final_text'])
            dist=collector.edit_distance(target,text);base=collector.edit_distance(target,baseline)
            distances.append(dist);exact+=dist==0;improved+=dist<base;degraded+=dist>base
            calls+=len(arm['logical_calls']);ppl+=len(arm['plausibility_calls'])
        result[policy]=dict(recordings=len(rows),exact_intended=exact,intended_micro_cer=sum(distances)/max(1,chars),
            improved_vs_rules=improved,degraded_vs_rules=degraded,
            logical_refiner_calls=calls,logical_plausibility_calls=ppl)
    differences=[]
    for row in rows:
        a,b=[normalize(row['arms'][p]['final_text']) for p in ['current','student']]
        if a!=b:
            target=normalize(row['intended_script'])
            differences.append(dict(recording_group=row['recording_group'],
                current=row['arms']['current']['final_text'],student=row['arms']['student']['final_text'],
                current_distance=collector.edit_distance(a,target),student_distance=collector.edit_distance(b,target)))
    return dict(by_policy=result,current_student_differences=differences,
                normalized_same_current_student=len(rows)-len(differences))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--policies',nargs='+',default=['rules','current','student','always'])
    parser.add_argument('--text-probe-mode',action='store_true',help='Developer text expectations, no ASR or audio accuracy claims')
    args=parser.parse_args()
    if not {'rules','current','student'}.issubset(args.policies) or any(p not in {'rules','current','student','always'} for p in args.policies):
        parser.error('Require rules, current and student; optional always')
    source=args.source.resolve();folder=source/'pairs_full';out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=collector.verify_frozen(folder)
    rows=data.read_rows(source/'asr.jsonl')
    if any(r.get('data_kind')=='text_development_probe' for r in rows)!=args.text_probe_mode:
        raise RuntimeError('Text probes require explicit --text-probe-mode')
    if args.text_probe_mode and any(r.get('data_kind')!='text_development_probe' for r in rows):
        raise RuntimeError('Cannot mix text probes and ASR records')
    if protocol['settings']['window_edit_policy']!='compat':raise RuntimeError('Unsupported replay policy')
    import torch
    torch.set_num_threads(2)
    student=WeakRefinementGate(args.checkpoint,device='cpu')
    frozen=collector.load_system(folder)
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    needed=sum(p.stat().st_size for p in args.model.glob('*.safetensors'))
    if args.device.startswith('cuda') and torch.cuda.mem_get_info(args.device)[0]<needed+512*1024**2:
        raise RuntimeError('Insufficient free memory for original-precision Refiner')
    config=json.loads((folder/'run_configuration.json').read_text())
    weights={p.name:collector.file_hash(p) for p in args.model.glob('*.safetensors')}
    if weights!=config['weights_sha256']:raise RuntimeError('Refiner weight drift')
    refiner=TransformersRefiner(args.model,args.device,256)
    if str(next(refiner._model.parameters()).dtype)!=config['dtype']:raise RuntimeError('Refiner precision drift')
    plausibility=RepetitionPlausibility('cpu');logical=[];ppl=[]
    def stage_cpu():
        if plausibility._model is not None:plausibility._model.to('cpu')
        plausibility.device='cpu'
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    class RefinerAudit:
        def __getattr__(self,name):
            value=getattr(refiner,name)
            if not callable(value) or not (name.startswith('refine') or name.startswith('review_')):return value
            def invoke(*a,**k):
                stage_cpu();started=time.perf_counter();entry=dict(method=name,input=list(a),kwargs=k)
                try:
                    result=value(*a,**k);entry['output']=list(result);return result
                except BaseException as e:
                    entry.update(error_type=type(e).__name__,error=str(e));raise
                finally:
                    entry['wall_ms']=(time.perf_counter()-started)*1000;logical.append(entry)
            return invoke
    class PlausibilityAudit:
        def __getattr__(self,name):return getattr(plausibility,name)
        def decide(self,*a,**k):return self.invoke('decide',*a,**k)
        def decide_short_context(self,*a,**k):return self.invoke('decide_short_context',*a,**k)
        def invoke(self,name,*a,**k):
            started=time.perf_counter();entry=dict(method=name)
            try:
                if plausibility._ensure_model():plausibility._model.to(args.device);plausibility.device=args.device
                result=getattr(plausibility,name)(*a,**k);entry.update(decision=result.decision,reason=result.reason);return result
            except BaseException as e:
                entry.update(error_type=type(e).__name__,error=str(e));raise
            finally:
                entry['wall_ms']=(time.perf_counter()-started)*1000;ppl.append(entry);stage_cpu()
    apps=[];original_create=frozen.create_app
    def capture_app(*a,**k):
        app=original_create(*a,**k);apps.append(app);return app
    frozen.create_app=capture_app
    try:
        refine,review,gate=collector.create_runtime(frozen,folder,args.model,args.device,
                                          refiner=RefinerAudit(),plausibility=PlausibilityAudit())
    finally:frozen.create_app=original_create
    endpoint=next(r.endpoint for r in apps[0].routes if getattr(r,'path',None)=='/ws/stream')
    fallback=collector.closure_value(endpoint,'fallback_update')
    original_decide=gate.decide;active={};window_decisions=[]
    def decide(*a,**k):
        if gate.policy!='student':return original_decide(*a,**k)
        native=gate.base.decide(*a,**k);gate.last_native=native
        started=time.perf_counter();decision=student.decide_with_fallback(active['input'],native.action)
        window_decisions.append(dict(native=native.public_dict(),input=active['input'],decision=decision,
                                     gate_ms=(time.perf_counter()-started)*1000))
        if decision['action']==native.action:return native
        return replace(native,should_refine=False,action='keep',reasons=('frozen_student_veto',))
    gate.decide=decide
    manifest=dict(scope='text_only_development_sequential_probe' if args.text_probe_mode else 'cold_authoritative_final_snapshot_sequential_post_test_diagnostic',
        text_probe_mode=args.text_probe_mode,
        not_full_streaming_session=True,not_online_latency=True,new_training=False,human_gold=False,
        confidence_metadata={},policies=args.policies,source=str(source),checkpoint=str(args.checkpoint.resolve()),
        gate_weights_sha256=collector.file_hash(args.checkpoint/'gate.pt'),
        gate_metadata_sha256=collector.file_hash(args.checkpoint/'metadata.json'),
        selection_sha256=collector.file_hash(args.checkpoint.parent/'selection.json'),
        adapter_sha256=collector.file_hash(Path(__file__)),protocol_sha256=collector.file_hash(folder/'protocol.json'),
        refiner_weights_sha256=weights,dtype=config['dtype'],window_settings=protocol['settings'],
        order=['fixed_K3_windows','final_entity_normalization_no_second_ITN','regenerate_local_candidates','actual_local_review','deterministic_cleanup'],
        caches='fresh per recording and policy; no earlier streaming decisions',
        cost_scope='logical Refiner calls include retries; weights staging overhead prevents online wall-time conclusions')
    manifest_path=out/'configuration.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise RuntimeError('Replay resume configuration drift')
    collector.write_json(manifest_path,manifest)
    completed=data.read_rows(out/'replay.jsonl')
    done={r['recording_group'] for r in completed}
    prepared={}
    for case in data.read_rows(folder/'cases.jsonl'):
        prepared.setdefault(case['recording_group'],{})[case['window_index']]=case
    normalizer=frozen.ZhITNNormalizer(folder/'native/libzh_itn_bridge.so')
    normalize=lambda s:data.lexical(normalizer.normalize(s).text)
    for index,row in enumerate(rows,1):
        if args.text_probe_mode:
            identity,domain=row['recording_group'],row['domain']
        else:identity,domain,_=collector.recording_identity(row)
        if identity in done:continue
        raw=row['raw_text'].strip()
        partition=frozen.CumulativeWindowRefinement(None,window_size=3,one_punctuation_window=True,
            fixed_groups=True,group_numeric_self_corrections=True)
        chunks,_pending=partition._fixed_chunks(raw,source_chunks=collector.final_source_segments(row,frozen),
                                             final=True,preserve_cached_partition=False)
        groups=[tuple(chunks[i:i+3]) for i in range(0,len(chunks),3)]
        # Compute only pre-operation rule inputs; no teacher outputs enter the student.
        protector,matcher=collector.tools_for_domain(frozen,folder,domain);gate.policy='rules'
        baselines=[]
        for group in groups:
            baseline=refine(frozen.join_refined_segments(group),'Chinese',True,protector,None,matcher,
                            single_window=True,confidence_metadata={})['clean_text']
            baselines.append(baseline)
        for wi,case in prepared.get(identity,{}).items():
            if baselines[wi]!=case['baseline_text']:raise RuntimeError('Prepared rule baseline drift: '+case['case_id'])
        arms={}
        for policy in args.policies:
            protector,matcher=collector.tools_for_domain(frozen,folder,domain)
            gate.policy=policy;logical.clear();ppl.clear();window_decisions.clear()
            stats=frozen._RefinerSessionStats();parts=[];windows=[];cursor=0;started=time.perf_counter()
            for wi,group in enumerate(groups):
                text=frozen.join_refined_segments(group);start=raw.find(text,cursor)
                context=raw[max(0,start-96):start] if start>=0 else ''
                if start>=0:cursor=start+len(text)
                active['input']=dict(task_kind='window_refine',domain=domain,data_kind=row['data_kind'],raw_text=text,
                    baseline_text=baselines[wi],available_context=context,window_chunk_count=len(group))
                result=refine(text,'Chinese',True,protector,None,matcher,single_window=True,
                              confidence_metadata={},call_stats=stats)
                parts.append(result['clean_text']);windows.append(collector.compact_result(result))
            window_text=frozen.clean_transcript_deterministically(frozen.join_refined_segments(parts))
            entity=fallback(window_text,'Chinese',protector,None,matcher,'streaming_final_entity_normalization',
                            time.perf_counter(),{},apply_zh_itn=False)
            final_text,records,local_decisions=review_after_windows(frozen,review,entity['clean_text'],policy,
                                                                 student,domain,stats)
            stats.close_new_calls()
            if stats.snapshot()['failed_call_count'] or any(c.get('error_type') for c in [*logical,*ppl]):
                raise RuntimeError('Failed model operation; not valid replay evidence')
            arms[policy]=dict(window_text=window_text,pre_review_text=entity['clean_text'],final_text=final_text,
                windows=windows,window_gate_decisions=list(window_decisions),local_gate_decisions=local_decisions,
                reviews=records,logical_calls=list(logical),plausibility_calls=list(ppl),stats=stats.snapshot(),
                wall_ms=(time.perf_counter()-started)*1000)
        record=dict(recording_group=identity,seed_id=row['seed_id'],raw_text=raw,
            spoken_script=row['spoken_text'],intended_script=row['intended_text'],audio_script_verified=False,arms=arms)
        collector.append_row(out/'replay.jsonl',record);completed.append(record)
        print(f'REPLAY {index}/{len(rows)} {identity} '+json.dumps({p:dict(text=a['final_text'],calls=len(a['logical_calls'])) for p,a in arms.items()},ensure_ascii=False),flush=True)
    for key,path in [('gate_weights_sha256',args.checkpoint/'gate.pt'),('gate_metadata_sha256',args.checkpoint/'metadata.json'),
                     ('selection_sha256',args.checkpoint.parent/'selection.json')]:
        if collector.file_hash(path)!=manifest[key]:raise RuntimeError('Frozen policy changed during replay')
    collector.verify_frozen(folder)
    report=summarize(completed,normalize)
    report.update(scope=manifest['scope'],complete=len(completed)==len(rows),
        references='developer_text_expectations_not_audio_gold' if args.text_probe_mode else 'unverified_synthesis_scripts_not_human_gold')
    collector.write_json(out/'summary.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
