"""Actually run the frozen selected window model in Web final snapshots."""
from __future__ import annotations
import argparse
import copy
from functools import wraps
import json
from pathlib import Path
import pickle
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d
from experiments.scripts.evaluate_gate_streaming import without_confidence
from experiments.scripts.gate_terminal_gain_pilot import view_id,window_key,final_snapshot
from experiments.scripts.train_terminal_gain_linear import feature_text,executed_arm


def replay(root,device):
    selection_path=root/'linear_training/selection.json';selection=json.loads(selection_path.read_text())
    primary=selection['primary']
    if primary=='native':raise ValueError('No useful learned route selected')
    model_path=root/'linear_training'/primary/'model.pkl';metadata=selection['variants'][primary]
    if c.file_hash(model_path)!=metadata['model_sha256']:raise ValueError('Classifier changed after freezing')
    with model_path.open('rb') as stream:classifier=pickle.load(stream)
    column=list(classifier.classes_).index('skip_allowed');threshold=metadata['threshold']
    source=root/'dev_collection';folder=source/'runtime';c.verify_frozen(folder);web=c.load_system(folder)
    import torch
    from fastapi.testclient import TestClient
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2);threadpool_limits(limits=2)
    teacher=json.loads((folder/'run_configuration.json').read_text());model=Path(teacher['model'])
    if {p.name:c.file_hash(p) for p in model.glob('*.safetensors')}!=teacher['weights_sha256']:raise RuntimeError('Teacher changed')
    if torch.cuda.mem_get_info(device)[0]<sum(p.stat().st_size for p in model.glob('*.safetensors'))+1024**3:raise RuntimeError('Insufficient GPU memory')
    refiner=TransformersRefiner(model,device,256)
    if str(next(refiner._model.parameters()).dtype)!=teacher['dtype']:raise RuntimeError('Precision changed')
    plausibility=RepetitionPlausibility('cpu')
    if not plausibility._ensure_model():raise RuntimeError('MLM unavailable')
    calls=[];ppl=[];gates=[];active={};local=threading.local()
    def stage_cpu():
        plausibility._model.to('cpu');plausibility.device='cpu';torch.cuda.empty_cache()
    original_forward=refiner._model.forward
    @wraps(original_forward)
    def forward(*a,**k):
        entry=getattr(local,'entry',None);tokens=k.get('input_ids',a[0] if a else None)
        if entry is not None:
            entry['forward_passes']+=1
            if tokens is not None:entry['input_token_positions']+=int(tokens.numel())
        return original_forward(*a,**k)
    refiner._model.forward=forward
    class RefinerAudit:
        def __getattr__(self,name):
            value=getattr(refiner,name)
            if not callable(value) or not(name.startswith('refine') or name.startswith('review_')):return value
            def invoke(*a,**k):
                stage_cpu();entry=dict(method=name,input=list(a),kwargs=k,forward_passes=0,input_token_positions=0)
                started=time.perf_counter();local.entry=entry
                try:
                    result=value(*a,**k);entry['output']=list(result);return result
                except BaseException as error:entry['error']=repr(error);raise
                finally:entry['wall_ms']=(time.perf_counter()-started)*1000;calls.append(entry);local.entry=None
            return invoke
    class PlausibilityAudit:
        def __getattr__(self,name):return getattr(plausibility,name)
        def decide(self,*a,**k):return self.invoke('decide',*a,**k)
        def decide_short_context(self,*a,**k):return self.invoke('decide_short_context',*a,**k)
        def invoke(self,name,*a,**k):
            entry=dict(method=name,input=list(a),kwargs=k);started=time.perf_counter()
            try:
                plausibility._model.to(device);plausibility.device=device
                result=getattr(plausibility,name)(*a,**k);entry.update(decision=result.decision,reason=result.reason);return result
            finally:entry['wall_ms']=(time.perf_counter()-started)*1000;ppl.append(entry);stage_cpu()
    def router(value,native):
        if value['task_kind']!='window_refine':raise AssertionError('Learned local veto is forbidden')
        started=time.perf_counter();prob=float(classifier.predict_proba([feature_text(value)])[0,column])
        action='keep' if prob>=threshold else native
        gates.append(dict(input=value,key=window_key(value),probability=prob,action=action,
                          gate_ms=(time.perf_counter()-started)*1000))
        return dict(action=action)
    def recorded_request(asr_url,endpoint,session_id=None,data=b'',params=None):
        if endpoint=='/stream/start':return dict(session_id='frozen-final-linear')
        if endpoint=='/stream/finish':return without_confidence(copy.deepcopy(active['row']['final_payload']))
        if endpoint=='/stream/cancel':return dict(cancelled=True)
        raise AssertionError('No live ASR endpoint: '+endpoint)
    web._stream_request=recorded_request
    web.TransformersRefiner=lambda *_a,**_k:RefinerAudit()
    web.RepetitionPlausibility=lambda *_a,**_k:PlausibilityAudit()
    refiner.refine_with_numeric_rules('今天下午开会。')
    output=root/'linear_replay';output.mkdir(exist_ok=True)
    configuration=dict(scope='actual_classifier_callback_WebSocket_frozen_ASR_development_final_snapshot',
        selection_sha256=c.file_hash(selection_path),model_sha256=c.file_hash(model_path),primary=primary,threshold=threshold,
        runtime_protocol_sha256=c.file_hash(folder/'protocol.json'),source_rows_sha256=c.file_hash(source/'sources.jsonl'),
        adapter_sha256=c.file_hash(Path(__file__)),dtype=teacher['dtype'],refiner_weights_sha256=teacher['weights_sha256'],
        confidence='disabled',caches='fresh_per_view',local_policy='native',data_role='reused_development',
        online_latency=False,independent_test=False,human_gold=False,audio_script_verified=False)
    cfg_path=output/'configuration.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text())!=configuration:raise RuntimeError('Replay configuration changed')
    c.write_json(cfg_path,configuration);(output/'executed_code').mkdir(exist_ok=True)
    (output/'executed_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    baselines={view_id(r):r for r in d.read_rows(source/'recordings.jsonl')}
    predicted=json.loads((root/'linear_training/development_predictions.json').read_text())[primary]
    completed=d.read_rows(output/'recordings.jsonl');done={r['recording_view_id'] for r in completed}
    normalizer=web.ZhITNNormalizer(folder/'native/libzh_itn_bridge.so')
    normalize=lambda s:d.lexical(normalizer.normalize(s).text)
    for i,row in enumerate(d.read_rows(source/'sources.jsonl'),1):
        identity=view_id(row)
        if identity in done:continue
        active['row']=row;calls.clear();ppl.clear();gates.clear()
        app=web.create_app(model,device,'http://frozen-asr.invalid','Chinese',256,output/f'{identity}.trace.jsonl',
            entity_db=folder/'entities_snapshot.db',entity_fuzzy_mode='auto',refinement_gate_mode='tri_state',
            final_refinement_mode='off',zh_itn_library=folder/'native/libzh_itn_bridge.so',
            experimental_gate_router=router,experimental_gate_tasks=('window_refine',))
        with TestClient(app) as client:final,events,wall=final_snapshot(client)
        baseline=baselines[identity];mask={k for k,p in predicted[identity].items() if p>=threshold}
        expected=executed_arm(baseline,mask)
        if final['raw_text']!=row['raw_text']:raise AssertionError('Frozen ASR input changed')
        if any(m.get('error') for m in calls) or final['refiner_session_stats']['inflight_call_count']:raise AssertionError('Model failed or not drained')
        if {g['key'] for g in gates}!={r['key'] for r in baseline['native']['routes']}:raise AssertionError('Gate inputs changed')
        actual_mask={g['key'] for g in gates if g['action']=='keep'}
        if actual_mask!=mask:raise AssertionError('Classifier actions disagree with frozen predictions')
        if final['clean_text']!=expected['final_text']:raise AssertionError('Executed output disagrees with paired branch')
        record=dict(recording_view_id=identity,seed_id=row['seed_id'],variant=row['variant'],split='dev',
            template_group=row['template_group'],cohort=row['round3_cohort'],final=final,events=events,
            final_text=final['clean_text'],intended_distance=c.edit_distance(normalize(final['clean_text']),normalize(row['intended_text'])),
            native_distance=baseline['native']['intended_distance'],intended_chars=baseline['intended_chars'],
            model_calls=copy.deepcopy(calls),plausibility_calls=copy.deepcopy(ppl),gate_calls=copy.deepcopy(gates),
            snapshot_wall_ms=wall,expected_branch_verified=True)
        c.append_row(output/'recordings.jsonl',record);completed.append(record)
        print(f'LINEAR {i}/288 {row["seed_id"]} {row["variant"]} calls={len(calls)} veto={len(mask)}',flush=True)
    summary=dict(recording_views=len(completed),exact_script=sum(r['intended_distance']==0 for r in completed),
        intended_micro_cer=sum(r['intended_distance'] for r in completed)/sum(r['intended_chars'] for r in completed),
        degraded_views=sum(r['intended_distance']>r['native_distance'] for r in completed),
        refiner_calls=sum(len(r['model_calls']) for r in completed),plausibility_calls=sum(len(r['plausibility_calls']) for r in completed),
        gate_total_ms=sum(g['gate_ms'] for r in completed for g in r['gate_calls']),
        vetoes=sum(g['action']=='keep' for r in completed for g in r['gate_calls']),
        complete=len(completed)==288,all_executed_branches_match=True,independent_test=False,online_latency=False)
    c.verify_frozen(folder)
    if c.file_hash(selection_path)!=configuration['selection_sha256']:raise AssertionError('Policy changed during replay')
    c.write_json(output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();replay(a.root.resolve(),a.device)
