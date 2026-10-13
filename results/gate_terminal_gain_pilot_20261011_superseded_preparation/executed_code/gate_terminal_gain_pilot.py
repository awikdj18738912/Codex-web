"""Train-only per-window counterfactuals through the actual Web finalizer.

Frozen, recorded ASR final responses eliminate input drift. This is a cold
authoritative-final-snapshot diagnostic, not live ASR or streaming latency.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
from functools import wraps
import json
from pathlib import Path
import random
import shutil
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c, synthetic_gate_data as d
from experiments.scripts.evaluate_gate_streaming import without_confidence


def window_key(value):
    return c.digest({k:value[k] for k in ('task_kind','domain','raw_text','baseline_text','available_context')})


def route_window(value, native, veto):
    if value['task_kind']!='window_refine':
        raise AssertionError('Local review must never be routed by this experiment')
    key=window_key(value)
    return dict(action='keep' if veto=='all' or key in veto else native),key


def heldout_boundary():
    values=defaultdict(set);paths=set()
    for path in (ROOT/'results').rglob('texts.jsonl'):
        if path.parent.name in {'dev_data','risk_dev_data','test_data'} or (path.parent/'test_consumed.json').exists():
            paths.add(path)
            other=path.parent/'asr.jsonl'
            if other.exists():paths.add(other)
    for path in (ROOT/'results/learned_gate_weak_prototype_20261010').glob('*.jsonl'):
        if path.name in {'validation.jsonl','test.jsonl'}:paths.add(path)
    for path in sorted(paths):
        for row in d.read_rows(path):
            for key in ('seed_id','recording_group','pcm_sha256','audio_sha256','template_group'):
                if row.get(key):values[key].add(row[key])
            if row.get('split_group'):values['template_group'].add(row['split_group'])
            for key in ('spoken_text','intended_text','raw_text'):
                if isinstance(row.get(key),str):values['text'].add(c.normalized(row[key]))
    return values,{str(path):c.file_hash(path) for path in sorted(paths)}


def prepare(root,source,runtime,count,seed):
    if root.exists():raise FileExistsError('Use a new experiment directory')
    c.verify_frozen(runtime)
    if count<5:raise ValueError('Require at least five recordings for category coverage')
    forbidden,hashes=heldout_boundary()
    cases=d.read_rows(source/'pairs_full/cases.jsonl')
    opportunities={r['recording_group'] for r in cases if r.get('variant')=='clean'
                   and r.get('original_gate',{}).get('action')=='refine'}
    rows=d.read_rows(source/'asr.jsonl');eligible=[]
    for row in rows:
        if row.get('synthetic_split')!='train' or row.get('variant')!='clean':continue
        if row['recording_group'] not in opportunities:continue
        if any(row.get(k) in forbidden[k] for k in ('seed_id','recording_group','pcm_sha256','audio_sha256','template_group')):continue
        if any(c.normalized(row[k]) in forbidden['text'] for k in ('spoken_text','intended_text','raw_text')):continue
        eligible.append(row)
    rng=random.Random(seed);strata=defaultdict(lambda:defaultdict(list))
    for row in eligible:strata[row['category']][row['template_group']].append(row)
    queues={}
    for category,groups in sorted(strata.items()):
        for pool in groups.values():rng.shuffle(pool)
        ordered=[]
        while any(groups.values()):
            for group in sorted(groups):
                if groups[group]:ordered.append(groups[group].pop())
        queues[category]=ordered
    selected=[]
    while len(selected)<count and any(queues.values()):
        for category in sorted(queues):
            if queues[category] and len(selected)<count:selected.append(queues[category].pop(0))
    if len(selected)!=count:raise ValueError('Insufficient distinct train recordings')
    rng.shuffle(selected)
    for key in ('seed_id','recording_group','pcm_sha256'):
        if len({r[key] for r in selected})!=count:raise AssertionError('Duplicate '+key)
    root.mkdir(parents=True);(root/'logs').mkdir();(root/'executed_code').mkdir()
    shutil.copytree(runtime,root/'runtime')
    for row in selected:
        if c.file_hash(Path(row['audio_path']))!=row['audio_sha256']:raise AssertionError('Audio file changed')
        c.append_row(root/'sources.jsonl',row)
    (root/'executed_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    manifest=dict(recordings=count,eligible=len(eligible),seed=seed,role='train_only_terminal_gain_pilot',
        source=str(source),source_sha256=c.file_hash(source/'asr.jsonl'),
        sampling_inputs_sha256=c.file_hash(source/'pairs_full/cases.jsonl'),
        sampling='input-side original native REFINE eligibility; category and template round robin; no teacher outcomes used',
        source_rows_sha256=c.file_hash(root/'sources.jsonl'),
        runtime_protocol_sha256=c.file_hash(root/'runtime/protocol.json'),
        adapter_sha256=c.file_hash(Path(__file__)),excluded_sources_sha256=hashes,
        categories=dict(Counter(r['category'] for r in selected)),templates=len({r['template_group'] for r in selected}),
        human_gold=False,audio_script_verified=False,confidence='disabled',new_training=False,
        decision_unit='all occurrences of an identical available window input; source groups with distinct inputs intervened separately',
        scope='actual_WebSocket_final_snapshot_with_frozen_recorded_ASR_not_live_streaming_latency',created_unix=time.time())
    c.write_json(root/'manifest.json',manifest)
    return manifest


def paired_effect(native,skip,stable=True):
    delta=skip['intended_distance']-native['intended_distance']
    calls=native['cost']['refiner_calls']-skip['cost']['refiner_calls']
    ppl=native['cost']['plausibility_calls']-skip['cost']['plausibility_calls']
    tokens=native['cost']['refiner_token_positions']-skip['cost']['refiner_token_positions']
    label='benefit' if delta>0 else 'harm' if delta<0 else 'neutral'
    return dict(label=label if stable else 'uncertain',window_terminal_quality_gain=delta,
        skip_saves_refiner_calls=calls,skip_saves_plausibility_calls=ppl,skip_saves_refiner_token_positions=tokens,
        skip_has_compute_advantage_proxy=stable and delta<=0 and calls>=0 and ppl>=0 and tokens>=0
            and (calls>0 or ppl>0 or tokens>0),
        skip_final_same=native['final_text']==skip['final_text'],
        native_control_stable=stable,human_label=None,label_basis='paired_terminal_unverified_script_distance',
        reference='unverified_synthesis_script_not_human_gold')


def final_snapshot(client):
    events=[];started=time.perf_counter()
    with client.websocket_connect('/ws/stream?mode=streaming&domain=general') as ws:
        ready=ws.receive_json()
        if ready.get('event')!='ready':raise AssertionError('Web did not become ready')
        ws.send_json({'event':'finish'})
        while True:
            payload=ws.receive_json();events.append(payload)
            if payload.get('event')=='error':raise RuntimeError(str(payload))
            if payload.get('event')=='final':return payload,events,(time.perf_counter()-started)*1000


def run(root,device):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['adapter_sha256']!=c.file_hash(Path(__file__)):raise RuntimeError('Executed adapter changed')
    if manifest['source_rows_sha256']!=c.file_hash(root/'sources.jsonl'):raise RuntimeError('Frozen sources changed')
    folder=root/'runtime';c.verify_frozen(folder);web=c.load_system(folder)
    import torch
    import soundfile as sf
    from fastapi.testclient import TestClient
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    torch.set_num_threads(2)
    teacher=json.loads((folder/'run_configuration.json').read_text());model=Path(teacher['model'])
    if {p.name:c.file_hash(p) for p in model.glob('*.safetensors')}!=teacher['weights_sha256']:raise RuntimeError('Refiner weights changed')
    required=sum(p.stat().st_size for p in model.glob('*.safetensors'))+1024**3
    if torch.cuda.mem_get_info(device)[0]<required:raise RuntimeError('Insufficient memory for original-precision Refiner')
    refiner=TransformersRefiner(model,device,256)
    if str(next(refiner._model.parameters()).dtype)!=teacher['dtype']:raise RuntimeError('Refiner precision changed')
    plausibility=RepetitionPlausibility('cpu')
    if not plausibility._ensure_model():raise RuntimeError('MLM model unavailable')
    calls=[];ppl=[];routes=[];active={};local=threading.local()
    def stage_cpu():
        plausibility._model.to('cpu');plausibility.device='cpu';torch.cuda.empty_cache()
    original_forward=refiner._model.forward
    @wraps(original_forward)
    def audited_forward(*a,**k):
        entry=getattr(local,'entry',None)
        tensor=k.get('input_ids',a[0] if a else None)
        if entry is not None:
            entry['forward_passes']+=1
            if tensor is not None:entry['input_token_positions']+=int(tensor.numel())
        return original_forward(*a,**k)
    refiner._model.forward=audited_forward
    class RefinerAudit:
        def __getattr__(self,name):
            value=getattr(refiner,name)
            if not callable(value) or not(name.startswith('refine') or name.startswith('review_')):return value
            def invoke(*a,**k):
                stage_cpu();entry=dict(method=name,input=list(a),kwargs=k,forward_passes=0,input_token_positions=0)
                started=time.perf_counter();local.entry=entry
                try:
                    result=value(*a,**k);entry['output']=list(result);return result
                except BaseException as error:
                    entry['error']=repr(error);raise
                finally:
                    entry['wall_ms']=(time.perf_counter()-started)*1000;calls.append(entry);local.entry=None
            return invoke
    class PlausibilityAudit:
        def __getattr__(self,name):return getattr(plausibility,name)
        def decide(self,*a,**k):return self.invoke('decide',*a,**k)
        def decide_short_context(self,*a,**k):return self.invoke('decide_short_context',*a,**k)
        def invoke(self,name,*a,**k):
            entry=dict(method=name,input=list(a),kwargs=k);started=time.perf_counter()
            try:
                plausibility._model.to(device);plausibility.device=device
                result=getattr(plausibility,name)(*a,**k)
                entry.update(decision=result.decision,reason=result.reason);return result
            finally:
                entry['wall_ms']=(time.perf_counter()-started)*1000;ppl.append(entry);stage_cpu()
    def router(value,native):
        decision,key=route_window(value,native,active['veto'])
        routes.append(dict(key=key,input=value,native_action=native,action=decision['action']))
        return decision
    def recorded_request(asr_url,endpoint,session_id=None,data=b'',params=None):
        if endpoint=='/stream/start':return dict(session_id='frozen-final-input')
        if endpoint=='/stream/finish':return without_confidence(copy.deepcopy(active['row']['final_payload']))
        if endpoint=='/stream/cancel':return dict(cancelled=True)
        raise AssertionError('No live ASR or audio chunks in a final-snapshot experiment: '+endpoint)
    web.TransformersRefiner=lambda *_a,**_k:RefinerAudit()
    web.RepetitionPlausibility=lambda *_a,**_k:PlausibilityAudit()
    web._stream_request=recorded_request
    refiner.refine_with_numeric_rules('今天下午开会。')
    normalizer=web.ZhITNNormalizer(folder/'native/libzh_itn_bridge.so')
    normalize=lambda s:d.lexical(normalizer.normalize(s).text)
    cfg=dict(manifest_sha256=c.file_hash(root/'manifest.json'),dtype=teacher['dtype'],weights_sha256=teacher['weights_sha256'],
        scope=manifest['scope'],device=device,original_refiner=True,no_learned_gate=True,no_training=True,
        local_policy='native',final_source_preservation=True,caches='fresh per recording and arm',
        intervention='native REFINE only; identical input key occurrences grouped; other decisions unchanged',
        repeated_native_control=True,confidence='disabled',human_gold=False,
        timing='model-call wall times include staging and shared GPU noise, not ASR or online latency',
        compute_proxy='actual decoder input token positions; not FLOPs, plus separate MLM call count')
    if (root/'configuration.json').exists() and json.loads((root/'configuration.json').read_text())!=cfg:raise RuntimeError('Resume configuration changed')
    c.write_json(root/'configuration.json',cfg)
    completed=d.read_rows(root/'recordings.jsonl');done={r['seed_id'] for r in completed}
    def arm(row,name,veto):
        calls.clear();ppl.clear();routes.clear();active.update(row=row,veto=veto)
        app=web.create_app(model,device,'http://frozen-asr.invalid','Chinese',256,root/f'{row["seed_id"]}.{name}.trace.jsonl',
            entity_db=folder/'entities_snapshot.db',entity_fuzzy_mode='auto',refinement_gate_mode='tri_state',
            final_refinement_mode='off',zh_itn_library=folder/'native/libzh_itn_bridge.so',
            experimental_gate_router=router,experimental_gate_tasks=('window_refine',))
        with TestClient(app) as client:final,events,wall=final_snapshot(client)
        if final['raw_text']!=row['raw_text'].strip():raise AssertionError('Paired source changed')
        if any(m.get('error') for m in calls):raise AssertionError('Failed model operation')
        stats=final['refiner_session_stats']
        if stats.get('inflight_call_count') or stats.get('failed_call_count'):raise AssertionError('Incomplete final operation')
        record=dict(name=name,final_text=final['clean_text'],final=final,routes=copy.deepcopy(routes),
            model_calls=copy.deepcopy(calls),plausibility_calls=copy.deepcopy(ppl),events=events,snapshot_wall_ms=wall,
            intended_distance=c.edit_distance(normalize(final['clean_text']),normalize(row['intended_text'])),
            cost=dict(refiner_calls=len(calls),window_calls=sum(m['method'].startswith('refine') for m in calls),
                local_calls=sum(m['method'].startswith('review_') for m in calls),plausibility_calls=len(ppl),
                refiner_token_positions=sum(m['input_token_positions'] for m in calls),
                forward_passes=sum(m['forward_passes'] for m in calls),refiner_wall_ms=sum(m['wall_ms'] for m in calls),
                plausibility_wall_ms=sum(m['wall_ms'] for m in ppl)))
        c.append_row(root/'arms.jsonl',dict(seed_id=row['seed_id'],**record))
        return record
    rows=d.read_rows(root/'sources.jsonl')
    for i,row in enumerate(rows,1):
        if row['seed_id'] in done:continue
        audio,rate=sf.read(row['audio_path'],dtype='float32')
        if rate!=16000 or audio.ndim!=1 or c.digest(audio.astype('<f4').tobytes())!=row['pcm_sha256']:raise AssertionError('PCM hash mismatch')
        native=arm(row,'native',set())
        keys=sorted({r['key'] for r in native['routes']});by_key={r['key']:r['input'] for r in native['routes']}
        counterfactuals={key:arm(row,'veto_'+key[:12],{key}) for key in keys}
        all_skip=(counterfactuals[keys[0]] if len(keys)==1 else arm(row,'skip_all','all'))
        repeat=arm(row,'native_repeat',set())
        stable=(native['final_text']==repeat['final_text'] and native['cost']['refiner_calls']==repeat['cost']['refiner_calls']
                and native['cost']['plausibility_calls']==repeat['cost']['plausibility_calls']
                and native['cost']['refiner_token_positions']==repeat['cost']['refiner_token_positions'])
        effects=[]
        for key,skip in counterfactuals.items():
            native_keys={r['key'] for r in native['routes']}
            skip_keys={r['key'] for r in skip['routes']}
            valid=stable and native_keys==skip_keys and any(r['key']==key and r['action']=='keep' for r in skip['routes'])
            effects.append(dict(window_key=key,input=by_key[key],native_action='refine',**paired_effect(native,skip,valid)))
        record=dict(seed_id=row['seed_id'],template_group=row['template_group'],category=row['category'],
            recording_group=row['recording_group'],variant=row['variant'],synthetic_split='train',
            pcm_sha256=row['pcm_sha256'],intended_script=row['intended_text'],spoken_script=row['spoken_text'],raw_text=row['raw_text'],
            human_gold=False,audio_script_verified=False,native=native,native_repeat=repeat,
            counterfactuals=counterfactuals,skip_all=all_skip,skip_all_physical_reused=len(keys)==1,
            effects=effects,native_control_stable=stable,intended_chars=len(normalize(row['intended_text'])))
        c.append_row(root/'recordings.jsonl',record);completed.append(record)
        print(f'GAIN {i}/{len(rows)} {row["seed_id"]} '+json.dumps(dict(windows=len(keys),
            labels=[e['label'] for e in effects],native_calls=native['cost']['refiner_calls'],skip_calls=all_skip['cost']['refiner_calls'],
            native_text=native['final_text'],skip_text=all_skip['final_text'],stable=stable),ensure_ascii=False),flush=True)
    c.verify_frozen(folder)
    if c.file_hash(root/'sources.jsonl')!=manifest['source_rows_sha256']:raise AssertionError('Source drift')
    c.write_json(root/'job_state.json',dict(status='complete',recordings=len(completed),complete=len(completed)==len(rows),new_training=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--runtime',type=Path);p.add_argument('--count',type=int,default=36)
    p.add_argument('--seed',type=int,default=20261011);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.command=='prepare':print(json.dumps(prepare(a.root.resolve(),a.source.resolve(),a.runtime.resolve(),a.count,a.seed),ensure_ascii=False,indent=2))
    else:run(a.root.resolve(),a.device)
