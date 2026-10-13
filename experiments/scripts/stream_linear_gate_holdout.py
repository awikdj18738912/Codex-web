"""Frozen window classifier against actual paced HTTP-ASR/WebSocket audio.

Two repetitions reverse each clip's policy order. Bounded unacknowledged
chunks avoid silently hiding send backpressure inside finish-tail latency.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pickle
import random
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as d
from experiments.scripts.evaluate_gate_streaming import without_confidence
from experiments.scripts.train_terminal_gain_linear import feature_text


def policy_order(seed_id,repetition):
    reverse=(int(c.digest(seed_id)[:8],16)+repetition)%2
    return ['learned','current'] if reverse else ['current','learned']


def paced_session(client,pcm,chunk_samples=8000,max_inflight=2):
    events=[];failure=[];final=[];acks=[0];condition=threading.Condition()
    started=time.perf_counter()
    with client.websocket_connect('/ws/stream?mode=streaming&domain=general') as ws:
        ready=ws.receive_json()
        if ready.get('event')!='ready':raise RuntimeError('Web not ready')
        def receive():
            try:
                while True:
                    payload=ws.receive_json();events.append(dict(elapsed_seconds=time.perf_counter()-started,payload=payload))
                    with condition:
                        if payload.get('event')=='chunk_ack':acks[0]=max(acks[0],int(payload['chunk_index']))
                        if payload.get('event')=='final':final.append(payload)
                        condition.notify_all()
                    if final:return
                    if payload.get('event')=='error':raise RuntimeError(str(payload))
            except BaseException as error:
                with condition:failure.append(repr(error));condition.notify_all()
        reader=threading.Thread(target=receive,daemon=True);reader.start()
        audio_start=time.perf_counter();sent=0;backpressure=0.;peak=0
        for offset in range(0,len(pcm),chunk_samples*4):
            packet=pcm[offset:offset+chunk_samples*4]
            # Audio capture supplies each packet only after its samples exist.
            due=audio_start+(offset+len(packet))/4/16000
            remaining=due-time.perf_counter()
            if remaining>0:time.sleep(remaining)
            with condition:
                waiting=time.perf_counter()
                while sent-acks[0]>=max_inflight:
                    if failure:raise RuntimeError(str(failure))
                    if time.perf_counter()-waiting>90:raise TimeoutError('ASR acknowledgements stalled')
                    condition.wait(.1)
                backpressure+=time.perf_counter()-waiting
                if failure:raise RuntimeError(str(failure))
                sent+=1;peak=max(peak,sent-acks[0])
            # Do not hold the ACK lock across a potentially blocking transport.
            ws.send_bytes(packet)
        finish_sent=time.perf_counter();ws.send_json({'event':'finish'});reader.join(timeout=120)
        if reader.is_alive() or failure or not final:raise RuntimeError('No clean final: '+str(failure))
        completed=started+events[-1]['elapsed_seconds'];audio_seconds=len(pcm)/4/16000
        timing=dict(total_ws_seconds=completed-started,audio_seconds=audio_seconds,
            finish_tail_seconds=max(0,completed-finish_sent),
            completion_lag_from_audio_clock_seconds=max(0,completed-audio_start-audio_seconds),
            send_overrun_seconds=max(0,finish_sent-audio_start-audio_seconds),
            backpressure_seconds=backpressure,max_inflight_chunks=peak,acknowledged_chunks=acks[0],sent_chunks=sent)
    return final[0],events,timing


def run(root,device,asr_url,repetitions):
    import torch
    import soundfile as sf
    import httpx
    from fastapi.testclient import TestClient
    from threadpoolctl import threadpool_limits
    manifest=json.loads((root/'manifest.json').read_text());selection_path=root/'selection.json'
    if manifest['streaming_adapter_sha256']!=c.file_hash(Path(__file__)):raise ValueError('Frozen streaming adapter changed')
    if c.file_hash(selection_path)!=manifest['selection_sha256']:raise ValueError('Selection changed')
    model_path=root/'frozen_model/model.pkl'
    if c.file_hash(model_path)!=manifest['model_sha256']:raise ValueError('Classifier changed')
    selection=json.loads(selection_path.read_text());threshold=manifest['threshold']
    with model_path.open('rb') as stream:classifier=pickle.load(stream)
    column=list(classifier.classes_).index('skip_allowed')
    folder=root/'runtime';c.verify_frozen(folder);web=c.load_system(folder)
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    torch.set_num_threads(2);threadpool_limits(limits=2)
    teacher=json.loads((folder/'run_configuration.json').read_text());model=Path(teacher['model'])
    if {p.name:c.file_hash(p) for p in model.glob('*.safetensors')}!=teacher['weights_sha256']:raise ValueError('Refiner changed')
    if torch.cuda.mem_get_info(device)[0]<sum(p.stat().st_size for p in model.glob('*.safetensors'))+1024**3:raise RuntimeError('Insufficient GPU memory')
    refiner=TransformersRefiner(model,device,256);plausibility=RepetitionPlausibility('cpu')
    if str(next(refiner._model.parameters()).dtype)!=teacher['dtype']:raise ValueError('Precision drift')
    if not plausibility._ensure_model():raise RuntimeError('MLM unavailable')
    calls=[];ppl=[];gates=[];physical=dict(inflight=0);lock=threading.Lock();gpu_lock=threading.RLock()
    def begin():
        with lock:physical['inflight']+=1
    def end():
        with lock:physical['inflight']-=1
    def stage_cpu():
        plausibility._model.to('cpu');plausibility.device='cpu';torch.cuda.empty_cache()
    class RefinerAudit:
        def __getattr__(self,name):
            value=getattr(refiner,name)
            if not callable(value) or not(name.startswith('refine') or name.startswith('review_')):return value
            def invoke(*a,**k):
                begin();entry=dict(method=name,input=list(a),kwargs=k);started=time.perf_counter()
                try:
                    with gpu_lock:stage_cpu();result=value(*a,**k)
                    entry['output']=list(result);return result
                except BaseException as error:entry['error']=repr(error);raise
                finally:
                    entry['wall_ms']=(time.perf_counter()-started)*1000
                    with lock:calls.append(entry)
                    end()
            return invoke
    class PlausibilityAudit:
        def __getattr__(self,name):return getattr(plausibility,name)
        def decide(self,*a,**k):return self.invoke('decide',*a,**k)
        def decide_short_context(self,*a,**k):return self.invoke('decide_short_context',*a,**k)
        def invoke(self,name,*a,**k):
            begin();entry=dict(method=name,input=list(a),kwargs=k);started=time.perf_counter()
            try:
                with gpu_lock:
                    plausibility._model.to(device);plausibility.device=device
                    result=getattr(plausibility,name)(*a,**k);stage_cpu()
                entry.update(decision=result.decision,reason=result.reason);return result
            except BaseException as error:entry['error']=repr(error);raise
            finally:
                entry['wall_ms']=(time.perf_counter()-started)*1000
                with lock:ppl.append(entry)
                end()
    def router(value,native):
        if value['task_kind']!='window_refine':raise AssertionError('Local gate not allowed')
        begin();started=time.perf_counter()
        try:
            probability=float(classifier.predict_proba([feature_text(value)])[0,column])
            action='keep' if probability>=threshold else native
            entry=dict(input=value,probability=probability,action=action,gate_ms=(time.perf_counter()-started)*1000)
            with lock:gates.append(entry)
            return dict(action=action)
        finally:end()
    original_request=web._stream_request
    web._stream_request=lambda *a,**k:without_confidence(original_request(*a,**k))
    web.TransformersRefiner=lambda *_a,**_k:RefinerAudit()
    web.RepetitionPlausibility=lambda *_a,**_k:PlausibilityAudit()
    # Warm every backend with old train material, never with holdout labels.
    refiner.refine_with_numeric_rules('今天下午开会。')
    with gpu_lock:
        plausibility._model.to(device);plausibility.device=device
        plausibility.decide_short_context('请确确认编号。',1,3,'确');stage_cpu()
    classifier.predict_proba([feature_text(dict(task_kind='window_refine',domain='general',
        raw_text='请确认编号。',baseline_text='请确认编号。',available_context=''))])
    train=d.read_rows(ROOT/'results/gate_terminal_gain_pilot_20261011/sources.jsonl')[0]
    warm_audio,rate=sf.read(train['audio_path'],dtype='float32')
    with httpx.Client(timeout=120) as client:d.decode_wave(client,asr_url,warm_audio.astype('<f4').tobytes(),8000)
    source=root/'test_data';rows=d.read_rows(source/'tts.jsonl')+d.read_rows(root/'human_sources.jsonl')
    if len(rows)!=manifest['synthetic_audio']+manifest['human_audio']:raise ValueError('Missing heldout audio')
    random.Random(20261012).shuffle(rows)
    output=root/'streaming';output.mkdir(exist_ok=True)
    config=dict(selection_sha256=c.file_hash(selection_path),model_sha256=c.file_hash(model_path),threshold=threshold,
        runtime_protocol_sha256=c.file_hash(folder/'protocol.json'),adapter_sha256=c.file_hash(Path(__file__)),
        session_protocol='0.5sec packets paced by capture clock, max2 unacknowledged chunks',
        order='per-clip hashed order, reversed in second repetition',repetitions=repetitions,
        human_source_sha256=manifest['human_sources_sha256'],tts_source_sha256=c.file_hash(source/'tts.jsonl'),
        dtype=teacher['dtype'],refiner_weights_sha256=teacher['weights_sha256'],asr_url=asr_url,
        confidence='disabled',scope='actual_paced_WebSocket_and_HTTP_ASR_new_downstream_holdout',
        caches='fresh_per_session',warmups='Refiner,MLM,classifier,ASR prewarmed on old train fixtures',
        no_test_tuning=True,online_production_enabled=False)
    if (output/'configuration.json').exists() and json.loads((output/'configuration.json').read_text())!=config:raise ValueError('Resume configuration drift')
    c.write_json(output/'configuration.json',config);(output/'executed_code').mkdir(exist_ok=True)
    (output/'executed_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    completed=d.read_rows(output/'sessions.jsonl');done={(r['seed_id'],r['repetition'],r['policy']) for r in completed}
    normalizer=web.ZhITNNormalizer(folder/'native/libzh_itn_bridge.so')
    normalize=lambda s:d.lexical(normalizer.normalize(s).text)
    for repetition in range(repetitions):
        for i,row in enumerate(rows,1):
            audio,rate=sf.read(row['audio_path'],dtype='float32')
            if rate!=16000 or audio.ndim!=1 or c.file_hash(Path(row['audio_path']))!=row['audio_sha256']:raise ValueError('Heldout audio changed')
            pcm=audio.astype('<f4').tobytes()
            for policy in policy_order(row['seed_id'],repetition):
                if (row['seed_id'],repetition,policy) in done:continue
                calls.clear();ppl.clear();gates.clear()
                trace=output/f'{c.digest(row["seed_id"])[:16]}.{repetition}.{policy}.trace.jsonl'
                app=web.create_app(model,device,asr_url,'Chinese',256,trace,
                    entity_db=folder/'entities_snapshot.db',entity_fuzzy_mode='auto',refinement_gate_mode='tri_state',
                    final_refinement_mode='off',zh_itn_library=folder/'native/libzh_itn_bridge.so',
                    experimental_gate_router=router if policy=='learned' else None,experimental_gate_tasks=('window_refine',))
                try:
                    with TestClient(app) as client:final,events,timing=paced_session(client,pcm)
                except BaseException as error:
                    c.append_row(output/'failures.jsonl',dict(seed_id=row['seed_id'],repetition=repetition,
                        policy=policy,error=repr(error),model_calls=list(calls),plausibility_calls=list(ppl)))
                    raise
                drain=time.perf_counter()
                while True:
                    with lock:inflight=physical['inflight']
                    if not inflight:break
                    if time.perf_counter()-drain>60:raise RuntimeError('Late model call not drained')
                    time.sleep(.05)
                raw=str(final.get('raw_text',''));text=str(final.get('clean_text',''))
                record=dict(seed_id=row['seed_id'],template_group=row['template_group'],category=row['category'],
                    data_kind=row['data_kind'],human_gold=row.get('human_gold',False),reference_kind=row.get('reference_kind','unverified_synthetic_script'),
                    repetition=repetition,policy=policy,order=policy_order(row['seed_id'],repetition),
                    raw_text=raw,final_text=text,intended_text=row['intended_text'],
                    normalized_distance=c.edit_distance(normalize(text),normalize(row['intended_text'])),
                    normalized_reference_chars=len(normalize(row['intended_text'])),
                    lexical_distance=c.edit_distance(d.lexical(text),d.lexical(row['intended_text'])),
                    lexical_reference_chars=len(d.lexical(row['intended_text'])),
                    final=final,events=events,timing=timing,model_calls=list(calls),plausibility_calls=list(ppl),gate_calls=list(gates),
                    audio_sha256=row['audio_sha256'],pcm_sha256=c.digest(pcm),post_final_drain_seconds=time.perf_counter()-drain)
                c.append_row(output/'sessions.jsonl',record);completed.append(record)
                print(f'LIVE repeat={repetition+1} {i}/{len(rows)} {policy} '+json.dumps(dict(id=row['seed_id'],text=text,
                    calls=len(calls),tail=timing['finish_tail_seconds'],audio_lag=timing['completion_lag_from_audio_clock_seconds']),ensure_ascii=False),flush=True)
    if c.file_hash(selection_path)!=manifest['selection_sha256'] or c.file_hash(model_path)!=manifest['model_sha256']:raise ValueError('Policy changed during test')
    c.verify_frozen(folder)
    c.write_json(output/'job_state.json',dict(status='complete',sessions=len(completed),expected=len(rows)*repetitions*2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--asr-url',default='http://127.0.0.1:8766');p.add_argument('--repetitions',type=int,default=2)
    a=p.parse_args();run(a.root.resolve(),a.device,a.asr_url,a.repetitions)
