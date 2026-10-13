"""Real paced WebSocket sessions against existing ASR; no test adaptation.

Uses an in-process Web client, actual HTTP ASR and actual original-precision
Refiner. Each policy has its own session/windows/review caches. Synthetic audio
scripts remain unverified, so results are diagnostics rather than acceptance.
"""
from __future__ import annotations
import argparse
import json
import inspect
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.scripts import collect_gate_pairs as c,synthetic_gate_data as data
from system.weak_refinement_gate import WeakRefinementGate


def without_confidence(value):
    if isinstance(value,list):return [without_confidence(v) for v in value]
    if not isinstance(value,dict):return value
    return {k: {} if k=='confidence_metadata' else None if k in {'confidence','asr_confidence'}
            else without_confidence(v) for k,v in value.items()}


def session(client,pcm,chunk_samples,pace):
    events=[];failure=[];final=[];started=time.perf_counter();finish_sent=[]
    with client.websocket_connect('/ws/stream?mode=streaming&domain=general') as ws:
        ready=ws.receive_json()
        if ready.get('event')!='ready':raise RuntimeError('Web session not ready: '+str(ready))
        def receive():
            try:
                while True:
                    event=ws.receive_json();events.append(dict(elapsed_seconds=time.perf_counter()-started,payload=event))
                    if event.get('event')=='final':final.append(event);break
                    if event.get('event')=='error':raise RuntimeError(str(event))
            except BaseException as error:failure.append(repr(error))
        reader=threading.Thread(target=receive,daemon=True);reader.start()
        audio_start=time.perf_counter()
        for offset in range(0,len(pcm),chunk_samples*4):
            ws.send_bytes(pcm[offset:offset+chunk_samples*4])
            if pace:
                due=audio_start+min(len(pcm),offset+chunk_samples*4)/4/16000
                remaining=due-time.perf_counter()
                if remaining>0:time.sleep(remaining)
            if failure:raise RuntimeError(failure)
        finish_sent.append(time.perf_counter());ws.send_json({'event':'finish'})
        reader.join(timeout=90)
        if reader.is_alive() or failure or not final:raise RuntimeError('Stream did not finish cleanly: '+str(failure))
        tail=events[-1]['elapsed_seconds']-(finish_sent[0]-started)
    return final[0],events,dict(total_seconds=events[-1]['elapsed_seconds'],finish_tail_seconds=max(0,tail),
                              audio_seconds=len(pcm)/4/16000)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--gate-scope',choices=['both','window'],default='both')
    parser.add_argument('--prewarm-plausibility',action='store_true')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--asr-url',default='http://127.0.0.1:8766')
    parser.add_argument('--unpaced',action='store_true');parser.add_argument('--limit',type=int)
    args=parser.parse_args();root=args.root.resolve();source=(args.source or root/'test_data').resolve()
    output=(args.output or root/'streaming_test').resolve();output.mkdir(exist_ok=True)
    selection_path=root/'training/selection.json';selection=json.loads(selection_path.read_text());policy_hash=c.file_hash(selection_path)
    checkpoint=root/'training'/selection['primary_model'];gate=WeakRefinementGate(checkpoint,device='cpu')
    import torch
    import soundfile as sf
    from fastapi.testclient import TestClient
    torch.set_num_threads(2)
    folder=(args.runtime or root/'train_data/pairs_full').resolve();c.verify_frozen(folder);web=c.load_system(folder)
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    model=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner');teacher_cfg=json.loads((folder/'run_configuration.json').read_text())
    if {p.name:c.file_hash(p) for p in model.glob('*.safetensors')}!=teacher_cfg['weights_sha256']:raise RuntimeError('Refiner weight drift')
    refiner=TransformersRefiner(model,args.device,256);plausibility=RepetitionPlausibility('cpu')
    if str(next(refiner._model.parameters()).dtype)!=teacher_cfg['dtype']:raise RuntimeError('Refiner precision drift')
    calls=[];ppl=[];gate_calls=[];active=dict(domain='general');physical=dict(inflight=0)
    lock=threading.Lock()
    def stage_cpu():
        if plausibility._model is not None:plausibility._model.to('cpu')
        plausibility.device='cpu';torch.cuda.empty_cache()
    class RefinerAudit:
        def __getattr__(self,name):
            value=getattr(refiner,name)
            if not callable(value) or not(name.startswith('refine') or name.startswith('review_')):return value
            def invoke(*a,**k):
                with lock:physical['inflight']+=1
                started=time.perf_counter();entry=dict(method=name,input=list(a),kwargs=k)
                try:
                    stage_cpu();result=value(*a,**k);entry['output']=list(result);return result
                except BaseException as error:
                    entry.update(error=repr(error));raise
                finally:
                    entry['wall_ms']=(time.perf_counter()-started)*1000
                    with lock:calls.append(entry);physical['inflight']-=1
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
            finally:entry['wall_ms']=(time.perf_counter()-started)*1000;ppl.append(entry);stage_cpu()
    def router(value,native):
        if args.gate_scope == 'window' and value['task_kind'] != 'window_refine':
            return dict(action=native,reason='native_local_fallback')
        value=dict(value,domain=active['domain'])
        if value['task_kind']=='window_refine':value['baseline_text']=web.clean_transcript_deterministically(value['baseline_text'])
        started=time.perf_counter();decision=gate.decide_with_fallback(value,native)
        gate_calls.append(dict(input=value,decision=decision,gate_ms=(time.perf_counter()-started)*1000))
        return decision
    web.TransformersRefiner=lambda *_a,**_k:RefinerAudit()
    web.RepetitionPlausibility=lambda *_a,**_k:PlausibilityAudit()
    original_request=web._stream_request
    def controlled_request(*a,**k):return without_confidence(original_request(*a,**k))
    web._stream_request=controlled_request
    refiner.refine_with_numeric_rules('今天下午开会。')
    if args.prewarm_plausibility:
        if not plausibility._ensure_model():raise RuntimeError('Plausibility prewarm failed')
    normalizer=web.ZhITNNormalizer(folder/'native/libzh_itn_bridge.so')
    normalize=lambda s:data.lexical(normalizer.normalize(s).text)
    rows=data.read_rows(source/'tts.jsonl');rows=rows[:args.limit] if args.limit else rows
    configuration=dict(scope='actual_WebSocket_and_HTTP_ASR_synthetic_streaming_diagnostic',pace=not args.unpaced,
        chunk_seconds=.5,asr_url=args.asr_url,policy_sha256=policy_hash,checkpoint=str(checkpoint),
        gate_weights_sha256=c.file_hash(checkpoint/'gate.pt'),teacher_protocol_sha256=c.file_hash(folder/'protocol.json'),
        confidence='disabled_to_match_training_no_calibrated_scores',window_context='empty',
        baseline_preprocessor='deterministic_cleanup_matches_rules_preparation',
        independent_session_caches=True,human_gold=False,audio_script_verified=False,
        runner_sha256=c.file_hash(Path(__file__)),source_texts_sha256=c.file_hash(source/'texts.jsonl'))
    configuration.update(gate_scope=args.gate_scope,runtime=str(folder),plausibility_prewarmed=args.prewarm_plausibility,
                         data_role='consumed_regression' if (source/'test_consumed.json').exists() else 'unconsumed_diagnostic')
    cfgpath=output/'configuration.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text())!=configuration:raise RuntimeError('Streaming resume configuration changed')
    c.write_json(cfgpath,configuration);(output/'executed_code').mkdir(exist_ok=True)
    (output/'executed_code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    completed=data.read_rows(output/'sessions.jsonl');done={(r['seed_id'],r['policy']) for r in completed}
    for index,row in enumerate(rows,1):
        audio,rate=sf.read(row['audio_path'],dtype='float32');assert rate==16000 and audio.ndim==1
        if c.file_hash(Path(row['audio_path']))!=row['audio_sha256']:raise RuntimeError('Audio changed')
        pcm=audio.astype('<f4').tobytes();active['domain']=row['domain']
        for policy in ['current','learned']:
            if (row['seed_id'],policy) in done:continue
            calls.clear();ppl.clear();gate_calls.clear()
            trace=output/f'{row["seed_id"]}.{policy}.jsonl'
            scope_options = {}
            if args.gate_scope == 'window' and 'experimental_gate_tasks' in inspect.signature(web.create_app).parameters:
                scope_options['experimental_gate_tasks'] = ('window_refine',)
            elif args.runtime:
                scope_options['experimental_gate_tasks'] = ('window_refine','local_edit_review')
            app=web.create_app(model,args.device,args.asr_url,'Chinese',256,trace,
                entity_db=folder/'entities_snapshot.db',entity_fuzzy_mode='auto',refinement_gate_mode='tri_state',
                final_refinement_mode='off',zh_itn_library=folder/'native/libzh_itn_bridge.so',
                experimental_gate_router=router if policy=='learned' else None, **scope_options)
            with TestClient(app) as client:final,events,timing=session(client,pcm,8000,not args.unpaced)
            drained=time.perf_counter()
            while physical['inflight']:
                if time.perf_counter()-drained>60:raise RuntimeError('Refiner remains in flight after final')
                time.sleep(.05)
            final_text=str(final.get('clean_text',''));raw=str(final.get('raw_text',''))
            record=dict(seed_id=row['seed_id'],template_group=row['template_group'],category=row['category'],policy=policy,
                intended_script=row['intended_text'],spoken_script=row['spoken_text'],raw_text=raw,final_text=final_text,
                intended_chars=len(normalize(row['intended_text'])),intended_distance=c.edit_distance(normalize(final_text),normalize(row['intended_text'])),
                raw_distance=c.edit_distance(normalize(raw),normalize(row['intended_text'])),
                final=final,timing=timing,model_calls=list(calls),plausibility_calls=list(ppl),gate_calls=list(gate_calls),
                event_count=len(events),post_final_drain_seconds=time.perf_counter()-drained,
                audio_sha256=row['audio_sha256'],audio_script_verified=False)
            c.append_row(output/'sessions.jsonl',record);completed.append(record)
            c.write_json(output/f'{row["seed_id"]}.{policy}.events.json',events)
            print(f'STREAM {index}/{len(rows)} {policy} '+json.dumps(dict(text=final_text,calls=len(calls),tail=timing['finish_tail_seconds']),ensure_ascii=False),flush=True)
    if c.file_hash(selection_path)!=policy_hash:raise RuntimeError('Policy changed during test')
    import numpy as np
    summary={}
    for policy in ['current','learned']:
        values=[r for r in completed if r['policy']==policy]
        summary[policy]=dict(sessions=len(values),exact_intended=sum(r['intended_distance']==0 for r in values),
            intended_micro_cer=sum(r['intended_distance'] for r in values)/max(1,sum(r['intended_chars'] for r in values)),
            logical_refiner_calls=sum(len(r['model_calls']) for r in values),plausibility_calls=sum(len(r['plausibility_calls']) for r in values),
            gate_total_ms=sum(g['gate_ms'] for r in values for g in r['gate_calls']),
            finish_tail_p50_seconds=float(np.percentile([r['timing']['finish_tail_seconds'] for r in values],50)),
            finish_tail_p95_seconds=float(np.percentile([r['timing']['finish_tail_seconds'] for r in values],95)),
            failed_refiner_calls=sum(bool(c.get('error')) for r in values for c in r['model_calls']))
    pairs={sid:{r['policy']:r for r in completed if r['seed_id']==sid} for sid in {r['seed_id'] for r in completed}}
    differences=[dict(seed_id=sid,raw_equal=normalize(p['current']['raw_text'])==normalize(p['learned']['raw_text']),
        current=p['current']['final_text'],learned=p['learned']['final_text'],
        current_distance=p['current']['intended_distance'],learned_distance=p['learned']['intended_distance'])
        for sid,p in pairs.items() if normalize(p['current']['final_text'])!=normalize(p['learned']['final_text'])]
    report=dict(by_policy=summary,differences=differences,
        ASR_equal_pairs=sum(normalize(p['current']['raw_text'])==normalize(p['learned']['raw_text']) for p in pairs.values()),
        complete=len(completed)==2*len(rows),references='unverified_synthesis_scripts',scope=configuration['scope'])
    c.write_json(output/'summary.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
