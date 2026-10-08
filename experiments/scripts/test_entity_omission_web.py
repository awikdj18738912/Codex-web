"""Real-ASR/GPU-Refiner replay checks for the internal-omission rule."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

from evaluate_entity_library import ROOT, decode, read_jsonl, append_jsonl, write_json
from evaluate_entity_omission_rule import OUT, SCENES, profiles
from system.entity_store import EntityStore

WEB = OUT / 'web_replay_opt_in'


def prepare():
    WEB.mkdir(exist_ok=True)
    selection = json.loads((SCENES/'selection.json').read_text())
    ids = {'MultiMed_Chinese_train_0579','MultiMed_Chinese_train_0576','MultiMed_Chinese_train_0616'}
    rows = [r for r in selection['samples'] if r['id'] in ids]
    write_json(WEB/'selection.json', dict(seed=selection['seed'], noise_snr_db=selection['noise_snr_db'], samples=rows))
    definitions = profiles(EntityStore(SCENES/'orthopedics.db').list_entities(domain='orthopedics'))
    store = EntityStore(WEB/'terms.db')
    for d in definitions:
        item = store.upsert_entity(d.canonical_text, domain='orthopedics', entity_type=d.entity_type,
            normalization_policy=d.normalization_policy, aliases=d.aliases, priority=d.priority, source=d.source)
        if d.spelling_context:
            p=d.spelling_context
            store.set_spelling_context(item.entity_id, keywords=dict(p.keywords), minimum_score=p.minimum_score,
                                       source=p.source, allow_internal_omission=p.allow_internal_omission)
    write_json(WEB/'protocol.json', dict(role='Development regression examples; not independent generalization',
        method='Fresh real ASR responses replayed identically into unchanged WebSocket handler and real GPU Refiner',
        conditions=['baseline_omission_disabled','proposed_omission_enabled'],
        same_dictionary=True, new_error_aliases=False,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [WEB/'selection.json',WEB/'terms.db']}))


def run():
    import system.web_app as web
    import system.entity_matcher as matching
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    from system.zh_itn_bridge import DEFAULT_LIBRARY
    from fastapi.testclient import TestClient
    model_path=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner')
    model=TransformersRefiner(model_path,'cuda:0',256)
    calls=[]
    class Audited:
        def __getattr__(self,name):
            value=getattr(model,name)
            if not callable(value) or not name.startswith('refine'):return value
            def invoke(*args,**kwargs):
                result=value(*args,**kwargs)
                calls.append(dict(method=name,input=list(args),kwargs=kwargs,output=result))
                return result
            return invoke
    audited=Audited();web.TransformersRefiner=lambda *a,**k:audited
    plausibility=RepetitionPlausibility('cuda:0');web.RepetitionPlausibility=lambda *a,**k:plausibility
    state={}
    async def replay(_url,endpoint,session_id=None,data=b'',params=None,**kwargs):
        if endpoint=='/stream/start':state['index']=0;return dict(session_id='fixed-asr-replay',chunk_seconds=.5)
        if endpoint=='/stream/chunk':
            i=state['index'];state['index']+=1;return state['row']['chunk_payloads'][i]
        if endpoint=='/stream/finish':return state['row']['final_payload']
        if endpoint=='/stream/cancel':return dict(cancelled=True)
        raise ValueError(endpoint)
    web._stream_request_async=replay
    app=web.create_app(model_path,'cuda:0','http://replayed-asr.invalid','Chinese',256,None,
        entity_db=WEB/'terms.db',entity_fuzzy_mode='auto',refinement_gate_mode='tri_state',
        final_refinement_mode='off',zh_itn_library=DEFAULT_LIBRARY)
    original=matching.context_omission_evidence
    completed={(r['id'],r['variant'],r['condition']) for r in read_jsonl(WEB/'results.jsonl')}
    try:
        for row in read_jsonl(WEB/'asr.jsonl'):
            for condition in ['baseline','proposed']:
                if (row['id'],row['variant'],condition) in completed:continue
                # Only the new branch is disabled in the control. No thresholds,
                # metadata, ASR outputs, or model parameters differ.
                matching.context_omission_evidence=original if condition=='proposed' else lambda *a,**k:()
                state['row']=row;calls.clear();started=time.perf_counter();events=[]
                with TestClient(app) as client:
                    with client.websocket_connect('/ws/stream?mode=offline&domain=orthopedics&entity_asr_context=0') as socket:
                        ready=socket.receive_json();assert ready['event']=='ready'
                        for payload in row['chunk_payloads']:
                            socket.send_bytes(b'\0\0\0\0')
                            while True:
                                e=socket.receive_json();events.append(e)
                                if e['event']=='error':raise RuntimeError(e)
                                if e['event']=='chunk_ack':break
                        socket.send_json(dict(event='finish'))
                        while True:
                            e=socket.receive_json();events.append(e)
                            if e['event']=='error':raise RuntimeError(e)
                            if e['event']=='final':break
                assert e['raw_text']==row['raw_text']
                record=dict(id=row['id'],variant=row['variant'],condition=condition,
                    result=e,model_calls=list(calls),elapsed_seconds=time.perf_counter()-started)
                append_jsonl(WEB/'results.jsonl',record)
                print(row['id'],row['variant'],condition,e['clean_text'],flush=True)
    finally:
        matching.context_omission_evidence=original


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','asr','run']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='asr':decode(SimpleNamespace(output=WEB,asr_url='http://127.0.0.1:8766'))
    else:run()
