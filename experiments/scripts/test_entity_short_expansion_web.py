"""Actual Refiner integration for the one veto plus historical controls."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import urlencode

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from validate_entity_confirmation import OUT,BASE,classes,read,write
MODEL=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner')
DEST=OUT/'short_expansion_v3'


def main():
    from fastapi.testclient import TestClient
    from benchmark_multimed_entities import Runner
    import system.entity_audio_repair as repair_module
    old,new=classes();fixtures=[]
    for cohort,folders,input_file in [('development_200',('holdout','holdout_news'),OUT/'holdout_audio.jsonl'),
                                      ('validation_200',('holdout_v2',),OUT/'short_expansion/holdout_audio.jsonl')]:
        samples={r['id']:r for folder in folders for r in read(OUT/folder/'samples.jsonl')}
        data={r['id']:r for r in read(input_file)}
        changed=json.loads((DEST/(cohort+'_changes.json')).read_text())
        for change in changed:
            sample=samples[change['id']];record=data[change['id']]
            fixtures.append(dict(id=change['id'],pcm_path=sample['pcm_path'],trace=record['baseline_trace'],
                db=BASE/'dataset/library/entities.db',expected_old=change['before'],expected_new=change['after'],
                reference=sample['reference'],cohort=cohort))
    # Controls are fixed diagnostics, not extra independent accuracy samples.
    recent=ROOT/'results/generic_entity_optimization_20261010/holdout'
    rid='fleurs:data-00000-of-00002.arrow:358:1730'
    sample=next(s for s in json.loads((recent/'manifest.json').read_text())['samples'] if s['id']==rid)
    record=next(r for r in read(recent/'audio_results.jsonl') if r['id']==rid)
    fixtures.append(dict(id=rid,pcm_path=sample['audio_path'],trace=record['request_trace'],
        db=recent/'libraries'/sample['source']/'entities.db',expected_old=record['production_final']['text'],
        expected_new=record['production_final']['text'],cohort='historical_endpoint_omission'))
    olddata={r['id']:r for r in read(BASE/'full_evaluation/audio.jsonl')}
    samples={r['id']:r for r in read(BASE/'dataset/samples.jsonl')}
    for rid in ('leb1_005a4068ad48e06eedac','leb1_1ef4d56ce461f58ef3da','leb1_8c2510722ec5a19f9656'):
        record=olddata[rid]
        fixtures.append(dict(id=rid,pcm_path=samples[rid]['pcm_path'],trace=record['trace'],
            db=BASE/'dataset/library/entities.db',expected_old=record['production_final']['text'],
            expected_new=record['production_final']['text'],reference=samples[rid]['reference'],cohort='unchanged_control'))
    write(DEST/'web_plan.json',[{k:str(v) if isinstance(v,Path) else v for k,v in f.items() if k!='trace'} for f in fixtures])
    runner=Runner(SimpleNamespace(output=DEST,library=BASE/'dataset/library',model=MODEL,device='cuda:0',
                                  asr_url='http://127.0.0.1:8766',mode='streaming',chunk_seconds=2.0))
    active={}
    async def request(url,endpoint,session_id=None,data=b'',params=None,**kwargs):
        trace=active['fixture']['trace'];cursor=active['cursor']
        if endpoint=='/stream/cancel' and (cursor==len(trace) or trace[cursor]['endpoint']!=endpoint):return dict(cancelled=True)
        assert cursor<len(trace)
        expected=trace[cursor]
        assert endpoint==expected['endpoint'] and session_id==expected['session_id']
        assert (params or {}).get('context','')==expected['context']
        assert hashlib.sha256(data).hexdigest()==expected['data_sha256']
        active['cursor']+=1
        return deepcopy(expected['result'])
    runner.web._stream_request_async=request
    destination=DEST/'web_results.jsonl';done={(r['id'],r['arm']) for r in read(destination)} if destination.exists() else set()
    for fixture in fixtures:
        active['fixture']=fixture;pcm=Path(fixture['pcm_path']).read_bytes()
        for arm,guard in [('baseline',old),('proposed',new)]:
            if (fixture['id'],arm) in done:continue
            # Web imports this class inside its socket handler. Patch the
            # source module in this isolated test process, not Web globals.
            repair_module.EntityAudioRepair=guard
            app=runner.web.create_app(MODEL,'cuda:0','http://127.0.0.1:8766','Chinese',256,
                DEST/'service_logs'/(arm+'.jsonl'),entity_db=fixture['db'],entity_fuzzy_mode='auto',
                refinement_gate_mode='tri_state',final_refinement_mode='off',
                zh_itn_library=ROOT/'zh-itn/zh-itn/build/libzh_itn_bridge.so',entity_asr_context=False,entity_audio_repair=True)
            active['cursor']=0;runner.calls.clear()
            with TestClient(app) as client:
                query=urlencode(dict(mode='streaming',domain='general',language='Chinese',asr_chunk_seconds=2.0,
                                     entity_asr_context=0,entity_audio_repair=1))
                with client.websocket_connect('/ws/stream?'+query) as socket:
                    assert socket.receive_json()['event']=='ready'
                    for offset in range(0,len(pcm),64000):
                        socket.send_bytes(pcm[offset:offset+64000])
                        while True:
                            event=socket.receive_json()
                            if event.get('event')=='error':raise RuntimeError(event)
                            if event.get('event')=='chunk_ack':break
                    socket.send_json(dict(event='finish'))
                    while True:
                        event=socket.receive_json()
                        if event.get('event')=='error':raise RuntimeError(event)
                        if event.get('event')=='final':break
            expected_raw=fixture['expected_old' if arm=='baseline' else 'expected_new']
            assert event['raw_text'].strip()==expected_raw.strip(),(fixture['id'],arm,event['raw_text'],expected_raw)
            assert active['cursor']==len(fixture['trace'])
            with destination.open('a') as f:f.write(json.dumps(dict(id=fixture['id'],arm=arm,result=event,text_refinement_calls=list(runner.calls)),ensure_ascii=False)+'\n')
            print('WEB',fixture['id'],arm,event.get('clean_text',''),flush=True)
    results={(r['id'],r['arm']):r for r in read(destination)}
    assert len(results)==len(fixtures)*2
    for fixture in fixtures:
        a=results[fixture['id'],'baseline']['result'];b=results[fixture['id'],'proposed']['result']
        if fixture['cohort']!='development_200':assert a['clean_text']==b['clean_text'],fixture['id']
        else:
            assert '越秀区' in a['clean_text'] and '越秀区' not in b['clean_text']
    write(DEST/'web_verification.json',dict(samples=len(fixtures),paired_sessions=len(results),
        actual_refiner_loaded=True,same_real_ASR_trace_verified=True,
        unrelated_controls_unchanged=True,bounded_false_expansion_removed_after_Web=True))


if __name__=='__main__':main()
