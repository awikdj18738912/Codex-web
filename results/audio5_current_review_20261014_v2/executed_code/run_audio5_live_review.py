"""Run full audio through a fresh current-source Web app; preserve all events."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    # Frozen system code comes before the working tree on the import path.
    sys.path.insert(0,str(args.source_root.resolve()))
    import torch
    from threadpoolctl import threadpool_limits
    from fastapi.testclient import TestClient
    from system import web_app
    torch.set_num_threads(2);threadpool_limits(limits=2)
    out=args.output.resolve()
    audio=subprocess.run(['ffmpeg','-v','error','-i',str(args.audio),'-ac','1','-ar','16000',
                          '-f','f32le','pipe:1'],check=True,stdout=subprocess.PIPE).stdout
    trace=out/'current_live.jsonl'
    if trace.exists():raise FileExistsError('Never overwrite a live run')
    app=web_app.create_app(args.model,args.device,args.asr_url,'Chinese',256,trace,
        entity_db=out/'entities_snapshot.db',entity_fuzzy_mode='auto',refinement_gate_mode='tri_state',
        learned_window_gate_mode='on',confidence_supplement_mode='off',final_refinement_mode='off',
        zh_itn_library=args.zh_itn,entity_audio_repair=True)
    errors=[];started=time.perf_counter()
    with TestClient(app) as client,(out/'current_live.events.jsonl').open('w') as events:
        (out/'current_live.health.json').write_text(json.dumps(client.get('/health').json(),ensure_ascii=False,indent=2)+'\n')
        with client.websocket_connect('/ws/stream?mode=streaming&domain=general') as ws:
            first=ws.receive_json();events.write(json.dumps(first,ensure_ascii=False)+'\n')
            if first.get('event')!='ready':raise RuntimeError(str(first))
            for index,offset in enumerate(range(0,len(audio),64000)):
                ws.send_bytes(audio[offset:offset+64000])
                for _ in range(3000):
                    payload=ws.receive_json();events.write(json.dumps(payload,ensure_ascii=False)+'\n');events.flush()
                    if payload.get('event')=='error':errors.append(payload);raise RuntimeError(str(payload))
                    if payload.get('event')=='chunk_ack':break
                else:raise RuntimeError('No chunk acknowledgement')
                if (index+1)%30==0:
                    progress=dict(audio_seconds=min(offset+64000,len(audio))/64000,
                                  elapsed_seconds=time.perf_counter()-started)
                    (out/'current_live.progress.json').write_text(json.dumps(progress)+'\n')
                    print('LIVE_PROGRESS',progress,flush=True)
            ws.send_json({'event':'finish'})
            for _ in range(5000):
                payload=ws.receive_json();events.write(json.dumps(payload,ensure_ascii=False)+'\n');events.flush()
                if payload.get('event')=='error':raise RuntimeError(str(payload))
                if payload.get('event')=='final':
                    (out/'current_live.final.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
                    break
            else:raise RuntimeError('No final event')
    stats=payload['refiner_session_stats']
    if not stats['stats_complete'] or stats['failed_call_count']:raise RuntimeError('Incomplete real model calls')
    summary=dict(seconds=time.perf_counter()-started,decoded_audio_seconds=len(audio)/64000,
        fresh_ASR_and_real_refiner=True,source_root=str(args.source_root),
        primary_policy='tri_state_and_learned_on',confidence_supplement='off',
        entity_audio_repair=payload.get('entity_audio_repair'),stats=stats,errors=errors)
    (out/'current_live.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('LIVE_DONE',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--asr-url',default='http://127.0.0.1:8876')
    parser.add_argument('--model',type=Path,default=Path('/home/aim0/data/models/ASR/AgenticASR-Refiner'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--zh-itn',type=Path,default=Path('/home/aim0/桌面/tan/ASR/Codex-web/zh-itn/zh-itn/build/libzh_itn_bridge.so'))
    main(parser.parse_args())
