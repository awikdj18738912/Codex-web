"""Experiment-only raw ASR relay. No library, refiner, or text cleanup."""
import argparse
import asyncio
from pathlib import Path
import sys
import uuid

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'source_snapshot'))
# Use the identical HTTP adapter, without constructing the production pipeline.
from system.web_app import _stream_request_async
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn


def create(asr_url):
    app=FastAPI()
    @app.get('/health')
    def health():
        return dict(ok=True,scope='raw ASR relay only',asr_url=asr_url,
                    entity_library=False,entity_context=False,entity_audio_review=False,
                    refiner=False,numeric_normalization=False,text_cleanup=False)
    @app.websocket('/ws/stream')
    async def stream(ws:WebSocket):
        await ws.accept();sid=None;finished=False;calls=[]
        async def request(endpoint,data=b'',params=None):
            query=dict(params or {})
            assert not query.get('context')
            r=await _stream_request_async(asr_url,endpoint,sid,data,query)
            calls.append(dict(endpoint=endpoint,bytes=len(data),context_supplied=False))
            return r
        try:
            start=await request('/stream/start',params=dict(language='Chinese',chunk_seconds=2.0))
            sid=str(start['session_id']);assert not start.get('context_applied')
            assert not start.get('context_digest')
            await ws.send_json(dict(event='ready',trace_id=uuid.uuid4().hex,
                asr_session_id=sid,asr_chunk_seconds=start.get('chunk_seconds'),
                entity_asr_context_requested=False,entity_asr_context_applied=False,
                entity_asr_context_term_count=0,entity_asr_context_digest=None,
                entity_audio_repair_requested=False,refiner_enabled=False,
                postprocessing_enabled=False))
            index=0
            while True:
                message=await ws.receive()
                if message['type']=='websocket.disconnect':break
                if message.get('text'):
                    import json
                    if json.loads(message['text']).get('event')!='finish':continue
                    payload=await request('/stream/finish');finished=True
                    text=payload.get('text','');assert isinstance(text,str)
                    await ws.send_json(dict(event='final',raw_text=text,clean_text=text,
                        entity_asr_context_requested=False,entity_asr_context_applied=False,
                        entity_audio_repair_requested=False,session_refiner_call_count=0,
                        session_refiner_retry_call_count=0,numeric_normalization_enabled=False,
                        text_cleanup_enabled=False,asr_payload=payload,asr_requests=calls))
                    break
                audio=message.get('bytes')
                if audio:
                    await request('/stream/chunk',audio);index+=1
                    await ws.send_json(dict(event='chunk_ack',chunk_index=index))
        except WebSocketDisconnect:pass
        except Exception as e:
            try:await ws.send_json(dict(event='error',detail=f'{type(e).__name__}: {e}'))
            except Exception:pass
            raise
        finally:
            if sid and not finished:
                try:await asyncio.wait_for(request('/stream/cancel'),5)
                except Exception:pass
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True)
    p.add_argument('--asr-url',default='http://127.0.0.1:8766');a=p.parse_args()
    uvicorn.run(create(a.asr_url),host='127.0.0.1',port=a.port,ws_per_message_deflate=False,log_level='warning')
