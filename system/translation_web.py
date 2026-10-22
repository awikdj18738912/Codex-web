"""Optional GPU translation front end for the existing Codex-web application."""
import asyncio
import json
import math
from pathlib import Path
import re
import time
import uuid

import numpy as np
from fastapi import File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse


LANGUAGES={'en':('English','英语'),'fr':('French','法语'),'de':('German','德语'),
           'es':('Spanish','西班牙语'),'ru':('Russian','俄语'),'ja':('Japanese','日语'),
           'ko':('Korean','韩语'),'ar':('Arabic','阿拉伯语')}


def source_pieces(text, limit=700):
    """Bound each source window, preserving punctuation/characters exactly."""
    sentences=[]
    start=0
    for match in re.finditer(r'(?<!\d)[.!?。！？](?!\d)|\n',text):
        end=match.end()
        sentences.append(text[start:end]);start=end
    if start<len(text):
        sentences.append(text[start:])
    pieces=[];active=''
    for sentence in sentences:
        if active and len(active)+len(sentence)>limit:
            pieces.append(active);active=''
        while len(sentence)>limit:
            pieces.append(sentence[:limit]);sentence=sentence[limit:]
        active+=sentence
    if active:pieces.append(active)
    return pieces


# Preserve the import used by existing integrations while adopting fixed groups.
from .translation_windows import StableTranslationSnapshots as TranslationSnapshots
from .translation_scheduler import run_snapshot_pipeline, thread_call


def decode_audio(path, max_seconds=300):
    import av
    arrays=[];samples=0
    with av.open(str(path)) as container:
        resampler=av.AudioResampler(format='fltp',layout='mono',rate=16000)
        def append(frames):
            nonlocal samples
            for frame in frames:
                samples+=frame.samples
                if samples>16000*max_seconds:
                    raise ValueError('音频最长支持300秒，请截取后重试。')
                arrays.append(frame.to_ndarray().reshape(-1))
        for frame in container.decode(audio=0):
            append(resampler.resample(frame))
        append(resampler.resample(None))
    if not arrays:
        raise ValueError('文件中没有可解码音频。')
    return np.concatenate(arrays).astype('<f4',copy=False)


def install_translation_routes(app, translate, *, asr_url, output_dir):
    from .web_app import _stream_request
    import urllib.request
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    upload_dir=directory/'uploads';upload_dir.mkdir(exist_ok=True)
    jobs={}
    busy=asyncio.Lock()
    app.state.translation_busy=busy

    def asr_ready():
        try:
            with urllib.request.urlopen(asr_url.rstrip('/')+'/health',timeout=2) as response:
                return json.load(response).get('ok') is True
        except Exception:
            return False

    @app.get('/translation.js')
    def javascript():
        return FileResponse(Path(__file__).parent/'web'/'translation.js',media_type='application/javascript',
                            headers={'Cache-Control':'no-store'})

    @app.get('/api/translation/health')
    async def translation_health():
        return {'ready':True,'model':'LMT-60-0.6B','target_language':'zh',
            'languages':[{'code':code,'name':value[1]} for code,value in LANGUAGES.items()],
            'audio_ready':await asyncio.to_thread(asr_ready),'busy':busy.locked(),
            'pipeline':'ASR → LMT → Codex-web fixed_k3_context_v2 Refiner',
            'scheduling':'latest_snapshot_v2','pending_per_stage':1,
            'max_audio_seconds':300,'max_file_bytes':50*1024*1024}

    def validate(language,domain,interval):
        if language not in LANGUAGES:
            raise ValueError('请选择支持的源语言。')
        if not isinstance(domain,str) or not domain.strip() or len(domain)>64:
            raise ValueError('术语领域无效。')
        interval=float(interval)
        if not math.isfinite(interval) or not 0<=interval<=5000:
            raise ValueError('回放间隔需在0到5000毫秒之间。')
        return interval/1000

    def prune_uploads():
        for key,job in list(jobs.items()):
            if time.monotonic()-job['created']>600:
                job['path'].unlink(missing_ok=True);jobs.pop(key,None)

    @app.post('/api/translation/audio')
    async def upload_audio(file: UploadFile=File(...),language: str=Form('en'),
        domain: str=Form('trade'),chunk_seconds: float=Form(2),interval_ms: float=Form(0)):
        path=None
        try:
            interval=validate(language,domain,interval_ms)
            if not math.isfinite(chunk_seconds) or not .5<=chunk_seconds<=5:
                raise ValueError('音频分块需在0.5到5秒之间。')
            prune_uploads()
            if len(jobs)>=2:
                raise HTTPException(429,'待处理文件已满，请完成或取消已有任务。')
            ident=uuid.uuid4().hex;path=upload_dir/(ident+'.audio')
            size=0
            with path.open('xb') as dst:
                while block:=await file.read(1024*1024):
                    size+=len(block)
                    if size>50*1024*1024:
                        raise ValueError('文件不能超过50 MB。')
                    dst.write(block)
            if not size:
                raise ValueError('音频文件为空。')
            jobs[ident]={'path':path,'language':language,'domain':domain,'interval':interval,
                'chunk_seconds':chunk_seconds,'created':time.monotonic()}
            return {'id':ident,'websocket':'/ws/translation/audio/'+ident}
        except ValueError as exc:
            if path:path.unlink(missing_ok=True)
            raise HTTPException(400,str(exc)) from exc
        finally:
            await file.close()

    @app.delete('/api/translation/audio/{ident}')
    async def cancel_uploaded_audio(ident: str):
        job=jobs.pop(ident,None)
        if job:
            job['path'].unlink(missing_ok=True)
        return {'cancelled':True}

    async def run(ws, config, snapshots):
        await ws.accept()
        if busy.locked():
            await ws.send_json({'event':'error','detail':'模型正在处理另一任务，请稍后再试。'})
            await ws.close(code=1013);return
        async with busy:
            ident=uuid.uuid4().hex
            log=directory/(ident+'.jsonl')
            session=None;total_model_calls=0
            stream=None
            send_lock=asyncio.Lock()
            async def send(payload):
                async with send_lock:
                    with log.open('a') as file:
                        file.write(json.dumps(payload,ensure_ascii=False)+'\n')
                    await asyncio.wait_for(ws.send_json(payload),timeout=30)
            try:
                if config is None:
                    config=await asyncio.wait_for(ws.receive_json(),timeout=15)
                    if not isinstance(config,dict) or config.get('event')!='start':
                        raise ValueError('需要start消息。')
                interval=validate(config.get('language'),config.get('domain','trade'),config.get('interval_ms',500))
                language=config['language'];domain=config.get('domain','trade')
                session=app.state.new_text_refinement_session(domain)
                translator=TranslationSnapshots(translate)
                await send({'event':'ready','run_id':ident,'language':language,'target_language':'zh'})
                if snapshots is None:
                    text=config.get('text','')
                    if not isinstance(text,str) or not text.strip() or len(text)>12000:
                        raise ValueError('请输入1到12000字符的外语原文。')
                    chunk_chars=config.get('chunk_chars',40)
                    if isinstance(chunk_chars,bool) or not isinstance(chunk_chars,int) or not 4<=chunk_chars<=500:
                        raise ValueError('文本分块长度需在4到500之间。')
                    async def texts():
                        tick=time.monotonic()
                        for index,end in enumerate(range(chunk_chars,len(text)+chunk_chars,chunk_chars)):
                            if interval:
                                await asyncio.sleep(max(0,tick+index*interval-time.monotonic()))
                            end=min(end,len(text))
                            yield text[:end],end==len(text),{}
                    stream=texts()
                else:
                    stream=snapshots(config)
                last_refined_prefix=None
                last_result={}

                async def translate_snapshot(snapshot):
                    translated,elapsed=await thread_call(translator.update,snapshot.source,language)
                    return {'translation':translated,'translation_ms':elapsed,
                            'stable_translation':translator.stable_translation,
                            'translation_stats':dict(translator.last_stats)}

                async def refine_snapshot(snapshot,payload):
                    nonlocal last_refined_prefix,last_result,total_model_calls
                    translated=payload['translation']
                    stable=translated if snapshot.final else payload['stable_translation']
                    before=len(session.model_calls);start=time.perf_counter()
                    executed=bool(snapshot.final or (stable and stable!=last_refined_prefix))
                    if executed:
                        last_result=await thread_call(session.update,stable,
                            revision=snapshot.revision,final=snapshot.final)
                        last_refined_prefix=stable
                    calls=sum(len(t.get('model_calls',[])) for t in session.model_calls[before:])
                    total_model_calls+=calls
                    display=session.display.compose(translated)
                    result={**last_result,**display,'raw_text':translated,
                            'clean_text':display['display_text']}
                    if not executed:
                        result.update(refiner_executed=False,refiner_accepted=None,
                            refiner_reject_reasons=[],refinement_state='waiting_for_group')
                    return {**result,'event':'final' if snapshot.final else 'update',
                            'revision':snapshot.revision,'source_text':snapshot.source,
                            'translation':translated,'model_calls':calls,
                            'total_model_calls':total_model_calls,
                            'translation_ms':payload['translation_ms'],
                            'translation_stats':payload['translation_stats'],
                            'refinement_ms':(time.perf_counter()-start)*1000,
                            **snapshot.progress}

                async def watch_disconnect():
                    while True:
                        message=await ws.receive()
                        if message['type']=='websocket.disconnect':
                            return

                pipeline=asyncio.create_task(run_snapshot_pipeline(
                    stream,translate_snapshot,refine_snapshot,send))
                disconnect=asyncio.create_task(watch_disconnect())
                try:
                    done,_=await asyncio.wait((pipeline,disconnect),return_when=asyncio.FIRST_COMPLETED)
                    if pipeline in done:
                        await pipeline
                    else:
                        raise WebSocketDisconnect()
                finally:
                    for task in (pipeline,disconnect):
                        if not task.done():task.cancel()
                    await asyncio.gather(pipeline,disconnect,return_exceptions=True)
                (directory/(ident+'.refiner.json')).write_text(json.dumps(session.model_calls,ensure_ascii=False,indent=2)+'\n')
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                try:await send({'event':'error','detail':str(exc)[:500]})
                except (WebSocketDisconnect,RuntimeError,TimeoutError):pass
            finally:
                if stream is not None:
                    await stream.aclose()
                if session:
                    (directory/(ident+'.refiner.json')).write_text(json.dumps(session.model_calls,ensure_ascii=False,indent=2)+'\n')
                try:await ws.close()
                except RuntimeError:pass

    @app.websocket('/ws/translation')
    async def text_websocket(ws: WebSocket):
        await run(ws,None,None)

    @app.websocket('/ws/translation/audio/{ident}')
    async def audio_websocket(ws: WebSocket, ident: str):
        prune_uploads();job=jobs.pop(ident,None)
        if job is None:
            await ws.accept();await ws.send_json({'event':'error','detail':'音频任务不存在或已过期，请重新上传。'});await ws.close();return
        config={'language':job['language'],'domain':job['domain'],'interval_ms':job['interval']*1000}
        async def audio_updates(config):
            session_id=None
            try:
                samples=await thread_call(decode_audio,job['path'])
                initial=await thread_call(_stream_request,asr_url,'/stream/start',
                    params={'language':LANGUAGES[config['language']][0],'chunk_seconds':str(job['chunk_seconds'])})
                session_id=initial['session_id']
                width=round(job['chunk_seconds']*16000)
                tick=time.monotonic()
                for index,offset in enumerate(range(0,len(samples),width)):
                    if job['interval']:
                        await asyncio.sleep(max(0,tick+index*job['interval']-time.monotonic()))
                    block=samples[offset:offset+width]
                    started=time.perf_counter()
                    result=await thread_call(_stream_request,asr_url,'/stream/chunk',session_id,block.tobytes())
                    yield str(result.get('text','')),False,{'audio_processed_seconds':min(offset+width,len(samples))/16000,
                        'audio_total_seconds':len(samples)/16000,'asr_ms':(time.perf_counter()-started)*1000}
                started=time.perf_counter()
                result=await thread_call(_stream_request,asr_url,'/stream/finish',session_id)
                session_id=None
                yield str(result.get('text','')),True,{'audio_processed_seconds':len(samples)/16000,
                    'audio_total_seconds':len(samples)/16000,'asr_ms':(time.perf_counter()-started)*1000}
            finally:
                if session_id:
                    try:await thread_call(_stream_request,asr_url,'/stream/cancel',session_id)
                    except Exception:pass
        try:
            await run(ws,config,audio_updates)
        finally:
            job['path'].unlink(missing_ok=True)
