"""Audio verification must never delete genuine spoken glossary content."""
import asyncio
from system.entity_asr_context import EntityASRContext
from system.entity_asr_echo_guard import EntityASREchoGuard

TERMS=('ICU','MRSA','MRSA感染')
PROMPT='参考术语仅用于辅助拼写。术语列表：'+str(TERMS)


class Backend:
    def __init__(self,off_text,*,applied=True):
        self.off_text=off_text;self.applied=applied;self.sessions={};self.calls=[];self.counter=0
    async def __call__(self,endpoint,session_id=None,data=b'',params=None):
        self.calls.append((endpoint,session_id,data,params))
        if endpoint=='/stream/start':
            self.counter+=1;sid=str(self.counter);context=(params or {}).get('context','')
            self.sessions[sid]=dict(context=context,data=b'')
            return dict(session_id=sid,context_applied=bool(context) and self.applied)
        if endpoint=='/stream/cancel':self.sessions.pop(session_id,None);return dict(cancelled=True)
        s=self.sessions[session_id]
        if endpoint=='/stream/chunk':s['data']+=data
        text=PROMPT+'患者安心治病。' if s['context'] else self.off_text
        segment=dict(segment_id=1,text=text,vad_boundary=True)
        result=dict(text=text,context_applied=bool(s['context']),completed_segments=[segment])
        if endpoint=='/stream/finish':self.sessions.pop(session_id,None)
        return result


def run(coro):return asyncio.run(coro)


def test_echo_replays_same_pcm_without_context_and_continues_clean_session():
    async def scenario():
        backend=Backend('患者安心治病。');guard=EntityASREchoGuard(backend,EntityASRContext(PROMPT,TERMS),replay_chunk_bytes=4)
        start=await guard.request('/stream/start',params={'context':PROMPT,'language':'Chinese','chunk_seconds':2})
        original_id=start['session_id'];pcm=b'12345678'
        result=await guard.request('/stream/chunk',original_id,pcm)
        assert result['text']=='患者安心治病。'
        assert result['entity_asr_context_guard_revision']==1
        assert guard.fallback_active and not guard.active and guard._audio is None
        active=guard._sid
        assert backend.sessions[active]['data']==pcm
        assert backend.sessions[active]['context']==''
        assert original_id not in backend.sessions
        await guard.request('/stream/chunk',original_id,b'9999')
        assert backend.sessions[active]['data']==pcm+b'9999'
        final=await guard.request('/stream/finish',original_id)
        assert final['context_applied'] is False and not backend.sessions
    run(scenario())


def test_real_spoken_instruction_is_kept_when_no_context_audio_supports_it():
    async def scenario():
        backend=Backend(PROMPT+'患者安心治病。');guard=EntityASREchoGuard(backend,EntityASRContext(PROMPT,TERMS))
        start=await guard.request('/stream/start',params={'context':PROMPT})
        result=await guard.request('/stream/chunk',start['session_id'],b'1234')
        assert result['text']==PROMPT+'患者安心治病。'
        assert not guard.fallback_active and guard.revision==0
        assert guard.events[-1]['action']=='keep_audio_supported_text'
        await guard.request('/stream/cancel',start['session_id'])
        assert not backend.sessions and guard._audio is None
    run(scenario())


def test_ordered_word_list_echo_is_detected_without_instruction_prefix():
    async def scenario():
        prompt='ICU, MRSA, MRSA感染';backend=Backend('医生说可以出院。')
        guard=EntityASREchoGuard(backend,EntityASRContext(prompt,TERMS))
        start=await guard.request('/stream/start',params={'context':prompt})
        # Replace the backend's context-bearing output with a copied list.
        original=backend.__call__
        async def response(endpoint,session_id=None,data=b'',params=None):
            result=await original(endpoint,session_id,data,params)
            if endpoint=='/stream/chunk' and result.get('context_applied'):
                result['text']=prompt;result['completed_segments'][0]['text']=prompt
            return result
        guard._request=response
        result=await guard.request('/stream/chunk',start['session_id'],b'1234')
        assert result['text']=='医生说可以出院。' and guard.fallback_active
        await guard.request('/stream/cancel',start['session_id'])
    run(scenario())


def test_unsupported_context_never_buffers_or_starts_extra_session():
    async def scenario():
        backend=Backend('正常文字。',applied=False);guard=EntityASREchoGuard(backend,EntityASRContext(PROMPT,TERMS))
        start=await guard.request('/stream/start',params={'context':PROMPT})
        await guard.request('/stream/chunk',start['session_id'],b'1234')
        assert guard._audio is None and backend.counter==1
        await guard.request('/stream/cancel',start['session_id'])
    run(scenario())


def test_final_only_echo_returns_all_replayed_completed_segments_and_closes_cache():
    async def scenario():
        backend=Backend('复核后的真实文字。');guard=EntityASREchoGuard(backend,EntityASRContext(PROMPT,TERMS),replay_chunk_bytes=4)
        start=await guard.request('/stream/start',params={'context':PROMPT})
        guard._audio.write(b'12345678');guard._audio_size=8
        final=await guard.request('/stream/finish',start['session_id'])
        assert final['text']=='复核后的真实文字。' and final['completed_segments']
        assert guard._audio is None and not backend.sessions
    run(scenario())


def test_same_sound_spoken_names_are_not_rejected_for_asr_character_variants():
    from system.entity_asr_echo_guard import sound_key
    assert sound_key('郭健文李斯达吕慧慧')==sound_key('郭建文李思达吕惠惠')


def test_web_fallback_rebuilds_vad_sources_and_reports_effective_context(tmp_path,monkeypatch):
    from pathlib import Path
    from fastapi.testclient import TestClient
    import system.web_app as web
    from system.entity_store import EntityStore
    class Identity:
        def __init__(self,*args,**kwargs):pass
        def refine(self,text,**kwargs):return text,0.0
        def refine_with_numeric_rules(self,text,**kwargs):return text,0.0
    backend=Backend('患者安心治病。')
    async def request(url,endpoint,session_id=None,data=b'',params=None,**kwargs):
        return await backend(endpoint,session_id,data,params)
    monkeypatch.setattr(web,'TransformersRefiner',Identity)
    monkeypatch.setattr(web,'_stream_request_async',request)
    monkeypatch.setattr(web,'build_entity_asr_context',lambda *args:EntityASRContext(PROMPT,TERMS))
    db=tmp_path/'entities.db';EntityStore(db).upsert_entity('ICU',domain='medical',normalization_policy='normalize')
    app=web.create_app(Path(tmp_path),'cpu','http://unused','Chinese',32,None,
        entity_db=db,entity_fuzzy_mode='auto',entity_asr_context=True,refinement_gate_mode='tri_state')
    with TestClient(app) as client:
        with client.websocket_connect('/ws/stream?mode=streaming&domain=medical&entity_asr_context=1') as socket:
            assert socket.receive_json()['entity_asr_context_applied'] is True
            socket.send_bytes(b'1234')
            while socket.receive_json().get('event')!='chunk_ack':pass
            socket.send_json({'event':'finish'})
            while True:
                result=socket.receive_json()
                if result.get('event')=='error':raise AssertionError(result)
                if result.get('event')=='final':break
            assert '参考术语' not in result['raw_text'] and '参考术语' not in result['clean_text']
            assert result['clean_text']=='患者安心治病。'
            assert result['entity_asr_context_applied'] is False
            assert result['entity_asr_context_guard']['fallback_active'] is True
    assert not backend.sessions
