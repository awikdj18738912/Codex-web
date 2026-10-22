"""Exercise the Web switch at the ASR/audio-repair publication boundary."""
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import system.entity_ctc_sidecar as ctc_module
import system.web_app as web_app
from system.entity_store import EntityStore


RAW = '鬼灵们是这个副本的入口。'
FIXED = '鬼灵门是这个副本的入口。'


class IdentityRefiner:
    def __init__(self,*args,**kwargs):
        pass

    def refine(self,text,*,entity_hints=(),strict_placeholders=False):
        return text,1.


class FakeCtcSession:
    def __init__(self,definitions):
        self.definition=definitions[0]
        self.received=0
        self.event_count=0
        self.compute_seconds=0.
        self.error=None

    @property
    def audio_seconds(self):return self.received/64000

    def submit_audio(self,pcm):
        self.received+=len(pcm)
        self.event_count+=1

    async def finish(self):
        pass

    def propose(self,raw_text,*,final=False):
        if not final or '鬼灵们' not in raw_text:
            return []
        start=raw_text.index('鬼灵们')
        return [dict(start=start,end=start+3,observed='鬼灵们',
                     canonical='鬼灵门',entity_id=self.definition.entity_id,
                     retrieval_score=.95,evidence_surface='鬼灵门')]

    def close(self):
        pass


class FakeCtcRuntime:
    def __init__(self,model_dir):
        self.model_dir=Path(model_dir)

    def create_session(self,definitions):
        return FakeCtcSession(definitions)


class FakeAsr:
    def __init__(self):
        self.count=0
        self.replay={}

    def __call__(self,asr_url,endpoint,session_id=None,data=b'',params=None):
        if endpoint=='/stream/start':
            self.count+=1
            sid=str(self.count)
            self.replay[sid]=bool((params or {}).get('context'))
            return dict(session_id=sid,context_applied=self.replay[sid])
        if endpoint=='/stream/chunk':
            return dict(text=RAW,completed_segments=[dict(segment_id=1,text=RAW)])
        if endpoint=='/stream/finish':
            return dict(text=RAW,completed_segments=[dict(segment_id=1,text=RAW)])
        if endpoint=='/stream/cancel':
            return dict(cancelled=True)
        raise AssertionError(endpoint)


def receive_event(websocket,wanted):
    for _ in range(40):
        event=websocket.receive_json()
        if event.get('event')==wanted:
            return event
    raise AssertionError('No '+wanted+' event')


def test_ctc_shadow_and_on_are_session_scoped_and_visible():
    with tempfile.TemporaryDirectory() as directory:
        database=Path(directory)/'entities.db'
        EntityStore(database).upsert_entity('鬼灵门',normalization_policy='normalize')
        backend=FakeAsr()
        with (patch.object(web_app,'TransformersRefiner',IdentityRefiner),
              patch.object(web_app,'_stream_request',backend),
              patch.object(ctc_module,'EntityCtcRuntime',FakeCtcRuntime)):
            app=web_app.create_app(Path('/tmp/fake-refiner'),'cpu','http://fake-asr',
                                   'Chinese',32,None,database,'shadow',
                                   entity_audio_repair=True,entity_ctc_mode='shadow',
                                   entity_ctc_model_dir=Path(directory))
            with TestClient(app) as client:
                assert client.get('/health').json()['entity_ctc_mode']=='shadow'
                for mode,expected in (('shadow',RAW),('on',FIXED)):
                    with client.websocket_connect('/ws/stream?mode=streaming&entity_ctc='+mode) as websocket:
                        ready=receive_event(websocket,'ready')
                        assert ready['entity_ctc_mode']==mode
                        websocket.send_bytes(b'\0'*64000)
                        receive_event(websocket,'update')
                        websocket.send_json(dict(event='finish'))
                        final=receive_event(websocket,'final')
                        assert final['entity_ctc_mode']==mode
                        assert final['clean_text']==expected
                        assert final['entity_audio_repair']['incremental']['ctc']['mode']==mode


def test_ctc_capacity_falls_back_to_existing_audio_review():
    class FullCtcRuntime(FakeCtcRuntime):
        def create_session(self,definitions):
            return None

    with tempfile.TemporaryDirectory() as directory:
        database=Path(directory)/'entities.db'
        EntityStore(database).upsert_entity('鬼灵门',normalization_policy='normalize')
        with (patch.object(web_app,'TransformersRefiner',IdentityRefiner),
              patch.object(web_app,'_stream_request',FakeAsr()),
              patch.object(ctc_module,'EntityCtcRuntime',FullCtcRuntime)):
            app=web_app.create_app(Path('/tmp/fake-refiner'),'cpu','http://fake-asr',
                                   'Chinese',32,None,database,'shadow',
                                   entity_audio_repair=True,entity_ctc_mode='on',
                                   entity_ctc_model_dir=Path(directory))
            with TestClient(app) as client:
                with client.websocket_connect('/ws/stream?mode=streaming&entity_ctc=on') as websocket:
                    ready=receive_event(websocket,'ready')
                    assert ready['entity_ctc_mode']=='off'
                    websocket.send_bytes(b'\0'*64000)
                    receive_event(websocket,'update')
                    websocket.send_json(dict(event='finish'))
                    final=receive_event(websocket,'final')
                    assert final['entity_ctc_mode']=='off'
                    assert final['entity_audio_repair_requested'] is True

import pytest


@pytest.fixture(autouse=True)
def isolate_audio_review_from_direct_spelling(monkeypatch):
    """These cases exercise acoustic review; direct spelling has its own suite."""
    from system.entity_streaming_live import LiveEntityMixin
    async def no_direct_spelling(self, raw_text, *, final=False):
        return None
    monkeypatch.setattr(LiveEntityMixin, "_live_apply_glossary", no_direct_spelling)
