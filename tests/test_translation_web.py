"""Focused, backend-free tests for the optional translation web routes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# The requested invocation runs from the sibling Translator checkout, so make
# the Codex-web package importable without relying on the caller's PYTHONPATH.
WEB_ROOT = Path(__file__).resolve().parents[1]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from system import translation_web


def test_source_pieces_preserves_all_characters_and_caps_piece_size():
    source = ("A sentence with decimals 3.14 and punctuation!\n"
              "中文句子。" + "x" * 705 + "?end")

    pieces = translation_web.source_pieces(source)

    assert "".join(pieces) == source
    assert pieces
    assert all(len(piece) <= 700 for piece in pieces)


def test_translation_snapshots_reuses_only_current_unchanged_pieces():
    calls = []

    def translate(piece, language):
        calls.append((piece, language))
        return f"译:{piece}"

    snapshots = translation_web.TranslationSnapshots(translate)
    first_source = "Alpha. Bravo."
    first_translation, _ = snapshots.update(first_source, "en")
    assert first_translation == "译:Alpha. Bravo."

    revised_source = "Alpha. Charlie."
    revised_translation, _ = snapshots.update(revised_source, "en")
    assert revised_translation == "译:Alpha. Charlie."
    assert calls == [(first_source, "en"), (revised_source, "en")]
    assert "Bravo" not in revised_translation
    snapshots.update(revised_source,"en")
    assert len(calls)==2


class _FakeSession:
    def __init__(self):
        from system.window_refinement import StreamingRefinementDisplay
        self.display=StreamingRefinementDisplay()
        self.model_calls = []
        self.updates = []

    def update(self, text, *, revision, final=False):
        self.updates.append((text, revision, final))
        result={"raw_text":text,"clean_text":text,"final":final}
        self.display.accept(result,revision)
        return result


def _make_app(tmp_path, translate=None):
    app = FastAPI()
    sessions = []

    def new_session(domain):
        session = _FakeSession()
        sessions.append((domain, session))
        return session

    app.state.new_text_refinement_session = new_session
    calls = []

    def fake_translate(piece, language):
        calls.append((piece, language))
        return f"中文:{piece}"

    translation_web.install_translation_routes(
        app, translate or fake_translate, asr_url="http://asr.invalid",
        output_dir=tmp_path,
    )
    return app, calls, sessions


def test_text_websocket_sends_source_translation_update_and_final(tmp_path):
    app, translate_calls, sessions = _make_app(tmp_path)
    source = "A good sentence."

    with TestClient(app).websocket_connect("/ws/translation") as ws:
        ws.send_json({"event": "start", "language": "en", "text": source,
                      "chunk_chars": 40, "interval_ms": 0})
        events = []
        while True:
            payload = ws.receive_json()
            events.append(payload)
            if payload["event"] in {"error", "final"}:
                break

    names = [event["event"] for event in events]
    assert names == ["ready", "source", "translation", "final"]
    source_event, translation_event, final_event = events[1:]
    assert source_event["source_text"] == source
    assert translation_event["translation"] == f"中文:{source}"
    assert final_event["source_text"] == source
    assert final_event["translation"] == f"中文:{source}"
    assert final_event["revision"] == 1
    assert final_event["total_model_calls"] == 0
    assert final_event["model_calls"] == 0
    assert translate_calls == [(source, "en")]
    assert sessions[0][0] == "trade"
    assert sessions[0][1].updates == [(f"中文:{source}", 1, True)]


def test_audio_asr_reaches_final_while_translation_is_blocked(tmp_path,monkeypatch):
    import threading
    import numpy as np
    from system import web_app
    started=threading.Event();release=threading.Event();finished=threading.Event()
    asr_calls=[]
    def translate(source,language):
        started.set()
        assert release.wait(5),'test did not release model'
        finished.set()
        return '中文:'+source
    def asr(url,endpoint,session_id=None,data=b'',params=None):
        asr_calls.append(endpoint)
        if endpoint=='/stream/start':return {'session_id':'test-asr'}
        if endpoint=='/stream/chunk':
            count=asr_calls.count(endpoint)
            if count==2:
                assert started.wait(5),'translation did not start'
            return {'text':'One. Two. Three.'[:[4,9,16][count-1]]}
        if endpoint=='/stream/finish':return {'text':'One. Two. Three.'}
        return {'cancelled':True}
    monkeypatch.setattr(translation_web,'decode_audio',lambda path:np.zeros(96000,dtype='<f4'))
    monkeypatch.setattr(web_app,'_stream_request',asr)
    app,_,sessions=_make_app(tmp_path,translate)
    with TestClient(app) as client:
        uploaded=client.post('/api/translation/audio',files={'file':('test.wav',b'audio')},
            data={'language':'en','interval_ms':'0','chunk_seconds':'2'})
        assert uploaded.status_code==200
        path=uploaded.json()['websocket']
        with client.websocket_connect(path) as ws:
            events=[]
            try:
                while True:
                    event=ws.receive_json();events.append(event)
                    assert event['event']!='error',event
                    if event['event']=='source' and event['revision']==4:
                        break
                assert asr_calls.count('/stream/chunk')==3
                assert '/stream/finish' in asr_calls
                assert not finished.is_set(),'ASR waited for translation'
            finally:
                release.set()
            while events[-1]['event']!='final':
                event=ws.receive_json();events.append(event)
                assert event['event']!='error',event
        assert events[-1]['source_text']=='One. Two. Three.'
        assert events[-1]['pipeline_stats']['max_pending_per_stage']==1
        assert events[-1]['pipeline_stats']['merged_source_snapshots']>0
        assert sum(e['event']=='final' for e in events)==1
        assert events[-1]['revision']==4
    assert not list((tmp_path/'uploads').iterdir())


@pytest.mark.parametrize(
    "start, expected_detail",
    [
        ({"language": "xx", "text": "hello"}, "支持的源语言"),
        ({"language": "en", "interval_ms": -1, "text": "hello"}, "回放间隔"),
        ({"language": "en", "chunk_chars": 3, "text": "hello"}, "分块长度"),
        ({"language": "en", "text": "x" * 12001}, "12000字符"),
    ],
)
def test_invalid_text_websocket_requests_return_error_without_translation(
    tmp_path, start, expected_detail
):
    app, translate_calls, sessions = _make_app(tmp_path)
    payload = {"event": "start", "domain": "trade", "interval_ms": 0,
               "chunk_chars": 40, **start}

    with TestClient(app).websocket_connect("/ws/translation") as ws:
        ws.send_json(payload)
        received = []
        while not received or received[-1]["event"] != "error":
            received.append(ws.receive_json())

    assert expected_detail in received[-1]["detail"]
    assert translate_calls == []
    assert all(not session.updates for _, session in sessions)


def test_upload_requires_file_and_rejects_empty_file(tmp_path):
    app, _, _ = _make_app(tmp_path)
    client = TestClient(app)

    missing = client.post("/api/translation/audio")
    empty = client.post("/api/translation/audio", files={"file": ("empty.wav", b"")})

    assert missing.status_code == 422
    assert empty.status_code == 400
    assert "为空" in empty.json()["detail"]


def test_audio_websocket_unknown_id_returns_error_without_backend(tmp_path):
    app, _, _ = _make_app(tmp_path)

    with TestClient(app).websocket_connect("/ws/translation/audio/missing") as ws:
        message = ws.receive_json()

    assert message["event"] == "error"
    assert "不存在或已过期" in message["detail"]
