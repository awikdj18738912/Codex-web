"""Contract tests for bounded, final-only entity confirmation on short audio."""
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import system.entity_streaming_live as live
from system.entity_audio_repair import EntityAudioRepair
from system.entity_store import EntityDefinition
from system.entity_store import EntityStore


RAW = "今天我们讨论厉飞羽。"
FIXED = RAW.replace("厉飞羽", "厉飞宇")
STREAM_RAW = "今天我们讨论厉飞羽的相关问题。"
STREAM_FIXED = STREAM_RAW.replace("厉飞羽", "厉飞宇")
DEF = EntityDefinition(1, "厉飞宇", "TERM", "general", "normalize", 0, ())
PCM = b"\x17\x29" * 32000  # one second at the controller's 64 kB/s accounting rate


def candidate(text, *, observed="厉飞羽", canonical="厉飞宇", score=.92):
    start = text.index(observed)
    return dict(start=start, end=start + len(observed), observed=observed,
                canonical=canonical, entity_id=1, evidence_surface=canonical,
                retrieval_score=score)


class Backend:
    """Primary hypotheses and finish replay are independently controllable."""
    def __init__(self, final=RAW, *, primary_chunk="", replay_final=FIXED,
                 primary_segments=None, replay_chunk="", delay_replay_finish=0., replay_error=False):
        self.final, self.primary_chunk = final, primary_chunk
        self.primary_segments = primary_segments or []
        self.replay_final, self.replay_chunk = replay_final, replay_chunk
        self.delay_replay_finish, self.replay_error = delay_replay_finish, replay_error
        self.calls, self.sessions, self.started, self.n = [], {}, [], 0

    async def __call__(self, endpoint, sid=None, data=b"", params=None):
        params = dict(params or {})
        self.calls.append((endpoint, sid, bytes(data), params))
        if endpoint == "/stream/start":
            self.n += 1
            sid = str(self.n)
            replay = bool(params.get("context"))
            self.sessions[sid] = {"replay": replay, "pcm": bytearray()}
            self.started.append((sid, replay))
            return {"session_id": sid, "context_applied": replay}
        if endpoint == "/stream/cancel":
            self.sessions.pop(sid, None)
            return {"cancelled": True}
        state = self.sessions[sid]
        if endpoint == "/stream/chunk":
            state["pcm"].extend(data)
            text = self.replay_chunk if state["replay"] else self.primary_chunk
            segments = [] if state["replay"] else self.primary_segments
            return {"text": text, "completed_segments": segments}
        if endpoint == "/stream/finish":
            if state["replay"]:
                if self.delay_replay_finish:
                    await asyncio.sleep(self.delay_replay_finish)
                if self.replay_error:
                    raise RuntimeError("synthetic replay failure")
                return {"text": self.replay_final, "completed_segments": []}
            return {"text": self.final, "completed_segments": [dict(segment_id=1, text=self.final)]}
        raise AssertionError(endpoint)


class Selector:
    def __init__(self, entries):
        self.entries = entries

    def select(self, text):
        found = [candidate(text, **entry) for entry in self.entries
                 if entry["observed"] in text]
        return found, []


def install_selector(monkeypatch, entries=None):
    entries = entries or [dict(observed="厉飞羽", canonical="厉飞宇", score=.92)]
    monkeypatch.setattr(live, "_selector", lambda definitions: Selector(entries))


def make_repair(backend, **kwargs):
    options = dict(request_timeout=.5, short_finish_max_seconds=16., short_finish_wait=4.)
    options.update(kwargs)
    repair = EntityAudioRepair(backend, [DEF], **options)
    repair._live_primary_chunk_bytes = 10**9
    return repair


async def finish_once(repair, backend, audio=PCM, *, chunk_text=""):
    sid = (await repair.request("/stream/start", params={"chunk_seconds": 1}))['session_id']
    if audio:
        await repair.request("/stream/chunk", sid, audio)
    return await repair.request("/stream/finish", sid), sid


def test_final_only_tail_name_is_confirmed_and_publication_is_consistent(monkeypatch):
    async def scenario():
        install_selector(monkeypatch)
        backend = Backend(final=RAW)
        repair = make_repair(backend)
        final, _ = await finish_once(repair, backend)
        assert final["text"] == FIXED
        assert final["completed_segments"] == [dict(segment_id=1, text=FIXED,
                                                     confidence=None, confidence_metadata={})]
        assert final["entity_audio_repair_revision"] == repair.revision == 1
        snap = repair.snapshot()["incremental"]
        assert snap["short_finish_checks"] == 1
        assert snap["short_finish_changes"] == 1
        assert repair.replay_mode == "short_finish"
    asyncio.run(scenario())


def test_one_nonempty_decode_with_empty_stable_prefix_can_be_rechecked_at_finish(monkeypatch):
    async def scenario():
        install_selector(monkeypatch)
        backend = Backend(final=RAW, primary_chunk="")
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start"))["session_id"]
        await repair.request("/stream/chunk", sid, PCM)
        final = await repair.request("/stream/finish", sid)
        assert final["text"] == FIXED
        assert repair.snapshot()["incremental"]["short_finish_changes"] == 1
    asyncio.run(scenario())


def test_finish_replay_receives_entire_original_pcm_and_waits_for_replay_final(monkeypatch):
    async def scenario():
        install_selector(monkeypatch)
        backend = Backend(final=RAW, replay_chunk="")
        repair = make_repair(backend)
        final, _ = await finish_once(repair, backend, PCM)
        replay_sid = next(sid for ep, sid, _, _ in backend.calls
                          if ep == "/stream/chunk" and sid != "1")
        submitted = b"".join(data for ep, sid, data, _ in backend.calls
                              if ep == "/stream/chunk" and sid == replay_sid)
        assert submitted == PCM
        assert final["text"] == FIXED
        assert repair.snapshot()["incremental"]["short_finish_checks"] == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("duration,enabled", [(17, True), (1, False)])
def test_long_audio_or_disabled_short_finish_does_not_start_finish_review(monkeypatch, duration, enabled):
    async def scenario():
        install_selector(monkeypatch)
        backend = Backend(final=RAW)
        options = {"short_finish_wait": 0.} if not enabled else {}
        repair = make_repair(backend, **options)
        audio = PCM * duration
        final, _ = await finish_once(repair, backend, audio)
        assert final["text"] == RAW
        assert not any(ep == "/stream/start" and p.get("context") for ep, _, _, p in backend.calls)
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["timeout", "failure", "echo"])
def test_finish_timeout_failure_or_prompt_echo_preserves_raw_and_releases_resources(monkeypatch, kind):
    async def scenario():
        install_selector(monkeypatch)
        backend = Backend(final=RAW,
                          replay_final="参考术语仅用于辅助拼写" if kind == "echo" else FIXED,
                          delay_replay_finish=.1 if kind == "timeout" else 0.,
                          replay_error=kind == "failure")
        repair = make_repair(backend, short_finish_wait=.02, request_timeout=.5)
        final, _ = await finish_once(repair, backend)
        while repair._cleanup_tasks:
            await asyncio.gather(*tuple(repair._cleanup_tasks), return_exceptions=True)
        snap = repair.snapshot()["incremental"]
        assert final["text"] == RAW
        assert snap["short_finish_timed_out"] == (1 if kind == "timeout" else 0)
        assert snap["short_finish_errors"] == (1 if kind == "failure" else 0)
        assert repair._live_reserved_bytes == 0
        assert not repair._cleanup_tasks
        if kind == "timeout":
            replay_sid = next(sid for sid, replay in backend.started if replay)
            assert any(ep == "/stream/cancel" and sid == replay_sid
                       for ep, sid, _, _ in backend.calls)
    asyncio.run(scenario())


def test_ambiguous_same_span_candidates_are_rejected_without_review(monkeypatch):
    async def scenario():
        install_selector(monkeypatch, [dict(observed="厉飞羽", canonical="厉飞宇", score=.90),
                                       dict(observed="厉飞羽", canonical="厉飞雨", score=.89)])
        backend = Backend(final=RAW)
        repair = make_repair(backend)
        final, _ = await finish_once(repair, backend)
        assert final["text"] == RAW
        assert not any(ep == "/stream/start" and p.get("context") for ep, _, _, p in backend.calls)
    asyncio.run(scenario())


def test_stream_confirmed_patch_survives_finish_without_second_review(monkeypatch):
    async def scenario():
        install_selector(monkeypatch)
        backend = Backend(final=STREAM_RAW, primary_chunk=STREAM_RAW, replay_final=STREAM_FIXED,
                          primary_segments=[dict(segment_id=1, text=STREAM_RAW)],
                          replay_chunk=STREAM_FIXED)
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start"))["session_id"]
        await repair.request("/stream/chunk", sid, PCM)
        if repair._live_worker:
            await asyncio.gather(repair._live_worker, return_exceptions=True)
        before = sum(bool(ep == "/stream/start" and p.get("context")) for ep, _, _, p in backend.calls)
        final = await repair.request("/stream/finish", sid)
        after = sum(bool(ep == "/stream/start" and p.get("context")) for ep, _, _, p in backend.calls)
        assert final["text"] == STREAM_FIXED
        assert after == before
        assert repair.snapshot()["incremental"]["short_finish_checks"] == 0
    asyncio.run(scenario())


def test_websocket_finish_only_entity_repair_reaches_final_output():
    """Exercise the real WebSocket route and public controller together."""
    from fastapi.testclient import TestClient
    import system.web_app as web_app

    class IdentityRefiner:
        def __init__(self, *args, **kwargs):
            pass

        def refine(self, text, *, entity_hints=(), strict_placeholders=False):
            return text, 1.0

    class AsrBackend:
        def __init__(self):
            self.sessions = {}
            self.calls = []
            self.counter = 0

        def __call__(self, asr_url, endpoint, session_id=None, data=b"", params=None):
            params = dict(params or {})
            payload = bytes(data)
            self.calls.append((endpoint, session_id, payload, params))
            if endpoint == "/stream/start":
                self.counter += 1
                sid = str(self.counter)
                replay = bool(params.get("context"))
                self.sessions[sid] = {"replay": replay, "pcm": bytearray()}
                return {"session_id": sid, "context_applied": replay}
            state = self.sessions[session_id]
            if endpoint == "/stream/chunk":
                state["pcm"].extend(payload)
                # No replay chunk can confirm the candidate. Confirmation is
                # deliberately available only after replay finish.
                return {"text": "", "completed_segments": []}
            if endpoint == "/stream/finish":
                if state["replay"]:
                    return {"text": FIXED, "completed_segments": []}
                return {"text": RAW, "completed_segments":[dict(segment_id=1, text=RAW)]}
            if endpoint == "/stream/cancel":
                self.sessions.pop(session_id, None)
                return {"cancelled": True}
            raise AssertionError(endpoint)

    pcm = b"\x21\x43" * 32000
    backend = AsrBackend()
    with tempfile.TemporaryDirectory() as temp:
        entity_db = Path(temp) / "entities.db"
        EntityStore(entity_db).upsert_entity("厉飞宇", entity_type="PERSON", domain="general",
                                             normalization_policy="normalize")
        with patch.object(web_app, "TransformersRefiner", IdentityRefiner), \
             patch.object(web_app, "_stream_request", backend):
            app = web_app.create_app(Path(temp) / "refiner", "cpu", "http://fake-asr",
                                     "Chinese", 32, None, entity_db=entity_db,
                                     entity_asr_context=False, entity_audio_repair=True)
            with TestClient(app) as client:
                with client.websocket_connect("/ws/stream?mode=streaming&entity_asr_context=0") as ws:
                    assert ws.receive_json()["event"] == "ready"
                    ws.send_bytes(pcm)
                    ack = ws.receive_json()
                    assert ack["event"] == "chunk_ack"
                    ws.send_json({"event": "finish"})
                    while True:
                        message = ws.receive_json()
                        if message.get("event") == "final":
                            final = message
                            break
    assert final["raw_text"] == FIXED, final["entity_audio_repair"].get("incremental")
    assert final["clean_text"] == FIXED
    repair = final["entity_audio_repair"]
    assert repair["incremental"]["short_finish_checks"] == 1
    assert any(item.get("stage") == "asr_audio_repair" for item in final["entity_normalizations"])
    replay_sid = next(sid for sid, state in backend.sessions.items() if state["replay"])
    replay_pcm = b"".join(data for endpoint, sid, data, _ in backend.calls
                           if endpoint == "/stream/chunk" and sid == replay_sid)
    assert replay_pcm == pcm
