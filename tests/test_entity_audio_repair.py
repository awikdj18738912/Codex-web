"""Generic entity audio verification and its real Web session wiring."""
import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from system.entity_audio_repair import EntityAudioRepair, GenericEntitySelector, verify_entity_audio_changes
from system.entity_store import EntityDefinition, EntityStore


def definition(name="罗剑", *, policy="normalize", kind="PERSON"):
    return EntityDefinition(1, name, kind, "medical", policy, 0, ())


class Backend:
    def __init__(self, original="嘉宾是罗建。", replay="嘉宾是罗剑。", *, supported=True, fail=False, slow=False):
        self.original, self.replay = original, replay
        self.supported, self.fail, self.slow = supported, fail, slow
        self.sessions, self.calls = {}, []
        self.counter = 0
        self.replay_pcm = b""
        self.replay_started = asyncio.Event()

    async def __call__(self, endpoint, session_id=None, data=b"", params=None):
        self.calls.append((endpoint, session_id, data, params))
        if endpoint == "/stream/start":
            self.counter += 1
            sid = str(self.counter)
            context = bool((params or {}).get("context"))
            self.sessions[sid] = dict(context=context, pcm=b"")
            return dict(session_id=sid, context_applied=context and self.supported)
        if endpoint == "/stream/cancel":
            self.sessions.pop(session_id, None)
            return dict(cancelled=True)
        session = self.sessions[session_id]
        if session["context"]:
            self.replay_started.set()
            if self.fail:
                raise RuntimeError("Replay unavailable")
            if self.slow:
                await asyncio.sleep(10)
        if endpoint == "/stream/chunk":
            session["pcm"] += data
            if session["context"]:
                self.replay_pcm = session["pcm"]
        text = self.replay if session["context"] else self.original
        result = dict(text=text, language="Chinese", confidence=0.99, confidence_covers_full_text=True,
                      completed_segments=[dict(segment_id=1, text=text, confidence=0.99)])
        if endpoint == "/stream/finish":
            self.sessions.pop(session_id)
        return result


async def finish(backend, guard, pcm=b"12345678", params=None):
    params = params if params is not None else dict(language="Chinese", chunk_seconds=2.0)
    ready = await guard.request("/stream/start", params=params)
    partial = await guard.request("/stream/chunk", ready["session_id"], pcm)
    final = await guard.request("/stream/finish", ready["session_id"])
    return ready, partial, final


def test_same_pcm_replayed_and_only_entity_updated_at_finish():
    async def scenario():
        backend = Backend(replay="嘉宾是罗剑。谢谢大家。")
        guard = EntityAudioRepair(backend, [definition()], replay_chunk_bytes=4)
        params = dict(language="Chinese", chunk_seconds=2.0, custom_flag="keep")
        ready, partial, result = await finish(backend, guard, params=params)
        assert ready["context_applied"] is False
        assert ready["entity_audio_repair"]["short_expansion_rule"] == "spelling_or_complete_edge_sound_v1"
        assert partial["text"] == backend.original
        assert result["text"] == "嘉宾是罗剑。"
        assert backend.replay_pcm == b"12345678"
        assert result["completed_segments"][0]["text"] == result["text"]
        assert result["confidence"] is None and result["confidence_covers_full_text"] is False
        assert result["entity_audio_repair_revision"] == 1
        assert result["entity_audio_repair"]["change_count"] == 1
        starts = [call[3] for call in backend.calls if call[0] == "/stream/start"]
        assert len(starts) == 2
        assert starts[0]["chunk_seconds"] == 2.0
        assert starts[1]["chunk_seconds"] == 10
        assert params == dict(language="Chinese", chunk_seconds=2.0, custom_flag="keep")
        assert guard._audio is None and not backend.sessions
    asyncio.run(scenario())


def test_adjacent_entities_can_be_repaired_without_changing_surrounding_text():
    async def scenario():
        backend = Backend("好，附加型，附加型有四种。", "好，附加刑，附加刑有四种。")
        guard = EntityAudioRepair(backend, [definition("附加刑", kind="TERM")])
        _, _, result = await finish(backend, guard)
        assert result["text"] == backend.replay
        assert result["entity_audio_repair"]["change_count"] == 2
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["unchanged", "unsupported", "failure", "cache_limit", "preserve"])
def test_failure_or_unconfirmed_candidate_keeps_first_pass_and_closes_resources(mode):
    async def scenario():
        backend = Backend(replay="嘉宾是罗建。" if mode == "unchanged" else "嘉宾是罗剑。",
                          supported=mode != "unsupported", fail=mode == "failure")
        guard = EntityAudioRepair(backend, [definition(policy="preserve" if mode == "preserve" else "normalize")],
                                  max_audio_bytes=4 if mode == "cache_limit" else 1024)
        _, _, result = await finish(backend, guard)
        assert result["text"] == backend.original
        assert result["confidence"] == 0.99
        assert not result["entity_audio_repair"]["applied"]
        assert not backend.sessions and guard._audio is None
        if mode in {"preserve", "cache_limit"}:
            assert backend.counter == 1
    asyncio.run(scenario())


def test_timeout_and_disconnect_cancel_replay_session_and_discard_audio():
    async def scenario(cancel):
        backend = Backend(slow=True)
        guard = EntityAudioRepair(backend, [definition()], verification_timeout=0.1 if not cancel else 30)
        start = await guard.request("/stream/start")
        await guard.request("/stream/chunk", start["session_id"], b"1234")
        task = asyncio.create_task(guard.request("/stream/finish", start["session_id"]))
        if cancel:
            await asyncio.wait_for(backend.replay_started.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result["text"] == backend.original
            assert result["entity_audio_repair"]["reason"] == "TimeoutError"
        assert not backend.sessions and guard._audio is None
    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


def test_candidate_selection_is_identical_for_every_entity_type():
    expected = GenericEntitySelector([definition()]).select("嘉宾是罗建。")
    assert expected[1] == ["罗剑"]
    for kind in ["TERM", "PERSON", "ORG", "PROJECT", "MODEL", "LOC"]:
        assert GenericEntitySelector([replace(definition(), entity_type=kind)]).select("嘉宾是罗建。") == expected


def test_short_sound_fragment_does_not_expand_to_an_unrelated_longer_glossary_word():
    candidate = dict(start=2, end=4, observed="乐曲", canonical="越秀区",
                     evidence_surface="越秀区", entity_id=1)
    source = "柏林乐曲博物馆。"
    assert verify_entity_audio_changes(source, "柏林越秀区博物馆。", [candidate]) == (source, [])


def test_short_fragment_with_spelling_evidence_keeps_an_audio_confirmed_suffix():
    candidate = dict(start=0, end=2, observed="李贤", canonical="礼贤镇",
                     evidence_surface="礼贤镇", entity_id=1)
    value, changes = verify_entity_audio_changes("李贤附近。", "礼贤镇附近。", [candidate])
    assert value == "礼贤镇附近。" and changes == [candidate]


def test_explicit_short_alias_can_normalize_to_a_longer_canonical_without_shared_characters():
    candidate = dict(start=2, end=4, observed="塔维", canonical="TAVI手术",
                     evidence_surface="塔维", entity_id=1)
    value, changes = verify_entity_audio_changes("进行塔维治疗。", "进行TAVI手术治疗。", [candidate])
    assert value == "进行TAVI手术治疗。" and changes == [candidate]


@pytest.mark.parametrize("observed", ["加喀", "喀尔"])
def test_short_complete_sound_prefix_or_suffix_can_restore_a_registered_name(observed):
    candidate = dict(start=2, end=4, observed=observed, canonical="贾卡尔",
                     evidence_surface="贾卡尔", entity_id=1)
    value, changes = verify_entity_audio_changes(f"开往{observed}。", "开往贾卡尔。", [candidate])
    assert value == "开往贾卡尔。" and changes == [candidate]


def test_weak_glossary_similarity_does_not_replay_ordinary_phrase():
    async def scenario():
        backend = Backend("还有什么检查？", "还有什么检查？")
        guard = EntityAudioRepair(backend, [definition("检查报告", kind="TERM")])
        _, _, result = await finish(backend, guard)
        status = result["entity_audio_repair"]
        assert result["text"] == backend.original
        assert backend.counter == 1 and backend.replay_pcm == b""
        assert status["candidate_count"] > 0
        assert status["max_candidate_score"] < status["minimum_replay_score"]
        assert status["reason"] == "weak_candidates"
        assert status["checks"] == 0 and status["prompt_terms"] == []
        assert not backend.sessions and guard._audio is None
    asyncio.run(scenario())


def test_useful_lower_scoring_entity_still_reaches_audio_verification():
    async def scenario():
        backend = Backend("那什么叫防面过大呢？", "那什么叫防卫过当呢？")
        guard = EntityAudioRepair(backend, [definition("防卫过当", kind="TERM")])
        _, _, result = await finish(backend, guard)
        assert backend.counter == 2
        assert result["text"] == backend.replay
        assert result["entity_audio_repair"]["change_count"] == 1
    asyncio.run(scenario())


def test_web_session_runs_generic_replay_and_exposes_entity_correction(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import system.web_app as web

    class Refiner:
        def refine(self, text, **kwargs):
            return text, 0.99

    monkeypatch.setattr(web, "TransformersRefiner", lambda *a, **k: Refiner())
    monkeypatch.setattr(web, "RepetitionPlausibility", lambda *a, **k: SimpleNamespace())
    backend = Backend()

    async def request(url, endpoint, session_id=None, data=b"", params=None, **kwargs):
        return await backend(endpoint, session_id, data, params)

    monkeypatch.setattr(web, "_stream_request_async", request)
    db = tmp_path / "entities.db"
    EntityStore(db).upsert_entity("罗剑", entity_type="PERSON", domain="medical", normalization_policy="normalize")
    app = web.create_app(Path("unused"), "cpu", "http://backend", "Chinese", 64, None,
                         entity_db=db, entity_fuzzy_mode="auto", entity_audio_repair=True)
    with TestClient(app) as client:
        assert client.get("/health").json()["entity_audio_repair_enabled"]
        with client.websocket_connect("/ws/stream?mode=offline&domain=medical&entity_audio_repair=1&entity_asr_context=0") as socket:
            ready = socket.receive_json()
            assert ready["entity_audio_repair"]["enabled"]
            assert not ready["entity_asr_context_applied"]
            socket.send_bytes(b"12345678")
            while socket.receive_json().get("event") != "chunk_ack":
                pass
            socket.send_json(dict(event="finish"))
            events = []
            while True:
                event = socket.receive_json()
                events.append(event)
                assert event.get("event") != "error", event
                if event.get("event") == "final":
                    break
            assert event["raw_text"] == "嘉宾是罗剑。"
            assert event["clean_text"] == "嘉宾是罗剑。"
            assert event["entity_audio_repair"]["change_count"] == 1
            assert any(c.get("stage") == "asr_audio_repair" for c in event["entity_normalizations"])
            assert any(e.get("stage") == "entity_audio_repair" for e in events)
        with client.websocket_connect("/ws/stream?entity_audio_repair=1&entity_asr_context=1") as socket:
            assert socket.receive_json()["event"] == "error"
    assert backend.replay_pcm == b"12345678" and not backend.sessions
