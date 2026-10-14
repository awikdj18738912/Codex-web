"""Focused tests for the replay chunk-size adapter in this experiment."""
import asyncio
import math
from copy import deepcopy

import pytest

from results.entity_replay_decode_step_20261014.candidate import EfficientEntityRepair
from system.entity_store import EntityDefinition


def entity(name="罗剑"):
    return EntityDefinition(1, name, "PERSON", "medical", "normalize", 0, ())


class FakeBackend:
    def __init__(self, *, replay_text="嘉宾是罗剑。", initial_context_applied=False,
                 slow_replay=False):
        self.replay_text = replay_text
        self.initial_context_applied = initial_context_applied
        self.slow_replay = slow_replay
        self.sessions = {}
        self.calls = []
        self.replay_pcm = b""
        self.replay_started = asyncio.Event()
        self.next_session = 0

    async def __call__(self, endpoint, session_id=None, data=b"", params=None):
        self.calls.append((endpoint, session_id, data, deepcopy(params)))
        if endpoint == "/stream/start":
            self.next_session += 1
            sid = str(self.next_session)
            context = bool((params or {}).get("context"))
            self.sessions[sid] = {"context": context, "pcm": bytearray()}
            applied = context or (self.next_session == 1 and self.initial_context_applied)
            return {"session_id": sid, "context_applied": applied}
        if endpoint == "/stream/cancel":
            self.sessions.pop(session_id, None)
            return {"cancelled": True}

        session = self.sessions[session_id]
        if session["context"]:
            self.replay_started.set()
            if self.slow_replay:
                await asyncio.sleep(10)
        if endpoint == "/stream/chunk":
            session["pcm"].extend(data)
            if session["context"]:
                self.replay_pcm = bytes(session["pcm"])

        text = self.replay_text if session["context"] else "嘉宾是罗建。"
        result = {
            "text": text,
            "language": "Chinese",
            "confidence": 0.99,
            "confidence_covers_full_text": True,
            "completed_segments": [{"segment_id": 1, "text": text, "confidence": 0.99}],
        }
        if endpoint == "/stream/finish":
            self.sessions.pop(session_id, None)
        return result


async def stream(guard, pcm=b"same-pcm"):
    ready = await guard.request("/stream/start", params={"language": "Chinese", "chunk_seconds": 2.0})
    await guard.request("/stream/chunk", ready["session_id"], pcm)
    return ready, await guard.request("/stream/finish", ready["session_id"])


def test_initial_and_review_start_params_are_isolated_and_primary_params_unchanged():
    async def scenario():
        backend = FakeBackend()
        guard = EfficientEntityRepair(backend, [entity()], streaming_replay=False,
                                      local_replay=False, replay_chunk_bytes=4)
        primary_start = {"language": "Chinese", "chunk_seconds": 2.0, "tag": "primary-start"}
        ready = await guard.request("/stream/start", params=primary_start)
        primary_chunk = {"chunk_seconds": 2.0, "tag": "primary-chunk"}
        await guard.request("/stream/chunk", ready["session_id"], b"same-pcm", primary_chunk)
        primary_finish = {"chunk_seconds": 2.0, "tag": "primary-finish"}
        result = await guard.request("/stream/finish", ready["session_id"], params=primary_finish)

        starts = [c for c in backend.calls if c[0] == "/stream/start"]
        assert starts[0][3] == primary_start
        assert starts[1][3]["language"] == "Chinese"
        assert "罗剑" in starts[1][3]["context"]
        assert starts[1][3]["chunk_seconds"] == 10.0
        assert primary_start == {"language": "Chinese", "chunk_seconds": 2.0, "tag": "primary-start"}
        assert primary_chunk == {"chunk_seconds": 2.0, "tag": "primary-chunk"}
        assert primary_finish == {"chunk_seconds": 2.0, "tag": "primary-finish"}
        chunks = [c for c in backend.calls if c[0] == "/stream/chunk"]
        finishes = [c for c in backend.calls if c[0] == "/stream/finish"]
        assert chunks[0][3] == primary_chunk and finishes[0][3] == primary_finish
        assert backend.replay_pcm == b"same-pcm"
        assert result["text"] == "嘉宾是罗剑。"
        assert not backend.sessions and guard._audio is None

    asyncio.run(scenario())


def test_initial_backend_context_suppresses_an_extra_verification():
    async def scenario():
        backend = FakeBackend(initial_context_applied=True)
        guard = EfficientEntityRepair(backend, [entity()], streaming_replay=False)
        ready, result = await stream(guard)
        assert ready["context_applied"] is True
        assert result["text"] == "嘉宾是罗建。"
        assert len([c for c in backend.calls if c[0] == "/stream/start"]) == 1
        assert not backend.sessions and guard._audio is None

    asyncio.run(scenario())


def test_empty_entity_definitions_do_not_start_a_replay():
    async def scenario():
        backend = FakeBackend()
        guard = EfficientEntityRepair(backend, [], streaming_replay=False)
        _, result = await stream(guard)
        assert result["text"] == "嘉宾是罗建。"
        assert len([c for c in backend.calls if c[0] == "/stream/start"]) == 1
        assert not backend.sessions and guard._audio is None

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_close_replay_session_and_audio(cancel):
    async def scenario():
        backend = FakeBackend(slow_replay=True)
        guard = EfficientEntityRepair(backend, [entity()], streaming_replay=False,
                                      local_replay=False,
                                      verification_timeout=30 if cancel else 0.05)
        ready = await guard.request("/stream/start", params={"chunk_seconds": 2.0})
        await guard.request("/stream/chunk", ready["session_id"], b"same-pcm")
        task = asyncio.create_task(guard.request("/stream/finish", ready["session_id"]))
        if cancel:
            await asyncio.wait_for(backend.replay_started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result["text"] == "嘉宾是罗建。"
            assert result["entity_audio_repair"]["reason"] == "TimeoutError"
        assert not backend.sessions and guard._audio is None
        assert any(call[0] == "/stream/cancel" for call in backend.calls)

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_review_chunk_seconds_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="finite and positive"):
        EfficientEntityRepair(FakeBackend(), [entity()], review_decode_seconds=value,
                              streaming_replay=False)
