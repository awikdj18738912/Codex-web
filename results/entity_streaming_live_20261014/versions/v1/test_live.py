"""Bounded-state tests for the private streaming entity experiment.

These fakes verify routing and publication behavior only. They do not measure
ASR accuracy or establish that an acoustic correction is correct.
"""
import asyncio
import importlib.util
from pathlib import Path
import sys

import pytest
from system.entity_store import EntityDefinition


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("system._live_candidate_test", ROOT / "candidate.py")
candidate_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate_module
SPEC.loader.exec_module(candidate_module)
Repair = candidate_module.StreamingEntityRepair
live = candidate_module.live

SOURCE = "今天我们一起讨论厉飞羽的相关问题。"
TARGET = SOURCE.replace("厉飞羽", "厉飞宇")
DEFINITION = EntityDefinition(1, "厉飞宇", "TERM", "general", "normalize", 0, ())


def entity_candidate(text=SOURCE, observed="厉飞羽", canonical="厉飞宇", score=.92):
    start = text.index(observed)
    return dict(start=start, end=start + len(observed), observed=observed,
                canonical=canonical, entity_id=1, evidence_surface=canonical,
                retrieval_score=score)


class FakeBackend:
    """Primary ASR stays raw; the prompted local replay is controlled by test."""
    def __init__(self, *, replay_text=TARGET, replay_segments=None, primary=None,
                 delay=0., start_delay=0., fail=False):
        self.replay_text = replay_text
        self.replay_segments = replay_segments
        self.primary = list(primary or [(SOURCE, [dict(segment_id=1, text=SOURCE)])])
        self.delay = delay
        self.start_delay = start_delay
        self.fail = fail
        self.calls = []
        self.sessions = {}
        self.counter = 0
        self.primary_chunks = 0

    async def __call__(self, endpoint, sid=None, data=b"", params=None):
        self.calls.append((endpoint, sid, bytes(data), dict(params or {})))
        if endpoint == "/stream/start":
            if self.start_delay and (params or {}).get("context"):
                await asyncio.sleep(self.start_delay)
            self.counter += 1
            sid = str(self.counter)
            replay = bool((params or {}).get("context"))
            self.sessions[sid] = dict(replay=replay, pcm=bytearray())
            return dict(session_id=sid, context_applied=replay)
        if endpoint == "/stream/cancel":
            self.sessions.pop(sid, None)
            return dict(cancelled=True)
        state = self.sessions[sid]
        if endpoint == "/stream/chunk":
            state["pcm"].extend(data)
            if state["replay"]:
                if self.delay:
                    await asyncio.sleep(self.delay)
                if self.fail:
                    raise RuntimeError("synthetic local review failure")
                segments = self.replay_segments
                if segments is None:
                    segments = [dict(segment_id=1, text=self.replay_text)]
                return dict(text=self.replay_text, completed_segments=segments)
            index = min(self.primary_chunks, len(self.primary) - 1)
            self.primary_chunks += 1
            item = self.primary[index]
            text, segments = item[:2]
            result = dict(text=text, completed_segments=segments)
            if len(item) > 2:
                result["asr_timing"] = {"model_ms": item[2]}
            return result
        if endpoint == "/stream/finish":
            if state["replay"]:
                return dict(text=self.replay_text, completed_segments=[])
            index = min(self.primary_chunks, len(self.primary) - 1)
            text, segments = self.primary[index][:2]
            self.sessions.pop(sid, None)
            return dict(text=text, completed_segments=segments)
        raise AssertionError(f"unexpected endpoint: {endpoint}")


def freeze_selector(monkeypatch, candidates):
    class FrozenSelector:
        def select(self, text):
            return [dict(c) for c in candidates if c["observed"] in text], []
    monkeypatch.setattr(live, "_selector", lambda definitions: FrozenSelector())


def make_repair(backend, *, window_seconds=32., cache_seconds=96., finish_wait=.8,
                request_timeout=.2):
    return Repair(backend, [DEFINITION], candidate_score=.63, candidate_margin=.04,
                  window_seconds=window_seconds, cache_seconds=cache_seconds,
                  finish_wait=finish_wait, request_timeout=request_timeout,
                  replay_chunk_bytes=64000)


async def wait_worker(repair, timeout=2.):
    worker = repair._live_worker
    if worker is not None:
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), timeout)


async def wait_cleanup(repair, timeout=2.):
    async def drain():
        while repair._cleanup_tasks:
            await asyncio.gather(*tuple(repair._cleanup_tasks), return_exceptions=True)
    await asyncio.wait_for(drain(), timeout)


def test_confirmed_local_revision_updates_raw_and_segments_and_survives_later_raw(monkeypatch):
    async def scenario():
        freeze_selector(monkeypatch, [entity_candidate()])
        backend = FakeBackend(primary=[(SOURCE, [dict(segment_id=1, text=SOURCE)])] * 3)
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start", params={"chunk_seconds": 2}))['session_id']
        await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        await wait_worker(repair)
        next_raw = await repair.request("/stream/chunk", sid, b"b" * 64000)
        assert next_raw["text"] == TARGET
        assert next_raw["completed_segments"] == [dict(segment_id=1, text=TARGET,
                                                           confidence=None, confidence_metadata={})]
        final = await repair.request("/stream/finish", sid)
        assert final["text"] == TARGET
        assert final["completed_segments"][0]["text"] == TARGET
        assert final["entity_audio_repair_revision"] == next_raw["entity_audio_repair_revision"]
    asyncio.run(scenario())


def test_primary_params_are_preserved_and_local_replay_uses_ten_second_decode(monkeypatch):
    async def scenario():
        freeze_selector(monkeypatch, [entity_candidate()])
        backend = FakeBackend()
        repair = make_repair(backend)
        params = {"chunk_seconds": 2, "language": "Chinese", "custom": "keep"}
        sid = (await repair.request("/stream/start", params=params))["session_id"]
        await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        await wait_worker(repair)
        starts = [c[3] for c in backend.calls if c[0] == "/stream/start"]
        assert starts[0] == params
        assert starts[1]["chunk_seconds"] == 10.0
        assert starts[1]["language"] == "Chinese" and starts[1]["custom"] == "keep"
        assert starts[1].get("context")
    asyncio.run(scenario())


def test_repeated_candidate_does_not_send_audio_twice(monkeypatch):
    async def scenario():
        freeze_selector(monkeypatch, [entity_candidate()])
        backend = FakeBackend(primary=[(SOURCE, [dict(segment_id=1, text=SOURCE)])] * 3)
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start"))["session_id"]
        await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        await wait_worker(repair)
        await repair.request("/stream/chunk", sid, b"b" * 64000)
        await wait_worker(repair)
        replay_chunks = [c for c in backend.calls if c[0] == "/stream/chunk" and c[1] != sid]
        assert len(replay_chunks) == 1
        # The fake replay confirms the candidate on its first 1-second chunk,
        # so review stops early and the remaining window is not sent.
        assert replay_chunks[0][2] == b"a" * 64000
    asyncio.run(scenario())


@pytest.mark.parametrize("raw,canonical", [("五个", "误工"), ("不是宗", "不业之杖")])
def test_low_score_candidates_do_not_enter_audio_review(monkeypatch, raw, canonical):
    async def scenario():
        text = f"我们讨论{raw}的问题并继续说明。"
        c = entity_candidate(text, raw, canonical, .58)
        freeze_selector(monkeypatch, [c])
        backend = FakeBackend(primary=[(text, [dict(segment_id=1, text=text)])])
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start"))["session_id"]
        result = await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        await wait_worker(repair)
        assert result["text"] == text
        assert not [c for c in backend.calls if c[0] == "/stream/start" and c[3].get("context")]
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["timeout", "failure", "echo"])
def test_failed_or_echo_review_keeps_source_and_finish_never_starts_another_review(monkeypatch, mode):
    async def scenario():
        freeze_selector(monkeypatch, [entity_candidate()])
        echo = "参考术语仅用于辅助拼写"
        backend = FakeBackend(replay_text=echo if mode == "echo" else TARGET,
                             delay=.08 if mode == "timeout" else 0.,
                             fail=mode == "failure")
        repair = make_repair(backend, request_timeout=.01 if mode == "timeout" else .2,
                             finish_wait=.03)
        sid = (await repair.request("/stream/start"))["session_id"]
        partial = await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        await wait_worker(repair)
        assert partial["text"] == SOURCE
        starts_before_finish = sum(c[0] == "/stream/start" for c in backend.calls)
        final = await repair.request("/stream/finish", sid)
        assert final["text"] == SOURCE
        assert sum(c[0] == "/stream/start" for c in backend.calls) == starts_before_finish
        assert repair._live_reserved_bytes >= 0
        assert repair._live_reserved_bytes == 0
        assert not any(c[0] == "/stream/chunk" and len(c[2]) > 32 * 64000
                       for c in backend.calls if c[1] != sid)
    asyncio.run(scenario())


@pytest.mark.parametrize("late", ["start", "chunk"])
def test_late_review_start_or_chunk_is_cancelled_and_releases_budget(monkeypatch, late):
    async def scenario():
        freeze_selector(monkeypatch, [entity_candidate()])
        backend = FakeBackend(start_delay=.06 if late == "start" else 0.,
                              delay=.06 if late == "chunk" else 0.)
        repair = make_repair(backend, request_timeout=.01)
        sid = (await repair.request("/stream/start"))["session_id"]
        await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        await wait_worker(repair)
        assert repair._live_reserved_bytes >= 0
        await wait_cleanup(repair)
        assert repair._live_reserved_bytes == 0
        replay_starts = [c for c in backend.calls if c[0] == "/stream/start" and c[3].get("context")]
        assert replay_starts
        replay_session = next((key for key, value in backend.sessions.items()
                               if value["replay"]), None)
        assert replay_session is None
        cancelled = [c[1] for c in backend.calls if c[0] == "/stream/cancel"]
        assert len(cancelled) == 1
        assert cancelled[0] != sid
    asyncio.run(scenario())


def test_finish_does_not_start_review_for_final_only_candidate(monkeypatch):
    async def scenario():
        freeze_selector(monkeypatch, [entity_candidate()])
        interim = "今天讨论其他内容。"
        backend = FakeBackend(primary=[(interim, [dict(segment_id=1, text=interim)]),
                                       (SOURCE, [dict(segment_id=1, text=SOURCE)])])
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start"))["session_id"]
        await repair.request("/stream/chunk", sid, b"a" * 64000)
        # A queued window is discarded at finish and its reservation released.
        queued_audio = b"q" * 64000
        repair._live_queue.append(dict(start=0, end=len(queued_audio), pcm=queued_audio,
                                        text=interim, candidates=[entity_candidate()]))
        repair._live_reserved_bytes += len(queued_audio)
        final = await repair.request("/stream/finish", sid)
        assert final["text"] == SOURCE
        assert not [c for c in backend.calls if c[0] == "/stream/start" and c[3].get("context")]
        assert repair._live_queue == []
        assert repair._live_reserved_bytes == 0
    asyncio.run(scenario())


def test_oversized_window_is_skipped_and_audio_cache_stays_within_96_seconds(monkeypatch):
    async def scenario():
        freeze_selector(monkeypatch, [entity_candidate()])
        backend = FakeBackend()
        repair = make_repair(backend, window_seconds=32., cache_seconds=96.)
        sid = (await repair.request("/stream/start"))["session_id"]
        # One stable segment spans more than the permitted local review window.
        await repair.request("/stream/chunk", sid, b"a" * (40 * 64000))
        await wait_worker(repair)
        assert not [c for c in backend.calls if c[0] == "/stream/start" and c[3].get("context")]
        # Further audio without a matching entity grows only the bounded cache.
        freeze_selector(monkeypatch, [])
        await repair.request("/stream/chunk", sid, b"b" * (70 * 64000))
        assert len(repair._live_pcm) <= 96 * 64000
        assert repair._live_pcm_start == 14 * 64000
    asyncio.run(scenario())


def test_changed_source_withdraws_previous_revision_and_increments_version(monkeypatch):
    async def scenario():
        freeze_selector(monkeypatch, [entity_candidate()])
        changed = "今天我们一起讨论厉飞羽的新问题。"
        backend = FakeBackend(primary=[(SOURCE, [dict(segment_id=1, text=SOURCE)]),
                                       (SOURCE, [dict(segment_id=1, text=SOURCE)]),
                                       (changed, [dict(segment_id=1, text=changed)]),
                                       (changed, [dict(segment_id=1, text=changed)])])
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start"))["session_id"]
        await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        await wait_worker(repair)
        published = await repair.request("/stream/chunk", sid, b"b" * 64000)
        assert published["text"] == TARGET
        withdrawn = await repair.request("/stream/chunk", sid, b"c" * 64000)
        assert withdrawn["text"] == changed
        assert withdrawn["entity_audio_repair_revision"] > published["entity_audio_repair_revision"]
    asyncio.run(scenario())


def test_stable_partial_prefix_can_start_local_review_without_committing_segment(monkeypatch):
    async def scenario():
        text = "今天我们一起讨论厉飞羽的相关问题，后续还需要继续展开详细讨论。"
        target = text.replace("厉飞羽", "厉飞宇")
        freeze_selector(monkeypatch, [entity_candidate(text)])
        partial_events = []
        backend = FakeBackend(replay_text=target,
                              primary=[(text, partial_events, 5.),
                                       (text, partial_events, 5.),
                                       (text, partial_events, 5.)])
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start"))["session_id"]
        first = await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        assert first["completed_segments"] == []
        second = await repair.request("/stream/chunk", sid, b"b" * 64000)
        await wait_worker(repair)
        assert any(c[0] == "/stream/start" and c[3].get("context") for c in backend.calls)
        revised = await repair.request("/stream/chunk", sid, b"c" * 64000)
        assert revised["text"] == target
        assert revised["completed_segments"] == []
        assert second["completed_segments"] == []
    asyncio.run(scenario())


def test_repeated_partial_transport_without_new_decode_does_not_start_review(monkeypatch):
    async def scenario():
        text = "今天我们一起讨论厉飞羽的相关问题，后续还需要继续展开详细讨论。"
        freeze_selector(monkeypatch, [entity_candidate(text)])
        partial_events = []
        backend = FakeBackend(primary=[(text, partial_events), (text, partial_events)])
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start"))["session_id"]
        await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        await repair.request("/stream/chunk", sid, b"b" * 64000)
        await wait_worker(repair)
        assert not [c for c in backend.calls if c[0] == "/stream/start" and c[3].get("context")]
    asyncio.run(scenario())


def test_entity_scoped_across_two_stable_segments_keeps_text_and_segments_aligned(monkeypatch):
    async def scenario():
        text = "今天我们一起讨论厉飞羽的相关问题。"
        split = text.index("羽")
        first_seg = text[:split]
        second_seg = text[split:]
        segments = [dict(segment_id=1, text=first_seg), dict(segment_id=2, text=second_seg)]
        freeze_selector(monkeypatch, [entity_candidate(text)])
        backend = FakeBackend(replay_text=TARGET,
                              replay_segments=[dict(segment_id=1, text=TARGET)],
                              primary=[(text, segments), (text, segments)])
        repair = make_repair(backend)
        sid = (await repair.request("/stream/start"))["session_id"]
        await repair.request("/stream/chunk", sid, b"a" * (4 * 64000))
        await wait_worker(repair)
        revised = await repair.request("/stream/chunk", sid, b"b" * 64000)
        assert revised["text"] == TARGET
        assert "".join(e["text"] for e in revised["completed_segments"]) == TARGET
        assert [e["segment_id"] for e in revised["completed_segments"]] == [1, 2]
    asyncio.run(scenario())
