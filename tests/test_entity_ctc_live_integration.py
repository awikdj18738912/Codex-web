"""Isolated integration coverage for the live CTC/entity repair hooks."""
import asyncio
import struct
import threading
import time

import pytest

import system.entity_streaming_live as live
from system.entity_audio_repair import EntityAudioRepair
from system.entity_store import EntityDefinition
from system.entity_ctc_sidecar import EntityCtcSession


RAW = "今天我们一起讨论厉飞羽的相关问题。"
FIXED = RAW.replace("厉飞羽", "厉飞宇")
OTHER = RAW.replace("厉飞羽", "厉飞雨")
DEF = EntityDefinition(1, "厉飞宇", "TERM", "general", "normalize", 0, ())
OTHER_DEF = EntityDefinition(2, "厉飞雨", "TERM", "general", "normalize", 0, ())


def candidate(canonical="厉飞宇"):
    start = RAW.index("厉飞羽")
    return dict(start=start, end=start + 3, observed="厉飞羽", canonical=canonical,
                entity_id=1 if canonical == "厉飞宇" else 2,
                evidence_surface=canonical, retrieval_score=.94)


class FakeCtc:
    def __init__(self, proposals=(), *, submit_error=False, delay=0.):
        self.proposals = list(proposals)
        self.submit_error = submit_error
        self.delay = delay
        self.audio_seconds = 0.
        self.event_count = 1
        self.compute_seconds = 0.
        self.calls = []
        self.closed = False

    def submit_audio(self, data):
        if self.submit_error:
            raise RuntimeError("synthetic CTC failure")
        self.audio_seconds += len(data) / 64000

    def propose(self, text, *, final=False):
        self.calls.append((text, final))
        if self.delay:
            time.sleep(self.delay)
        return [dict(p) for p in self.proposals]

    async def finish(self):
        return None

    def close(self):
        self.closed = True


class FakeBackend:
    """Primary ASR emits RAW; a prompted replay emits its configured answer."""
    def __init__(self, replay_text=FIXED):
        self.replay_text = replay_text
        self.calls = []
        self.sessions = {}
        self.n = 0

    async def __call__(self, endpoint, sid=None, data=b"", params=None):
        params = dict(params or {})
        self.calls.append((endpoint, sid, bytes(data), params))
        if endpoint == "/stream/start":
            self.n += 1
            sid = str(self.n)
            replay = bool(params.get("context"))
            self.sessions[sid] = replay
            return dict(session_id=sid, context_applied=replay)
        if endpoint == "/stream/chunk":
            replay = self.sessions[sid]
            text = self.replay_text if replay else RAW
            return dict(text=text, completed_segments=[dict(segment_id=1, text=text)])
        if endpoint == "/stream/finish":
            replay = self.sessions[sid]
            return dict(text=self.replay_text if replay else RAW,
                        completed_segments=[] if replay else [dict(segment_id=1, text=RAW)])
        if endpoint == "/stream/cancel":
            self.sessions.pop(sid, None)
            return dict(cancelled=True)
        raise AssertionError(endpoint)


def freeze_selector(monkeypatch, candidates):
    class Selector:
        def select(self, text):
            return [dict(c) for c in candidates if c["observed"] in text], []
    monkeypatch.setattr(live, "_selector", lambda definitions: Selector())


def make_repair(backend, ctc, mode, *, definitions=(DEF,), request_timeout=.15):
    repair = EntityAudioRepair(backend, list(definitions), ctc_session=ctc, ctc_mode=mode,
        request_timeout=request_timeout, short_finish_max_seconds=4,
        short_finish_wait=1., finish_wait=.2, replay_chunk_bytes=64000)
    repair._live_primary_chunk_bytes = 1_000_000
    return repair


async def run_short(repair):
    sid = (await repair.request("/stream/start", params={"chunk_seconds": 1}))['session_id']
    partial = await repair.request("/stream/chunk", sid, b"a" * 64000)
    final = await repair.request("/stream/finish", sid)
    return partial, final


def test_shadow_keeps_partial_raw_and_still_runs_audio_review(monkeypatch):
    async def scenario():
        freeze_selector(monkeypatch, [candidate()])
        ctc = FakeCtc([candidate()])
        backend = FakeBackend()
        repair = make_repair(backend, ctc, "shadow")
        partial, final = await run_short(repair)
        assert partial["text"] == RAW
        assert final["text"] == FIXED
        assert repair.changes[0]["match_type"] == "AUDIO_REPLAY_CONFIRMED"
        assert any(call[0] == "/stream/start" and call[3].get("context") for call in backend.calls)
        assert repair.snapshot()["incremental"]["ctc"]["shadow_proposals"] >= 1
    asyncio.run(scenario())


def test_on_commits_final_ctc_proposal_at_eos_and_only_its_span(monkeypatch):
    async def scenario():
        # No glossary candidate means the ordinary audio path has no reason to edit.
        freeze_selector(monkeypatch, [])
        raw_span = RAW.index("厉飞羽")
        proposal = dict(candidate(), start=raw_span, end=raw_span + 3)
        ctc = FakeCtc([proposal])
        backend = FakeBackend()
        repair = make_repair(backend, ctc, "on")
        partial, final = await run_short(repair)
        expected = RAW[:raw_span] + "厉飞宇" + RAW[raw_span + 3:]
        assert partial["text"] == RAW
        assert final["text"] == expected
        assert final["completed_segments"][0]["text"] == expected
        assert repair.changes[0]["match_type"] == "CTC_ACOUSTIC_SUPPORTED"
        assert repair.snapshot()["incremental"]["ctc"]["committed"] == 1
        assert not any(call[0] == "/stream/start" and call[3].get("context") for call in backend.calls)
    asyncio.run(scenario())


def test_no_ctc_proposal_preserves_short_recording_audio_correction(monkeypatch):
    async def scenario():
        freeze_selector(monkeypatch, [candidate()])
        backend = FakeBackend()
        repair = make_repair(backend, FakeCtc([]), "on")
        _, final = await run_short(repair)
        assert final["text"] == FIXED
        assert repair.changes[0]["match_type"] == "AUDIO_REPLAY_CONFIRMED"
        assert repair.snapshot()["incremental"]["ctc"]["committed"] == 0
    asyncio.run(scenario())


def test_audio_review_can_replace_overlapping_ctc_patch_with_other_entity(monkeypatch):
    async def scenario():
        # The replay confirms a competing glossary entity at the same source span.
        freeze_selector(monkeypatch, [candidate("厉飞雨")])
        ctc = FakeCtc([candidate("厉飞宇")])
        backend = FakeBackend(replay_text=OTHER)
        repair = make_repair(backend, ctc, "on", definitions=(DEF, OTHER_DEF))
        _, final = await run_short(repair)
        assert final["text"] == OTHER
        assert repair.changes[0]["canonical"] == "厉飞雨"
        assert repair.changes[0]["match_type"] == "AUDIO_REPLAY_CONFIRMED"
        assert repair.snapshot()["incremental"]["ctc"]["committed"] == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["submit", "proposal_timeout"])
def test_ctc_failure_or_timeout_leaves_audio_review_available(monkeypatch, failure):
    async def scenario():
        freeze_selector(monkeypatch, [candidate()])
        ctc = FakeCtc([], submit_error=failure == "submit",
                      delay=1.1 if failure == "proposal_timeout" else 0.)
        backend = FakeBackend()
        repair = make_repair(backend, ctc, "on")
        _, final = await run_short(repair)
        assert final["text"] == FIXED
        assert repair.changes[0]["match_type"] == "AUDIO_REPLAY_CONFIRMED"
        assert any(call[0] == "/stream/start" and call[3].get("context") for call in backend.calls)
        if failure == "submit":
            assert repair.snapshot()["incremental"]["ctc"]["errors"] >= 1
    asyncio.run(scenario())


def test_real_session_limits_backlog_and_rejects_nonfinite_pcm():
    class Stream:
        def __init__(self, gate=None):
            self.gate = gate

        def accept_audio(self, audio):
            if self.gate is not None:
                assert self.gate.wait(5.)
            return []

    class Model:
        def __init__(self, gate=None):
            self.gate = gate

        def create_stream(self):
            return Stream(self.gate)

    gate = threading.Event()
    blocked = EntityCtcSession(Model(gate), object(), (DEF,), -2., .1)
    try:
        blocked.submit_audio(b'\0' * 64000)
        with pytest.raises(RuntimeError, match='backlog'):
            blocked.submit_audio(b'\0' * (8 * 64000))
    finally:
        gate.set()
        blocked.close()

    invalid = EntityCtcSession(Model(), object(), (DEF,), -2., .1)
    try:
        invalid.submit_audio(struct.pack('<f', float('nan')))
        with pytest.raises(ValueError, match='non-finite'):
            invalid._futures[-1].result(timeout=3.)
        assert isinstance(invalid.error, ValueError)
    finally:
        invalid.close()


@pytest.fixture(autouse=True)
def isolate_audio_review_from_direct_spelling(monkeypatch):
    """These cases exercise acoustic review; direct spelling has its own suite."""
    from system.entity_streaming_live import LiveEntityMixin
    async def no_direct_spelling(self, raw_text, *, final=False):
        return None
    monkeypatch.setattr(LiveEntityMixin, "_live_apply_glossary", no_direct_spelling)
