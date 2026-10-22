"""Concurrency and cleanup for the isolated entity replay session."""
import asyncio

from system.entity_audio_repair import _LegacyEntityAudioRepair as EntityAudioRepair
from system.entity_store import EntityDefinition

SOURCE = "今天我们一起讨论厉飞羽的相关问题。"
TARGET = SOURCE.replace("羽", "宇")
DEFINITION = EntityDefinition(1, "厉飞宇", "TERM", "general", "normalize", 0, ())


def idle_repair(backend):
    repair = EntityAudioRepair(backend, [DEFINITION])
    # These tests focus on background lifecycle after admission. The separate
    # fast-upload test exercises the real admission gate without this fixture.
    repair._streaming._idle_gaps.extend([.08] * 16)
    return repair


class ReplayBackend:
    def __init__(self, replay=TARGET):
        self.replay = replay
        self.sessions = {}
        self.received = {}
        self.starts = []
        self.count = 0
        self.active = 0
        self.max_active = 0
        self.background_started = asyncio.Event()

    async def __call__(self, endpoint, session_id=None, data=b"", params=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            if endpoint == "/stream/start":
                self.starts.append(params)
                self.count += 1
                sid = str(self.count)
                context = bool((params or {}).get("context"))
                self.sessions[sid] = context
                self.received[sid] = bytearray()
                if context:
                    self.background_started.set()
                return {"session_id": sid, "context_applied": context}
            if endpoint == "/stream/cancel":
                self.sessions.pop(session_id, None)
                return {}
            context = self.sessions[session_id]
            self.received[session_id].extend(data)
            text = self.replay if context else SOURCE
            if endpoint == "/stream/finish":
                self.sessions.pop(session_id)
            return {"text": text, "completed_segments": [{"segment_id": 1, "text": text}]}
        finally:
            self.active -= 1


async def wait_caught_up(helper):
    async def wait():
        while helper.processed_bytes != helper.audio_size:
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), 3)


def test_background_starts_before_finish_and_separate_sessions_receive_identical_pcm():
    async def scenario():
        backend = ReplayBackend()
        repair = idle_repair(backend)
        params = {"language": "Chinese", "chunk_seconds": 2.0, "custom_flag": "keep"}
        start = await repair.request("/stream/start", params=params)
        chunk = b"p" * (16 * 64000)
        for _ in range(3):
            partial = await repair.request("/stream/chunk", start["session_id"], chunk)
            assert partial["text"] == SOURCE
        await asyncio.wait_for(backend.background_started.wait(), 3)
        await wait_caught_up(repair._streaming)
        # New audio after the snapshot must reach both sessions exactly once.
        await repair.request("/stream/chunk", start["session_id"], b"next")
        await wait_caught_up(repair._streaming)
        final = await repair.request("/stream/finish", start["session_id"])
        assert final["text"] == TARGET
        assert final["entity_audio_repair"]["streaming_cache_used"]
        assert final["entity_audio_repair"]["replay_mode"] == "streaming"
        assert backend.count == 2
        assert backend.received['1'] == backend.received['2'] == chunk * 3 + b"next"
        assert len(backend.starts) == 2
        assert backend.starts[0]["chunk_seconds"] == 2.0
        assert backend.starts[1]["chunk_seconds"] == 10
        assert params == {"language": "Chinese", "chunk_seconds": 2.0, "custom_flag": "keep"}
        assert backend.max_active == 1
        assert not backend.sessions
        assert repair._audio is None
    asyncio.run(scenario())


def test_unrelated_shadow_rewrite_is_not_published_and_legacy_fallback_remains():
    async def scenario():
        backend = ReplayBackend("完全无关的内容。")
        repair = idle_repair(backend)
        start = await repair.request("/stream/start")
        await repair.request("/stream/chunk", start['session_id'], b"p" * (48 * 64000))
        await asyncio.wait_for(backend.background_started.wait(), 3)
        await wait_caught_up(repair._streaming)
        final = await repair.request("/stream/finish", start['session_id'])
        assert final['text'] == SOURCE
        assert not final['entity_audio_repair']['streaming_cache_used']
        assert backend.count == 3  # old full replay remains after rejected cache
        assert not backend.sessions
    asyncio.run(scenario())


def test_disconnect_cleans_shadow_without_cancelling_first_pass_early():
    async def scenario():
        backend = ReplayBackend()
        repair = idle_repair(backend)
        ready = await repair.request('/stream/start')
        await repair.request('/stream/chunk',ready['session_id'],b'p'*(48*64000))
        await asyncio.wait_for(backend.background_started.wait(),3)
        assert ready['session_id'] in backend.sessions
        await repair.request('/stream/cancel',ready['session_id'])
        await asyncio.gather(repair._streaming.task,return_exceptions=True)
        assert not backend.sessions
        assert repair._streaming.audio.closed
    asyncio.run(scenario())


def test_large_glossary_and_disabled_feature_use_legacy_mode():
    async def scenario(definitions, enabled):
        backend=ReplayBackend()
        repair=EntityAudioRepair(backend,definitions,streaming_replay=enabled)
        ready=await repair.request('/stream/start')
        await repair.request('/stream/chunk',ready['session_id'],b'p'*(48*64000))
        final=await repair.request('/stream/finish',ready['session_id'])
        assert not final['entity_audio_repair']['streaming_cache_used']
        assert final['text']==TARGET
        assert not backend.sessions
    asyncio.run(scenario([DEFINITION],False))
    definitions=[DEFINITION]+[EntityDefinition(i+2,f'其他词条{chr(0x4e00+i)}', 'TERM','general','normalize',0,()) for i in range(8)]
    asyncio.run(scenario(definitions,True))


def test_cancel_during_start_returns_promptly_and_late_session_is_cleaned():
    async def scenario():
        backend = ReplayBackend()
        entered = asyncio.Event()
        release = asyncio.Event()
        async def slow_start(endpoint, session_id=None, data=b"", params=None):
            if endpoint == '/stream/start' and (params or {}).get('context'):
                entered.set()
                await release.wait()
            return await backend(endpoint, session_id, data, params)
        repair = idle_repair(slow_start)
        ready = await repair.request('/stream/start')
        await repair.request('/stream/chunk', ready['session_id'], b'p'*(48*64000))
        await asyncio.wait_for(entered.wait(), 3)
        # Cancellation must not wait for the ASR start response.
        repair.close()
        await asyncio.wait_for(asyncio.gather(repair._streaming.task, return_exceptions=True), .2)
        release.set()
        await asyncio.gather(*repair._streaming.late_cleanup)
        assert list(backend.sessions) == [ready['session_id']]
        await backend('/stream/cancel', ready['session_id'])
        assert not backend.sessions
    asyncio.run(scenario())


def test_waiting_foreground_runs_before_another_background_request():
    async def scenario():
        backend = ReplayBackend()
        repair = idle_repair(backend)
        helper = repair._streaming
        order = []
        async def record(endpoint, session_id=None, data=b'', params=None):
            order.append(endpoint)
            return {}
        repair._request = record
        await helper.network_lock.acquire()
        background = asyncio.create_task(helper._background_request('background'))
        foreground = asyncio.create_task(helper.primary_request('foreground'))
        await asyncio.sleep(.02)
        helper.network_lock.release()
        await asyncio.gather(background, foreground)
        assert order == ['foreground', 'background']
        repair.close()
    asyncio.run(scenario())


def test_fast_upload_does_not_start_a_second_pass_before_finish():
    async def scenario():
        backend = ReplayBackend()
        repair = EntityAudioRepair(backend, [DEFINITION])
        ready = await repair.request('/stream/start')
        for _ in range(12):
            await repair.request('/stream/chunk', ready['session_id'], b'p'*(4*64000))
        assert backend.count == 1
        assert repair._streaming.task is None
        final = await repair.request('/stream/finish', ready['session_id'])
        assert final['text'] == TARGET
        assert not final['entity_audio_repair']['streaming_cache_used']
        assert backend.count == 2
        assert not backend.sessions
    asyncio.run(scenario())
