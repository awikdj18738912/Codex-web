"""Isolated, low-priority speculative replay for entity audio confirmation.

The first-pass ASR session is never replaced. Only a complete, anchored replay
can be reused at finish; large glossaries retain the existing candidate replay.
"""
from __future__ import annotations

import asyncio
from collections import deque
import json
import tempfile
import statistics
import time

from .entity_asr_context import EntityASRContext
from .entity_asr_echo_guard import EntityASREchoGuard
from .refinement_guard import join_refined_segments

STREAMING_REPLAY_RULE = "isolated_idle_replay_v2"


class StreamingEntityReplay:
    def __init__(self, owner):
        self.owner = owner
        self.task = None
        self.detector = None
        self.audio = tempfile.SpooledTemporaryFile(max_size=1048576, mode="w+b")
        self.audio_size = 0
        self.processed_bytes = 0
        self.primary_waiters = 0
        self.network_lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.finishing = False
        self.closed = False
        self.active_sessions = set()
        self.late_cleanup = set()
        self.result = None
        self.reason = "waiting_stable_candidate"
        self.started = None
        self.completed = None
        self.started_audio_seconds = None
        self.selection_key = None
        self._idle_gaps = deque(maxlen=16)
        self._last_primary_end = None
        self.terms = tuple(dict.fromkeys(d.canonical_text for d in owner.definitions))
        # Cover every potential term without changing a running ASR prompt.
        # Above the existing prompt limit, retain candidate-driven legacy mode.
        self.eligible = bool(self.terms) and len(self.terms) <= 8
        if not self.eligible:
            self.reason = "glossary_exceeds_streaming_prompt_limit"

    def snapshot(self):
        return {
            "rule": STREAMING_REPLAY_RULE, "eligible": self.eligible,
            "started": self.started is not None, "reason": self.reason,
            "processed_audio_seconds": round(self.processed_bytes / 64000, 3),
            "started_at_audio_seconds": self.started_audio_seconds,
            "complete": self.result is not None,
            "median_idle_gap_ms": round(statistics.median(self._idle_gaps) * 1000, 3)
            if self._idle_gaps else None,
            "wall_seconds": round((self.completed or time.monotonic()) - self.started, 3)
            if self.started is not None else 0.0,
        }

    async def primary_request(self, endpoint, session_id=None, data=b"", params=None):
        if endpoint == "/stream/chunk" and self._last_primary_end is not None:
            self._idle_gaps.append(max(0, time.monotonic() - self._last_primary_end))
        self.primary_waiters += 1
        try:
            async with self.network_lock:
                return await self.owner._request(endpoint, session_id, data, params)
        finally:
            self.primary_waiters -= 1
            if endpoint == "/stream/chunk":
                self._last_primary_end = time.monotonic()
            self.changed.set()

    def _has_idle_capacity(self):
        return len(self._idle_gaps) >= 6 and statistics.median(self._idle_gaps) >= 0.04

    async def _background_request(self, endpoint, session_id=None, data=b"", params=None):
        while True:
            # Give the next foreground request a chance to arrive after ack.
            # A running ASR request cannot be preempted, so serialize one chunk
            # at a time and never queue a batch ahead of the foreground.
            if not self.finishing:
                await asyncio.sleep(0.01)
                if not self._has_idle_capacity():
                    await asyncio.sleep(0.02)
                    continue
            async with self.network_lock:
                if self.primary_waiters:
                    pass
                else:
                    pending = asyncio.create_task(
                        self.owner._request(endpoint, session_id, data, params)
                    )
                    try:
                        payload = await asyncio.wait_for(asyncio.shield(pending), 2.0)
                    except (TimeoutError, asyncio.CancelledError):
                        # Release foreground priority immediately. If an
                        # in-flight start returns late, learn and cancel its
                        # temporary SID without delaying the user's result.
                        cleanup = asyncio.create_task(
                            self._cleanup_late_request(pending, endpoint, session_id)
                        )
                        self.late_cleanup.add(cleanup)
                        cleanup.add_done_callback(self.late_cleanup.discard)
                        raise
                    if endpoint == "/stream/start":
                        self.active_sessions.add(payload["session_id"])
                    elif endpoint in {"/stream/finish", "/stream/cancel"}:
                        self.active_sessions.discard(session_id)
                    return payload
            await asyncio.sleep(0.001)

    async def _cleanup_late_request(self, pending, endpoint, session_id):
        try:
            payload = await pending
            if endpoint == "/stream/start":
                await asyncio.wait_for(
                    self.owner._request("/stream/cancel", payload["session_id"]), 1.0
                )
            elif endpoint in {"/stream/finish", "/stream/cancel"}:
                self.active_sessions.discard(session_id)
        except Exception:
            pass

    def capture(self, data):
        if self.closed or not self.eligible:
            return
        self.audio.seek(0, 2)
        self.audio.write(data)
        self.audio_size += len(data)
        self.changed.set()

    def consider(self):
        owner = self.owner
        if (self.closed or not self.eligible or self.task is not None
                or owner._audio_size < 45 * 64000 or not owner._timeline_valid
                or not owner._segments or not owner.enabled
                or not self._has_idle_capacity()):
            return
        text = join_refined_segments(e["text"] for e in owner._segments.values())
        key = tuple((sid, e["text"]) for sid, e in owner._segments.items())
        if key == self.selection_key or (self.detector and not self.detector.done()):
            return
        self.selection_key = key
        self.detector = asyncio.create_task(self._detect(text))

    async def _detect(self, text):
        try:
            from .entity_audio_repair import GenericEntitySelector
            candidates, _ = await asyncio.to_thread(
                lambda: GenericEntitySelector(self.owner.definitions).select(text)
            )
            if self.closed or not self.owner._timeline_valid:
                return
            if max((c["retrieval_score"] for c in candidates), default=0) < self.owner.minimum_replay_score:
                return
            self.started = time.monotonic()
            self.started_audio_seconds = round(self.audio_size / 64000, 3)
            self.reason = "running"
            self.task = asyncio.create_task(self._run())
        except asyncio.CancelledError:
            raise
        except Exception:
            self.reason = "candidate_detection_failed"

    async def _run(self):
        from .entity_audio_repair import REPLAY_PREFIX, REPLAY_DECODE_SECONDS
        context = EntityASRContext(
            REPLAY_PREFIX + json.dumps(self.terms, ensure_ascii=False), self.terms
        )
        guard = EntityASREchoGuard(
            self._background_request, context,
            replay_chunk_bytes=self.owner.replay_chunk_bytes,
            max_audio_bytes=self.owner.max_audio_bytes,
        )
        try:
            ready = await guard.request("/stream/start", params=dict(
                self.owner._params, context=context.text,
                chunk_seconds=REPLAY_DECODE_SECONDS))
            if ready.get("context_applied") is not True:
                self.reason = "backend_context_unsupported"
                return
            while True:
                self.changed.clear()
                if self.processed_bytes < self.audio_size:
                    self.audio.seek(self.processed_bytes)
                    data = self.audio.read(min(self.owner.replay_chunk_bytes,
                                               self.audio_size - self.processed_bytes))
                    await guard.request("/stream/chunk", ready["session_id"], data)
                    self.processed_bytes += len(data)
                    continue
                if self.finishing:
                    payload = await guard.request("/stream/finish", ready["session_id"])
                    self.result = (str(payload.get("text", "")), guard.fallback_active)
                    self.reason = "complete"
                    return
                await self.changed.wait()
        except asyncio.CancelledError:
            self.reason = "cancelled"
            raise
        except Exception:
            self.reason = "background_replay_failed"
        finally:
            self.completed = time.monotonic()
            for sid in tuple(self.active_sessions):
                try:
                    await asyncio.wait_for(self.owner._request("/stream/cancel", sid), 1.0)
                except Exception:
                    pass
                self.active_sessions.discard(sid)
            guard.close()

    async def finish(self):
        self.finishing = True
        self.changed.set()
        if self.detector and not self.detector.done():
            self.detector.cancel()
            await asyncio.gather(self.detector, return_exceptions=True)
        if self.task is None:
            return None
        if self.audio_size - self.processed_bytes > 10 * 64000:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.reason = "background_too_far_behind"
            return None
        try:
            # Avoid a new long tail when fast uploads leave the shadow behind.
            await asyncio.wait_for(asyncio.shield(self.task), timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            if asyncio.current_task().cancelling():
                raise
            self.reason = "finish_budget_exhausted"
        return self.result

    def close(self):
        self.closed = True
        for task in (self.detector, self.task):
            if task and not task.done():
                task.cancel()
        # Worker reads from this spool only synchronously between awaits.
        if self.audio is not None:
            self.audio.close()
