"""Event-controlled tests for bounded translation snapshot scheduling."""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest

WEB_ROOT = Path(__file__).resolve().parents[1]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

from system.translation_scheduler import run_snapshot_pipeline, thread_call


def _run(coro):
    return asyncio.run(coro)


def test_fast_producer_coalesces_behind_blocked_translation_and_preserves_terminal():
    async def scenario():
        translate_started = asyncio.Event()
        release_translate = asyncio.Event()
        producer_done = asyncio.Event()
        published = []
        translated_sources = []

        async def stream():
            try:
                for source, final in [("old", False), ("middle", False), ("old", True)]:
                    yield source, final, {"p": source}
                    if source == "old" and not final:
                        await translate_started.wait()
            finally:
                producer_done.set()

        async def translate(snapshot):
            translated_sources.append(snapshot.source)
            if len(translated_sources) == 1:
                translate_started.set()
                await release_translate.wait()
            return {"translation": f"T:{snapshot.source}", "translation_ms": 1}

        async def refine(source, payload):
            return {"event": "final" if source.final else "refined",
                    "source_text": source.source, "refined": payload["translation"]}

        async def publish(event):
            published.append(event)

        pipeline = asyncio.create_task(run_snapshot_pipeline(stream(), translate, refine, publish))
        await asyncio.wait_for(translate_started.wait(), 2)
        # The stream must exhaust and deliver the last source while translation
        # remains blocked on its first snapshot.
        await asyncio.wait_for(producer_done.wait(), 2)
        assert not release_translate.is_set()
        release_translate.set()
        await asyncio.wait_for(pipeline, 2)

        finals = [event for event in published if event.get("event") == "final"]
        translations = [event for event in published if event.get("event") == "translation"]
        assert len(finals) == 1
        assert finals[0]["source_text"] == "old"
        assert finals[0]["refined"] == "T:old"
        assert len(translations) == 1
        assert translations[0]["source_text"] == "old"
        assert translated_sources == ["old", "old"]
        stats = finals[0]["pipeline_stats"]
        assert stats["max_pending_per_stage"] == 1
        assert stats["merged_source_snapshots"] >= 1
        assert stats["discarded_translation_results"] == 1

    _run(scenario())


def test_translation_continues_while_refinement_is_slow_and_old_translation_merges():
    async def scenario():
        refinement_started = asyncio.Event()
        release_refinement = asyncio.Event()
        translated_b = asyncio.Event()
        published_b = asyncio.Event()
        producer_done = asyncio.Event()
        published = []
        calls = []

        async def stream():
            try:
                yield "A", False, {}
                await refinement_started.wait()
                yield "B", False, {}
                await published_b.wait()
                yield "C", True, {}
            finally:
                producer_done.set()

        async def translate(snapshot):
            calls.append(snapshot.source)
            if snapshot.source == "B":
                translated_b.set()
            return {"translation": f"T:{snapshot.source}", "translation_ms": 1}

        async def refine(source, payload):
            if source.source == "A":
                refinement_started.set()
                await release_refinement.wait()
            return {"event": "final" if source.final else "refined",
                    "source_text": source.source, "refined": payload["translation"]}

        async def publish(event):
            published.append(event)
            if event.get("event") == "translation" and event.get("source_text") == "B":
                published_b.set()

        pipeline = asyncio.create_task(run_snapshot_pipeline(stream(), translate, refine, publish))
        await asyncio.wait_for(refinement_started.wait(), 2)
        await asyncio.wait_for(translated_b.wait(), 2)
        await asyncio.wait_for(producer_done.wait(), 2)
        assert not release_refinement.is_set()
        assert "B" in calls
        assert any(e.get("event") == "translation" and e.get("source_text") == "B" for e in published)
        release_refinement.set()
        await asyncio.wait_for(pipeline, 2)

        finals = [e for e in published if e.get("event") == "final"]
        assert len(finals) == 1
        assert finals[0]["source_text"] == "C"
        assert finals[0]["pipeline_stats"]["merged_translation_snapshots"] >= 1
        assert not any(e.get("event") == "refined" and e.get("source_text") == "B" for e in published)

    _run(scenario())


def test_translate_error_cancels_producer_and_closes_async_generator():
    async def scenario():
        producer_waiting = asyncio.Event()
        generator_closed = asyncio.Event()

        async def stream():
            try:
                yield "A", False, {}
                producer_waiting.set()
                await asyncio.Event().wait()
            finally:
                generator_closed.set()

        async def translate(snapshot):
            raise LookupError("translate failed")

        async def refine(source, payload):
            raise AssertionError("refine must not run")

        async def publish(event):
            pass

        with pytest.raises(LookupError, match="translate failed"):
            await asyncio.wait_for(run_snapshot_pipeline(stream(), translate, refine, publish), 2)
        assert producer_waiting.is_set()
        assert generator_closed.is_set()

    _run(scenario())


def test_refine_error_cancels_producer_and_closes_async_generator():
    async def scenario():
        producer_waiting = asyncio.Event()
        generator_closed = asyncio.Event()

        async def stream():
            try:
                yield "A", False, {}
                producer_waiting.set()
                await asyncio.Event().wait()
            finally:
                generator_closed.set()

        async def translate(snapshot):
            return {"translation": "T:A", "translation_ms": 1}

        async def refine(source, payload):
            raise RuntimeError("refine failed")

        async def publish(event):
            pass

        with pytest.raises(RuntimeError, match="refine failed"):
            await asyncio.wait_for(run_snapshot_pipeline(stream(), translate, refine, publish), 2)
        assert producer_waiting.is_set()
        assert generator_closed.is_set()

    _run(scenario())


def test_cancelling_thread_call_waits_for_real_worker_to_release():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocking_call():
            started.set()
            release.wait()
            finished.set()
            return "done"

        task = asyncio.create_task(thread_call(blocking_call))
        assert await asyncio.wait_for(asyncio.to_thread(started.wait), 2)
        task.cancel()
        # Yield one loop turn so cancellation reaches thread_call; its task must
        # remain alive until the synchronous worker actually exits.
        await asyncio.sleep(0)
        assert not task.done()
        assert not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert finished.is_set()

    _run(scenario())
