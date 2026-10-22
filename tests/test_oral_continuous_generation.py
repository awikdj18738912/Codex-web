import asyncio
import json
from types import SimpleNamespace

import pytest

from pipeline.src.generators import oral_text_generator as oral
from pipeline.src.utils.oral_generation_state import OralGenerationState


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_legacy_migration_recovers_jsonl_ahead_of_checkpoint(tmp_path):
    original = [{"id": 0, "oral_text": "第一条。"}, {"id": 2, "oral_text": "第二条。"}]
    (tmp_path / "oral.jsonl").write_text("".join(json.dumps(x) + "\n" for x in original))
    (tmp_path / "oral.ckpt.json").write_text(json.dumps({"generated_count": 1, "attempted_count": 3}))
    state = OralGenerationState(tmp_path)
    assert state.generated == 2
    assert state.next_id == state.completed == 3
    attempt_id = state.reserve()
    assert attempt_id == 3
    state.finish(attempt_id, {"id": attempt_id, "oral_text": "第三条。"})
    state.close()
    resumed = OralGenerationState(tmp_path)
    assert resumed.generated == 3
    assert [x["id"] for x in records(tmp_path / "oral.jsonl")] == [0, 2, 3]
    resumed.close()


def test_committed_result_survives_failure_before_jsonl_append(tmp_path, monkeypatch):
    state = OralGenerationState(tmp_path)
    attempt_id = state.reserve()
    monkeypatch.setattr(state, "_append", lambda _: (_ for _ in ()).throw(OSError("interrupted write")))
    with pytest.raises(OSError):
        state.finish(attempt_id, {"id": attempt_id, "oral_text": "已完成结果。"})
    state.close()
    resumed = OralGenerationState(tmp_path)
    assert resumed.generated == 1
    assert resumed.pending_ids() == []
    assert records(tmp_path / "oral.jsonl") == [{"id": 0, "oral_text": "已完成结果。"}]
    resumed.close()


def test_reserved_ids_are_retried_and_second_writer_is_rejected(tmp_path):
    state = OralGenerationState(tmp_path)
    assert state.reserve() == 0
    assert state.reserve() == 1
    with pytest.raises(RuntimeError, match="already owns"):
        OralGenerationState(tmp_path)
    state.finish(1, {"id": 1, "oral_text": "先完成。"})
    state.close()
    resumed = OralGenerationState(tmp_path)
    assert resumed.pending_ids() == [0]
    assert resumed.next_id == 2
    resumed.finish(0, {"id": 0, "oral_text": "后完成。"})
    assert [x["id"] for x in records(tmp_path / "oral.jsonl")] == [1, 0]
    resumed.close()


def generator(tmp_path, monkeypatch, generate_one):
    monkeypatch.setattr(oral, "LLM_MAX_CONCURRENT", 2)
    instance = oral.OralTextGenerator.__new__(oral.OralTextGenerator)
    instance.raw_dir = tmp_path
    instance.llm = SimpleNamespace(max_concurrent=2)
    instance.generate_one = generate_one
    return instance


def test_slow_request_does_not_block_refilling_other_slot(tmp_path, monkeypatch):
    async def run():
        release_slow = asyncio.Event()
        slow_done = False

        async def generate_one(scene, attempt_id):
            nonlocal slow_done
            if attempt_id == 0:
                await release_slow.wait()
                slow_done = True
            if attempt_id == 4:
                assert not slow_done
                release_slow.set()
            await asyncio.sleep(0)
            return {"id": attempt_id, "oral_text": f"第{attempt_id}条。"}

        instance = generator(tmp_path, monkeypatch, generate_one)
        assert await asyncio.wait_for(instance.generate_scene("daily_chat", 6), timeout=3) == 6
        output = records(tmp_path / "daily_chat/oral.jsonl")
        assert len(output) == 6
        assert {x["id"] for x in output} == set(range(6))

    asyncio.run(run())


def test_cancel_and_resume_preserves_completed_records_and_pending_ids(tmp_path, monkeypatch):
    async def run():
        third_started = asyncio.Event()
        never = asyncio.Event()

        async def generate_one(scene, attempt_id):
            if attempt_id == 1:
                return {"id": 1, "oral_text": "中断前已完成。"}
            if attempt_id == 2:
                third_started.set()
            await never.wait()

        instance = generator(tmp_path, monkeypatch, generate_one)
        task = asyncio.create_task(instance.generate_scene("daily_chat", 4))
        await asyncio.wait_for(third_started.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [x["id"] for x in records(tmp_path / "daily_chat/oral.jsonl")] == [1]

        async def resumed_one(scene, attempt_id):
            return {"id": attempt_id, "oral_text": f"恢复结果{attempt_id}。"}

        instance.generate_one = resumed_one
        assert await instance.generate_scene("daily_chat", 4) == 4
        output = records(tmp_path / "daily_chat/oral.jsonl")
        assert len(output) == len({x["id"] for x in output}) == 4
        assert next(x for x in output if x["id"] == 1)["oral_text"] == "中断前已完成。"

    asyncio.run(run())


def test_failed_generations_are_replaced_without_exceeding_target(tmp_path, monkeypatch):
    async def run():
        async def generate_one(scene, attempt_id):
            return None if attempt_id == 1 else {"id": attempt_id, "oral_text": "有效文本。"}

        instance = generator(tmp_path, monkeypatch, generate_one)
        assert await instance.generate_scene("daily_chat", 5) == 5
        output = records(tmp_path / "daily_chat/oral.jsonl")
        assert len(output) == 5
        assert len({x["id"] for x in output}) == 5
        checkpoint = json.loads((tmp_path / "daily_chat/oral.ckpt.json").read_text())
        assert checkpoint["attempted_count"] == 6
        assert checkpoint["pending_attempts"] == 0

    asyncio.run(run())
