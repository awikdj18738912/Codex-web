"""Small real-teacher pilot; uses production generators in an isolated output tree."""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from contextvars import ContextVar
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MODEL = "/home/aim0/data/models/ASR/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
os.environ.update(VLLM_BASE_URL="http://127.0.0.1:8010/v1", VLLM_MODEL_NAME="refiner-teacher",
                  OPENROUTER_API_KEY="", OPENROUTER_MODEL="")
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))
from configs import settings

for key, subdir in (("SEEDS_DIR", "seeds"), ("RAW_DIR", "raw"),
                    ("PROCESSED_DIR", "processed"), ("FINAL_DIR", "final")):
    path = OUT / subdir
    path.mkdir(parents=True, exist_ok=True)
    setattr(settings, key, str(path))

from src.generators.seed_generator import SeedGenerator
from src.generators.oral_text_generator import OralTextGenerator
from src.generators.clean_text_generator import CleanTextGenerator
from src.generators.asr_simulator import ASRSimulator
from src.pipeline.dedup import deduplicate
from src.services.llm_client import AsyncLLMClient, LLMClient
from src.utils.numeric_policy import lock_training_target, normalize_source


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "pipeline/scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assemble = load_script("pilot_assemble", "04_assemble.py")
finalize = load_script("pilot_finalize", "05_finalize.py")
exporter = load_script("pilot_export", "export_sft.py")
context = ContextVar("request_context", default={})
CALLS = []


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


class AuditedClient(AsyncLLMClient):
    async def generate(self, **kwargs):
        started = time.monotonic()
        record = {**context.get(), "request": kwargs}
        try:
            result = await super().generate(**kwargs)
            record["response"] = result
            return result
        except Exception as exc:
            record["error"] = repr(exc)
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            CALLS.append(record)
            append_jsonl(OUT / "api_calls.jsonl", record)


async def tagged(stage, item_id, coroutine):
    token = context.set({"stage": stage, "id": item_id})
    try:
        return await coroutine
    finally:
        context.reset(token)


# Hand-written diagnostic inputs are kept separate from generated training pairs.
FIXED = [
    ("boundary_short", "daily_chat", "真。让我感到惊讶。", "真让我感到惊讶。"),
    ("boundary_filler", "daily_chat", "嗯真。让我感到惊讶。", "真让我感到惊讶。"),
    ("normal_sentences", "daily_chat", "我到了。你在哪里？", "我到了。你在哪里？"),
    ("normal_short_sentences", "daily_chat", "真好。让我看看。", "真好。让我看看。"),
    ("english_pytorch", "vibe_coding", "嗯，我想用那个PyTorch来训练。", "我想用PyTorch来训练。"),
    ("english_fastapi", "vibe_coding", "这个FastAPI接口，接口有点慢。", "这个FastAPI接口有点慢。"),
    ("english_models", "vibe_coding", "嗯，用RTX 3090跑GPT-4o的客户端测试，测试一下。", None),
    ("english_identifier", "vibe_coding", "那个，把user_id这个字段，字段保留下来。", None),
    ("numeric_rules", "daily_chat", "嗯，预算是三百元，那个，明天下午三点开会。", None),
    ("numeric_correction", "daily_chat", "预算是三百元，不对，预算改成五百元。", None),
    ("single", "daily_chat", "明天去北京，不对，明天去上海。", "明天去上海。"),
    ("rollback", "daily_chat", "我想用红色，改成蓝色，还是用红色吧。", None),
    ("multiple", "daily_chat", "周一开会，不对周二，最后定周三开会。", None),
    ("spelling_cue", "explanation", "收件人是钟宁，钟表的钟，宁静的宁。", "收件人是钟宁。"),
    ("negation_condition", "daily_chat", "嗯，如果明天下雨，就不要取消室内会议。", "如果明天下雨，就不要取消室内会议。"),
    ("meaningful_repeat", "daily_chat", "这件事很重要，真的很重要。", "这件事很重要，真的很重要。"),
]


async def main():
    started = time.monotonic()
    random.seed(20261017)
    client = AuditedClient(base_url=settings.VLLM_BASE_URL, model="refiner-teacher",
                           max_concurrent=2, timeout=600, max_retries=1)
    oral = OralTextGenerator(client, rng=random.Random(20261017))
    clean = CleanTextGenerator(client)
    clean.qc.llm = client
    seedgen = SeedGenerator(client)
    # A pilot needs enough seeds for eight examples, not a full production pool.
    explanation_config = seedgen.loader._seed_config["seed_generation"]["explanation"]
    explanation_config["user_template"] = explanation_config["user_template"].replace("40 distinct", "8 distinct")
    simulator = ASRSimulator(client, mode="llm_sim")
    clean_attempts = []
    fixed_results = []

    async def make_clean(scene, item, kind):
        item = {**item, "meta": {**item.get("meta", {}), "pilot_kind": kind}}
        result = await tagged("clean", item["id"], clean.generate_one(scene, item))
        calls = [entry for entry in CALLS if entry.get("stage") == "clean" and entry.get("id") == item["id"]]
        raw_target = None
        if calls:
            try:
                raw_target = LLMClient._parse_json(calls[-1].get("response", "")).get("clean_text")
            except (ValueError, AttributeError):
                pass
        normalized, numeric_meta = normalize_source(item["oral_text"])
        attempt = {"id": item["id"], "scene": scene, "pilot_kind": kind,
                   "source": item["oral_text"], "normalized_source": normalized,
                   "raw_teacher_target": raw_target, "accepted": result is not None,
                   "clean": result, "numeric_meta": numeric_meta}
        clean_attempts.append(attempt)
        append_jsonl(OUT / "clean_attempts.jsonl", attempt)
        print(f"CLEAN {item['id']} accepted={result is not None}", flush=True)
        return result

    async def diagnostic(entry):
        name, scene, text, expected = entry
        labels = ["boundary_punctuation"] if name.startswith("boundary_") else []
        item = {"id": f"fixed_{name}", "scene": scene, "language": "zh",
                "oral_text": text, "speech_phenomena": labels}
        result = await make_clean(scene, item, "diagnostic")
        row = {"id": item["id"], "source": text, "expected_exact": expected,
               "accepted": result is not None,
               "target": result.get("clean_text") if result else None,
               "expected_exact_match": result["clean_text"] == expected if result and expected else None}
        fixed_results.append(row)
        if result:
            records, _ = assemble._assemble_from_clean(scene, [result])
            for record in records:
                record["meta"]["pilot_kind"] = "diagnostic"
            return records
        return []

    fixed_path = OUT / "fixed_checks.json"
    if fixed_path.exists():
        print("Resuming completed real-teacher fixed diagnostics...", flush=True)
        fixed_results.extend(json.loads(fixed_path.read_text(encoding="utf-8")))
        prior_attempts = [json.loads(line) for line in (OUT / "clean_attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        clean_attempts.extend(row for row in prior_attempts if row["pilot_kind"] == "diagnostic")
        diagnostics = []
        for row in clean_attempts:
            if row["clean"]:
                records, _ = assemble._assemble_from_clean(row["scene"], [row["clean"]])
                diagnostics.extend(records)
        CALLS.extend(json.loads(line) for line in (OUT / "api_calls.jsonl").read_text(encoding="utf-8").splitlines())
    else:
        print("Running 16 fixed diagnostics with the real teacher...", flush=True)
        diagnostics = [record for group in await asyncio.gather(*(diagnostic(x) for x in FIXED)) for record in group]
        dump_json(fixed_path, fixed_results)

    scenes = ("daily_chat", "vibe_coding", "explanation")
    print("Generating real seed pools for three scenes...", flush=True)
    await asyncio.gather(*(tagged("seed", scene, seedgen.generate_scene_seeds(scene)) for scene in scenes))
    for scene in scenes:
        oral.sampler._load(scene)  # Fail early if the model returned an unusable seed pool.

    print("Generating 24 Oral examples with default random controls...", flush=True)
    jobs = [(scene, index) for scene in scenes for index in range(1, 9)]
    oral_items = await asyncio.gather(*(tagged("oral", f"{s}_{i}", oral.generate_one(s, i)) for s, i in jobs))
    generated = []
    for (scene, index), item in zip(jobs, oral_items):
        if item:
            item["id"] = f"auto_{scene}_{index:02d}"
            item["meta"] = {"pilot_kind": "generated"}
            append_jsonl(OUT / "oral_generated.jsonl", item)
            generated.append((scene, item))
    cleaned = await asyncio.gather(*(make_clean(scene, item, "generated") for scene, item in generated))
    clean_items = [(scene, item) for (scene, _), item in zip(generated, cleaned) if item]
    clean_items = [(scene, item) for scene, item in clean_items
                   if (await clean.qc.validate_keywords_one(item))[0]]
    asr_items = await asyncio.gather(*(tagged("asr", item["id"], simulator.simulate_one(scene, item))
                                      for scene, item in clean_items))
    groups = defaultdict(list)
    for (scene, _), item in zip(clean_items, asr_items):
        if item:
            append_jsonl(OUT / "asr_generated.jsonl", item)
            groups[scene].append(item)
    assembled = []
    assembly_stats = {}
    for scene, items in groups.items():
        records, stats = assemble._assemble_from_asr(scene, items)
        assembled.extend(records)
        assembly_stats[scene] = stats

    print(f"Semantic QC: generated={len(assembled)}, diagnostic={len(diagnostics)}", flush=True)
    sem = asyncio.Semaphore(2)
    verdicts = await asyncio.gather(*(tagged("semantic_qc", item["id"], finalize.qc_one(item, client, sem))
                                      for item in assembled + diagnostics))
    accepted = []
    rejects = []
    qc_rows = []
    for item, passed, reason in verdicts:
        row = {"id": item["id"], "passed": bool(passed), "reason": reason}
        qc_rows.append(row)
        append_jsonl(OUT / "semantic_qc.jsonl", row)
        if passed and item["meta"]["pilot_kind"] == "generated":
            accepted.append(item)
        elif not passed:
            rejects.append({**item, "qc_reason": reason})
    unique = deduplicate(accepted, threshold=settings.DEDUP_THRESHOLD)
    for record in unique:
        append_jsonl(OUT / "final" / "train.jsonl", record)
    for record in rejects:
        append_jsonl(OUT / "rejected_semantic.jsonl", record)
    sft = [converted for item in unique if (converted := exporter.convert_one(item))]
    dump_json(OUT / "final" / "pilot_sft.json", sft)
    raw_checks = [row for row in clean_attempts if row["raw_teacher_target"] is not None]
    summary = {
        "teacher_model_path": MODEL, "service_model": "refiner-teacher",
        "base_url": settings.VLLM_BASE_URL, "gpu": 0, "max_model_len": 8192,
        "request_timeout_seconds": 600, "seed_pools_reused_from_v1": True,
        "elapsed_seconds_scope": "this process, including fresh fixed diagnostics; seed pools reused from v1",
        "comparison_baseline": "results/refiner_teacher_pilot_20261017",
        "default_sampling": {"passthrough": settings.PASSTHROUGH_RATIO,
                             "boundary_given_non_passthrough": settings.BOUNDARY_PUNCTUATION_RATIO,
                             "asr_simulation": settings.ASR_SIMULATION_RATIO},
        "scenes_tested": list(scenes), "scenes_not_tested": [s for s in settings.SCENE_DISTRIBUTION if s not in scenes],
        "generated_oral": len(generated), "generated_clean_accepted_before_keyword_qc": sum(x is not None for x in cleaned),
        "generated_clean_after_keyword_qc": len(clean_items), "generated_asr_accepted": sum(bool(x) for x in asr_items),
        "generated_assembled": len(assembled), "generated_semantic_qc_pass": len(accepted),
        "generated_unique_final": len(unique), "generated_sft_exported": len(sft),
        "generated_phenomena": dict(Counter(p for _, item in generated for p in item.get("speech_phenomena", []))),
        "generated_fragments": sum(bool(item.get("meta", {}).get("is_fragment")) for item in asr_items if item),
        "assembly_stats": assembly_stats,
        "fixed_diagnostics": len(FIXED), "fixed_clean_accepted": sum(row["accepted"] for row in fixed_results),
        "fixed_with_exact_reference": sum(row["expected_exact"] is not None for row in fixed_results),
        "fixed_exact_reference_matches": sum(row["expected_exact_match"] is True for row in fixed_results),
        "fixed_semantic_qc_pass": sum(passed for item, passed, _ in verdicts if item["meta"]["pilot_kind"] == "diagnostic"),
        "raw_teacher_clean_responses": len(raw_checks),
        "raw_teacher_rejected_by_numeric_entity_policy": sum(lock_training_target(row["normalized_source"], row["raw_teacher_target"]) is None for row in raw_checks),
        "raw_teacher_repaired_by_numeric_policy": sum((locked := lock_training_target(row["normalized_source"], row["raw_teacher_target"])) is not None and locked != row["raw_teacher_target"] for row in raw_checks),
        "api_call_count": len(CALLS), "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": ["Small pilot, not a student-model benchmark", "Same teacher generates labels and judges semantics; manual review still required",
                        "Fixed diagnostic inputs excluded from generated training pairs", "No training/validation split on this small pilot"],
    }
    dump_json(OUT / "summary.json", summary)
    by_id = {row["id"]: row for row in qc_rows}
    report = ["# 教师模型真实生成小批检查", "", "统计见 summary.json；固定检查句不混入自动生成训练对。", ""]
    for attempt in clean_attempts:
        record_id = attempt["id"]
        report.extend([f"## {record_id}", "", f"- 原始输入：{attempt['source']}",
                       f"- 规则后输入：{attempt['normalized_source']}",
                       f"- 教师原始标签：{attempt['raw_teacher_target']}",
                       f"- 配对校验：{'通过' if attempt['accepted'] else '拒收'}"])
        if attempt["clean"]:
            report.append(f"- 校验后标签：{attempt['clean']['clean_text']}")
        if record_id in by_id:
            report.append(f"- 语义 QC：{by_id[record_id]}")
        report.append("")
    report.extend(["# 最终自动生成训练对", ""])
    for record in unique:
        report.extend([f"## {record['id']}", "", f"输入：{record['input']}", "",
                       f"目标：{record['output']['refined_text']}", ""])
    (OUT / "pairs_review.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    await client.client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(main())
