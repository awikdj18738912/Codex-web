# -*- coding: utf-8 -*-
"""Finalize generated pairs with semantic QC, deduplication, and statistics."""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    DEDUP_THRESHOLD,
    FINAL_DIR,
    LLM_MAX_CONCURRENT,
    LLM_MAX_TOKENS_QC,
    PROCESSED_DIR,
    SCENE_DISTRIBUTION,
    TOTAL_SAMPLES,
)
from configs.scenes import get_enabled_scenes
from src.pipeline.dedup import compute_distribution, deduplicate
from src.pipeline.training_quality import AUDIT_SYSTEM, audit_pair
from src.pipeline.semantic_verdict import PARSER_VERSION, parse_semantic_verdict
from src.services.llm_client import AsyncLLMClient
from src.utils.io_utils import read_jsonl, setup_logging, write_jsonl

from tqdm import tqdm

logger = logging.getLogger("finalize")

SEMANTIC_QC_SYSTEM = AUDIT_SYSTEM
UPSTREAM_SEMANTIC_QC_SYSTEM = """Judge whether a Refiner target exactly captures the speaker's
final intended meaning. Resolve single, rollback, delayed, and multi-stage corrections using the
final value. Permit filler/repetition removal, external numeric normalization, punctuation, and
formatting. Reject omitted supported details, hallucinated details, summaries, over-specific
rewrites, and empty content. Already clean passthrough input is valid when unchanged. Return strict
JSON only: {\"pass\":true,\"reason\":\"brief reason\"}."""
SPOKEN_SOURCE_QC_SYSTEM = """你审核纯口语 ASR 训练样本的源文本。只判断 source 本身是否适合这个数据集，不看 target，不改写或删除文字。
合格文本应像该场景中说话人实际说出的一段连贯表达。普通流畅口语、简短回答、自然改口、停顿词、中英混用及明确口述的代码/公式都可以通过；不要求每句都含口癖。
认真检查三类问题：
1. 编辑标记：术语后用括号补充英文对照（如“困惑度（Perplexity）”）、旁白、舞台指令，不是可确认的口述内容，应拒绝。自然说出的“困惑度，也就是 Perplexity”应通过。不能仅凭括号本身拒绝，口述代码/公式中的括号也可通过。
2. 语言不符：zh 或 zh-en-mix 样本应以中文句法为主，英文术语可自然出现；整段英文应拒绝。
3. 拼接/跑题：多个实体或技术领域若只是用“然后”“我又加了”等词串起来，却没有合理的共同主题或说话目的，应拒绝。比如先讲真菌数据特征提取和流形降维，突然转到量子化学能量函数，再转到无人机安全对抗训练，即使每段各自通顺也不连贯。相反，围绕同一模型解释准确率、召回率和 F1 属于同一主题，应通过。
若不能确认括号内容是实际说出的、括号说明并非可听到的口语表达，应拒绝；对其他不确定情况可通过。返回严格 JSON：pass 必须是 JSON 布尔值，reason 必须用简短中文说明通过或拒绝的具体依据，不得留空。"""
SPOKEN_SOURCE_QC_USER = """场景：{scene}
语言要求：{language}

源文本：
{source}

这是否像该场景中说话人实际说出的、主题连贯的口语文本？逐项判断编辑标记、语言设置、主题连贯性。不能只因文本有技术词就推断它连贯；说明具体证据。请只判断上面的源文本。"""
UPSTREAM_SEMANTIC_QC_PROMPT_VERSION = "spoken-source-plus-meaning-v5"
UPSTREAM_SEMANTIC_QC_USER = """Input transcript:
{input}

Refined transcript:
{refined}

Does the refined text preserve the complete final intended meaning? Return JSON only."""


async def _qc_upstream_teacher(item: dict, inp: str, refined: str,
                               llm: AsyncLLMClient, sem: asyncio.Semaphore):
    """Judge spoken-source validity independently before judging its target."""
    response = None
    source_response = None
    source_verdict = None
    verdict_valid = False
    decision_stage = "source"
    source = str(item.get("oral_text") or inp)
    language = str((item.get("meta") or {}).get("language") or item.get("language") or "unknown")
    # These two source defects are objective for this Chinese oral corpus and
    # should not depend on the teacher's semantic judgment.
    if language in {"zh", "zh-en-mix"} and not re.search(r"[\u3400-\u9fff]", source):
        source_verdict = parse_semantic_verdict(json.dumps({
            "pass": False, "reason": "zh/zh-en-mix 源文本没有中文内容，整段为外语。"
        }, ensure_ascii=False))
    elif re.search(r"[\u3400-\u9fff]{1,12}[（(]\s*[A-Za-z][A-Za-z0-9 ._+/#&-]{1,60}[）)]", source):
        source_verdict = parse_semantic_verdict(json.dumps({
            "pass": False, "reason": "源文本含有中文术语后附括号英文对照，属于无法从文本确认的编者注释，不适合作为纯口语源文本。"
        }, ensure_ascii=False))
    if source_verdict is not None:
        source_response = json.dumps({"pass": source_verdict.passed,
                                      "reason": source_verdict.reason}, ensure_ascii=False)
        response = source_response
        return {**item, "meta": {**(item.get("meta") or {}), "quality_audit": {
            "mode": "upstream-semantic-v1", "consistent": False,
            "reason": "source_quality: " + source_verdict.reason,
            "parser_version": PARSER_VERSION,
            "prompt_version": UPSTREAM_SEMANTIC_QC_PROMPT_VERSION,
            "verdict_valid": True, "raw_response": response,
            "decision_stage": "source_rule",
            "source_audit": {"consistent": False, "verdict_valid": True,
                             "reason": source_verdict.reason, "raw_response": source_response,
                             "validator": "spoken-source-deterministic-v1"},
        }}}, False, "source_quality: " + source_verdict.reason
    async with sem:
        try:
            source_response = await llm.generate(
                system_prompt=SPOKEN_SOURCE_QC_SYSTEM,
                user_prompt=SPOKEN_SOURCE_QC_USER.format(
                    scene=(item.get("meta") or {}).get("scene") or item.get("scene") or "unknown",
                    language=(item.get("meta") or {}).get("language") or item.get("language") or "unknown",
                    source=str(item.get("oral_text") or inp),
                ),
                temperature=0,
                max_tokens=LLM_MAX_TOKENS_QC,
                response_format="json_object",
            )
            source_verdict = parse_semantic_verdict(source_response)
            if source_verdict.valid and source_verdict.passed:
                decision_stage = "meaning"
                response = await llm.generate(
                    system_prompt=UPSTREAM_SEMANTIC_QC_SYSTEM,
                    user_prompt=UPSTREAM_SEMANTIC_QC_USER.format(input=inp, refined=refined),
                    temperature=0.1,
                    max_tokens=LLM_MAX_TOKENS_QC,
                    response_format="json_object",
                )
                verdict = parse_semantic_verdict(response)
                passed, reason, verdict_valid = verdict.passed, verdict.reason, verdict.valid
            else:
                response = source_response
                passed, reason = False, "source_quality: " + source_verdict.reason
                verdict_valid = source_verdict.valid
        except Exception as exc:
            logger.warning("Semantic QC failed: %s", exc)
            passed, reason = False, "QC call failed"
    meta = dict(item.get("meta") or {})
    meta["quality_audit"] = {
        "mode": "upstream-semantic-v1", "consistent": passed, "reason": reason,
        "parser_version": PARSER_VERSION,
        "prompt_version": UPSTREAM_SEMANTIC_QC_PROMPT_VERSION,
        "verdict_valid": verdict_valid,
        "raw_response": response,
        "decision_stage": decision_stage,
        "source_audit": {
            "consistent": source_verdict.passed if source_verdict is not None else False,
            "verdict_valid": source_verdict.valid if source_verdict is not None else False,
            "reason": source_verdict.reason if source_verdict is not None else "QC call failed",
            "raw_response": source_response,
        },
    }
    return {**item, "meta": meta}, passed, reason


async def qc_one(item: dict, llm: AsyncLLMClient, sem: asyncio.Semaphore) -> tuple[dict, bool, str]:
    inp = item.get("input", "")
    refined = item.get("output", {}).get("refined_text", "")
    if not isinstance(inp, str) or not isinstance(refined, str) or not inp.strip() or not refined.strip():
        return item, False, "input or refined_text is empty"
    if len(refined) > len(inp) * 1.8 and len(refined) - len(inp) > 30:
        return item, False, "abnormal expansion"
    if len(inp.strip()) <= 2:
        return item, False, "input too short"
    plan = (item.get("meta") or {}).get("generation_plan") or {}
    expected = plan.get("expected_clean")
    if (item.get("meta") or {}).get("is_fragment"):
        from src.generators.asr_simulator import valid_partial_pair
        reference = (item.get("meta") or {}).get("fragment_reference_clean") or expected
        if not reference or not valid_partial_pair(item.get("oral_text", ""), reference, inp, refined):
            return item, False, "invalid_fragment_alignment"
    if expected and not (item.get("meta") or {}).get("is_fragment"):
        if refined != expected:
            return item, False, "target_disagrees_with_generation_intent"
    if (plan.get("generation_mode") == "upstream-teacher-v1"
            and not (item.get("meta") or {}).get("is_fragment")):
        return await _qc_upstream_teacher(item, inp, refined, llm, sem)
    async with sem:
        passed, reason, evidence = await audit_pair(
            inp, refined, llm,
            recognition_repair=(bool((item.get("meta") or {}).get("is_fragment"))
                                or plan.get("generation_mode") in {
                                    "teacher-natural-v1", "upstream-teacher-v1"}),
            generation_plan=(plan if (
                (plan.get("construction_version") == "synthetic-oral-v1"
                 or plan.get("generation_mode") == "upstream-teacher-v1")
                and not (item.get("meta") or {}).get("is_fragment")
            ) else None),
        )
    meta = dict(item.get("meta") or {})
    meta["quality_audit"] = evidence
    item = {**item, "meta": meta}
    return item, passed, reason


async def run_qc_scene(
    items: list[dict],
    target: int,
    concurrent: int,
    base_url: str | None = None,
) -> tuple[list[dict], list[dict], int]:
    """Check one scene until the requested number of records passes."""
    client_kwargs = {"max_concurrent": concurrent}
    if base_url:
        client_kwargs["base_url"] = base_url
    llm = AsyncLLMClient(**client_kwargs)
    sem = asyncio.Semaphore(concurrent)

    passed, rejected = [], []
    remaining = list(items)
    checked = 0
    pbar = tqdm(total=target, desc="QC", unit="item")

    while len(passed) < target and remaining:
        batch = remaining[:concurrent * 2]
        del remaining[:concurrent * 2]

        tasks = [qc_one(it, llm, sem) for it in batch]
        batch_results = await asyncio.gather(*tasks)
        checked += len(batch_results)

        for item, ok, reason in batch_results:
            if ok:
                passed.append(item)
                pbar.update(1)
                if len(passed) >= target:
                    break
            else:
                item["qc_reason"] = reason
                rejected.append(item)

    pbar.close()
    return passed, rejected, checked


def load_by_scene(processed_dir: Path) -> dict[str, list[dict]]:
    by_scene: dict[str, list[dict]] = {}
    enabled = set(get_enabled_scenes())
    for jsonl_file in sorted(processed_dir.glob("*.jsonl")):
        scene = jsonl_file.stem
        if scene not in enabled:
            logger.info("Ignoring disabled scene file %s", jsonl_file.name)
            continue
        records = read_jsonl(jsonl_file)
        by_scene[scene] = records
        logger.info("Loaded %d records from %s", len(records), jsonl_file.name)
    return by_scene


def validate_distribution(stats: dict) -> list[str]:
    warnings = []
    total = stats["total"]
    if total == 0:
        return warnings
    for scene, target_ratio in SCENE_DISTRIBUTION.items():
        actual_count = stats["scene"].get(scene, 0)
        actual_ratio = actual_count / total
        deviation = abs(actual_ratio - target_ratio)
        if deviation > 0.05:
            warnings.append(
                f"{scene}: target={target_ratio:.2%}, "
                f"actual={actual_ratio:.2%} (dev={deviation:.2%})"
            )
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Finalize, validate, and deduplicate pairs")
    parser.add_argument("--total", type=int, default=TOTAL_SAMPLES, help="target record count")
    parser.add_argument("--threshold", type=float, default=DEDUP_THRESHOLD)
    parser.add_argument("--skip-qc", action="store_true", help="skip semantic QC")
    parser.add_argument("--qc-concurrent", type=int, default=LLM_MAX_CONCURRENT,
                        help="QC concurrency (default: %d)" % LLM_MAX_CONCURRENT)
    parser.add_argument("--base-url", type=str, default=None, help="LLM service URL")
    parser.add_argument("--save-rejects", action="store_true",
                        help="write rejected records to data/final/rejects.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    processed_dir = Path(PROCESSED_DIR)
    if not processed_dir.exists():
        logger.error("Processed dir not found: %s", processed_dir)
        sys.exit(1)

    logger.info("=== Loading scene data ===")
    by_scene = load_by_scene(processed_dir)
    if not by_scene:
        logger.error("No data found in %s", processed_dir)
        sys.exit(1)

    # Compute per-scene targets.
    available_scenes = list(by_scene.keys())
    targets: dict[str, int] = {}
    for scene in available_scenes:
        ratio = SCENE_DISTRIBUTION[scene]
        targets[scene] = max(1, int(args.total * ratio))
        logger.info("  %s: available=%d, target=%d (%.0f%%)",
                    scene, len(by_scene[scene]), targets[scene], ratio * 100)

    # Run semantic QC until each scene reaches its target.
    all_rejects = []
    if args.skip_qc:
        logger.info("=== Skipping Semantic QC ===")
        passed_data = []
        for scene in available_scenes:
            items = by_scene[scene][:targets[scene]]
            passed_data.extend(items)
            logger.info("  %s: took %d/%d (no QC)", scene, len(items), targets[scene])
    else:
        passed_data = []
        for scene in available_scenes:
            target = targets[scene]
            items = by_scene.get(scene, [])
            logger.info("=== QC: %s (need %d, available %d) ===", scene, target, len(items))
            passed, rejected, checked = asyncio.run(
                run_qc_scene(items, target, args.qc_concurrent, args.base_url)
            )
            passed_data.extend(passed)
            all_rejects.extend(rejected)
            shortfall = max(0, target - len(passed))
            logger.info("  %s: passed=%d, rejected=%d, checked=%d, shortfall=%d",
                        scene, len(passed), len(rejected), checked, shortfall)

        if args.save_rejects and all_rejects:
            rejects_path = Path(FINAL_DIR) / "rejects.jsonl"
            rejects_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(all_rejects, rejects_path)
            logger.info("Rejects saved: %s (%d records)", rejects_path, len(all_rejects))

    # Global near-duplicate filtering.
    logger.info("=== Global dedup (threshold=%.2f) ===", args.threshold)
    before = len(passed_data)
    final_data = deduplicate(passed_data, text_key="input", threshold=args.threshold)
    logger.info("Dedup: %d -> %d (removed %d, %.1f%%)",
                before, len(final_data), before - len(final_data),
                (before - len(final_data)) / before * 100 if before else 0)

    # Distribution validation.
    logger.info("=== Distribution validation ===")
    stats = compute_distribution(final_data)
    warnings = validate_distribution(stats)
    for msg in warnings:
        logger.warning("Distribution drift: %s", msg)
    if not warnings:
        logger.info("Distribution check passed.")

    # Write the final training set.
    logger.info("=== Writing final dataset ===")
    final_dir = Path(FINAL_DIR)
    final_dir.mkdir(parents=True, exist_ok=True)

    # IDs and the full observed Oral are required to trace labels and validate
    # fragments again at acceptance/export. SFT export strips this provenance.
    train_data = final_data

    train_path = final_dir / "train.jsonl"
    write_jsonl(train_data, train_path)
    logger.info("Final: %s (%d records)", train_path, len(train_data))

    stats_output = {
        "total_samples": stats["total"],
        "target_samples": args.total,
        "coverage": stats["total"] / args.total if args.total else 0,
        "dedup_threshold": args.threshold,
        "distribution": stats,
        "warnings": warnings,
    }
    stats_path = final_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, ensure_ascii=False, indent=2)
    logger.info("Stats: %s", stats_path)
    logger.info("DONE: %d / %d (%.1f%%)", stats["total"], args.total,
                stats["total"] / args.total * 100 if args.total else 0)


if __name__ == "__main__":
    main()
