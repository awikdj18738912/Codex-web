#!/usr/bin/env python3
"""Prototype source-anchored numeric protection and zh-itn evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from postprocess_asr import ModelConfig, TransformersPostprocessor
from test_refiner_zh_itn import _find_itn_cli, _run_itn
from system.protection import EntityProtector
from system.refinement_guard import reject_reasons
from system.refinement_protocol import apply_structured_patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = PROJECT_ROOT / "experiments" / "fixtures" / "numeric_refinement_gold.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "numeric_refinement_gold_source_patch.json"
DEFAULT_MODEL = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")

PATCH_PROMPT = (
    "你是中文 ASR 文本纠错助手。只做最小的非数字编辑：删除明确的口头重复和语气词，"
    "修正普通错字，整理必要标点。不要改写数字、日期、时刻、金额、编号、量词或版本号；"
    "带数字的成语、专名和歧义表达必须逐字保留。遇到明确的数字自我修正（如‘十二，不对，十三’）"
    "可以删除被否定的旧说法，但必须保留最后明确确认的数字。不能判断时保留原文。"
    "只返回一个 JSON 对象。无需修改返回"
    '{"action":"keep","source":"","target":"","reason":"no_change"}。'
    "需要修改时返回 {\"action\":\"replace\",\"source\":\"输入中的唯一连续原文\","
    "\"target\":\"只替换该片段后的文本\",\"reason\":\"disfluency、repetition、"
    "self_correction、local_typo 或 punctuation 之一\"}。source 必须逐字且只出现一次。"
)
_JSON_START = re.compile(r"^\s*\{.*\}\s*$", re.DOTALL)
_CORRECTION_CUES = ("不对", "应该是", "哦不", "我是说", "改成")


def protect_patch(source: str, response: str, protector: EntityProtector):
    """Accept a local patch only when it leaves protected source spans intact."""

    if not _JSON_START.fullmatch(response):
        return source, False, "response_not_structured_json", None
    candidate, payload, issue = apply_structured_patch(source, response)
    if issue or not isinstance(payload, dict):
        return source, False, issue or "invalid_patch", payload
    if payload.get("action") == "keep":
        return source, True, "keep", payload

    patch_source = payload.get("source")
    patch_target = payload.get("target")
    reason = str(payload.get("reason", "")).strip().lower()
    if not isinstance(patch_source, str) or not isinstance(patch_target, str):
        return source, False, "missing_patch_text", payload
    patch_start = source.index(patch_source)
    patch_end = patch_start + len(patch_source)
    overlapping = [
        span for span in protector.protect(source).spans
        if span.start < patch_end and span.end > patch_start
    ]
    if not overlapping:
        return candidate, True, "non_numeric_patch", payload

    correction = (
        any(cue in patch_source for cue in _CORRECTION_CUES)
        and "self_correction" in reason
        and not reject_reasons(source, candidate)
    )
    if correction:
        return candidate, True, "validated_numeric_self_correction", payload

    cursor = -1
    for span in overlapping:
        # A patch cannot split a protected phrase at either edge.
        if span.start < patch_start or span.end > patch_end:
            return source, False, f"patch_cuts_protected_{span.entity_type.lower()}_span", payload
        relative_start = span.start - patch_start
        relative_end = span.end - patch_start
        literal = patch_source[relative_start:relative_end]
        found = patch_target.find(literal, cursor + 1)
        if found < 0 or patch_target.find(literal, found + len(literal)) >= 0:
            return source, False, f"protected_{span.entity_type.lower()}_span_changed", payload
        cursor = found
    return candidate, True, "protected_spans_preserved", payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--itn-cli", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        parser.error("case file is empty")
    model_path = args.model.expanduser().resolve()
    processor = TransformersPostprocessor(
        ModelConfig(
            model_path=str(model_path),
            device_map="auto",
            dtype="auto",
            trust_remote_code=False,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
    )
    sources = [case["input"] for case in cases]
    model_responses: list[str] = []
    model_errors: list[str | None] = []
    for start in range(0, len(sources), args.batch_size):
        batch = sources[start : start + args.batch_size]
        try:
            responses, _latency = processor.generate(batch, system_prompt=PATCH_PROMPT)
            model_responses.extend(responses)
            model_errors.extend([None] * len(responses))
        except Exception as error:
            model_responses.extend(batch)
            model_errors.extend([f"{type(error).__name__}: {error}"] * len(batch))
        print(f"Refiner: {min(start + len(batch), len(sources))}/{len(sources)}", flush=True)

    protector = EntityProtector(
        (), enable_rule_protection=True, protect_numeric_spans=True
    )
    accepted_texts = []
    accepted_rows = []
    for source, response, error in zip(sources, model_responses, model_errors):
        if error:
            refined, accepted, reason, payload = source, False, "refiner_error", None
        else:
            refined, accepted, reason, payload = protect_patch(source, response, protector)
        accepted_texts.append(refined)
        accepted_rows.append(
            {
                "refiner_response": response,
                "refiner_error": error,
                "patch_accepted": accepted,
                "patch_decision": reason,
                "patch": payload,
                "refined_text": refined,
            }
        )

    itn = _run_itn(_find_itn_cli(args.itn_cli), accepted_texts)
    rows = []
    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "exact_matches": 0}
    )
    for case, patch_result, itn_result in zip(cases, accepted_rows, itn):
        actual = itn_result["output"]
        exact = actual == case["expected"]
        category = case["category"]
        summary[category]["total"] += 1
        summary[category]["exact_matches"] += int(exact)
        rows.append(
            {
                **case,
                **patch_result,
                "final_output": actual,
                "itn_status": itn_result["status"],
                "itn_mappings": itn_result["mappings"],
                "exact_match": exact,
            }
        )
    report = {
        "scope": "experimental_structured_source_patch_then_zh_itn",
        "numeric_protection": "source spans validated on local patches; no numeric placeholders",
        "idiom_protection": "EntityProtector rule spans",
        "cases": str(args.cases.resolve()),
        "itn_cli": str(_find_itn_cli(args.itn_cli)),
        "model": str(model_path),
        "total": len(rows),
        "exact_matches": sum(row["exact_match"] for row in rows),
        "patches_accepted": sum(row["patch_accepted"] for row in rows),
        "patches_rejected": sum(not row["patch_accepted"] for row in rows),
        "by_category": dict(sorted(summary.items())),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"exact: {report['exact_matches']}/{len(rows)}, "
        f"patch accepted: {report['patches_accepted']}, rejected: {report['patches_rejected']}"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
