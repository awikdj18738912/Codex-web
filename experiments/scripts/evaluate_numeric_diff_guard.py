#!/usr/bin/env python3
"""Prototype full-text refinement with source-anchored numeric/idiom diff guards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from postprocess_asr import ModelConfig, TransformersPostprocessor
from test_refiner_zh_itn import _find_itn_cli, _run_itn, _strip_key_metadata
from system.protection import EntityProtector
from system.refinement_guard import reject_reasons

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = PROJECT_ROOT / "experiments" / "fixtures" / "numeric_refinement_gold.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "numeric_refinement_gold_diff_guard.json"
DEFAULT_MODEL = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")

NUMERIC_CHARS = "零〇○一二两兩三四五六七八九十百千万亿兆壹贰貳叁參肆伍陆陸柒捌玖拾佰仟幺洞拐勾"
NUMERIC_TOKEN = re.compile(
    rf"(?<![A-Za-z0-9])(?:[0-9０-９]+(?:[.．点:：][0-9０-９]+)*|"
    rf"[{NUMERIC_CHARS}]+(?:[点.．:：][{NUMERIC_CHARS}]+)*)(?:[%％])?"
)
PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：删除明显口癖和重复，修正普通错字，补必要标点，"
    "处理自我修正。数字、日期、时刻、金额、编号、版本及带数字的成语必须保持原文，"
    "不要将中文数字改成阿拉伯数字，也不要改变数字数量、顺序或相邻单位。"
    "遇到明确数字自我修正，可以去掉被否定的旧说法，但必须保留最后明确确认的数值；不确定时保留原文。"
    "不要总结、扩写或解释。只返回精修后的完整文本。"
)
CORRECTION_CUES = ("不对", "应该是", "哦不", "我是说", "改成")


def _protected_spans(source: str, protector: EntityProtector):
    spans = [(span.start, span.end, span.original, span.entity_type) for span in protector.protect(source).spans]
    spans.extend(
        (match.start(), match.end(), match.group(), "NUMERIC_TOKEN")
        for match in NUMERIC_TOKEN.finditer(source)
    )
    return sorted(set(spans), key=lambda span: (span[0], span[1], span[3]))


def _touches_span(start: int, end: int, spans) -> bool:
    if start == end:
        return any(left <= start <= right for left, right, _text, _kind in spans)
    return any(left < end and right > start for left, right, _text, _kind in spans)


def merge_outside_protected_spans(source: str, candidate: str, protector: EntityProtector):
    """Keep Refiner edits outside source-anchored numeric and idiom spans."""

    spans = _protected_spans(source, protector)
    cue_present = any(cue in source for cue in CORRECTION_CUES)
    if cue_present and not reject_reasons(source, candidate):
        idioms = [span for span in spans if span[3] == "IDIOM"]
        if all(candidate.count(text) >= source.count(text) for _a, _b, text, _kind in idioms):
            return candidate, "validated_explicit_correction", 0

    matcher = SequenceMatcher(a=source, b=candidate, autojunk=False)
    pieces = []
    blocked = 0
    for tag, left, right, new_left, new_right in matcher.get_opcodes():
        if tag == "equal":
            pieces.append(source[left:right])
        elif _touches_span(left, right, spans):
            pieces.append(source[left:right])
            blocked += 1
        else:
            pieces.append(candidate[new_left:new_right])
    merged = "".join(pieces)
    if NUMERIC_TOKEN.findall(source) != NUMERIC_TOKEN.findall(merged):
        return source, "numeric_surface_sequence_changed", blocked + 1
    for left, right, text, kind in spans:
        if kind == "IDIOM" and merged.count(text) < source.count(text):
            return source, "idiom_surface_changed", blocked + 1
    reasons = reject_reasons(source, merged)
    if reasons:
        return source, "refinement_guard:" + ",".join(reasons), blocked + 1
    return merged, "accepted_non_numeric_edits", blocked


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
    sources = [case["input"] for case in cases]
    model_path = args.model.expanduser().resolve()
    processor = TransformersPostprocessor(
        ModelConfig(
            model_path=str(model_path), device_map="auto", dtype="auto",
            trust_remote_code=False, max_new_tokens=args.max_new_tokens,
            do_sample=False, temperature=0.0, top_p=1.0,
        )
    )
    candidates: list[str] = []
    model_errors: list[str | None] = []
    for start in range(0, len(sources), args.batch_size):
        batch = sources[start : start + args.batch_size]
        try:
            output, _latency = processor.generate(batch, system_prompt=PROMPT)
            candidates.extend(output)
            model_errors.extend([None] * len(output))
        except Exception as error:
            candidates.extend(batch)
            model_errors.extend([f"{type(error).__name__}: {error}"] * len(batch))
        print(f"Refiner: {min(start + len(batch), len(sources))}/{len(sources)}", flush=True)

    protector = EntityProtector((), enable_rule_protection=True, protect_numeric_spans=True)
    merged_rows = []
    intermediate = []
    for source, candidate, error in zip(sources, candidates, model_errors):
        cleaned, metadata_error = _strip_key_metadata(candidate)
        if error or metadata_error:
            merged, decision, blocked = source, "refiner_or_metadata_error", 0
        else:
            merged, decision, blocked = merge_outside_protected_spans(source, cleaned, protector)
        intermediate.append(merged)
        merged_rows.append(
            {
                "refiner_candidate": candidate,
                "refiner_error": error or metadata_error,
                "merge_decision": decision,
                "blocked_edit_count": blocked,
                "merged_before_itn": merged,
            }
        )

    itn = _run_itn(_find_itn_cli(args.itn_cli), intermediate)
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "exact_matches": 0})
    rows = []
    for case, detail, segment in zip(cases, merged_rows, itn):
        actual = segment["output"]
        exact = actual == case["expected"]
        category = case["category"]
        summary[category]["total"] += 1
        summary[category]["exact_matches"] += int(exact)
        rows.append(
            {
                **case, **detail, "final_output": actual,
                "itn_status": segment["status"], "itn_mappings": segment["mappings"],
                "exact_match": exact,
            }
        )
    report = {
        "scope": "experimental_refiner_diff_guard_then_zh_itn",
        "protection": "source anchored numeric tokens and EntityProtector spans; no placeholders",
        "cases": str(args.cases.resolve()),
        "itn_cli": str(_find_itn_cli(args.itn_cli)),
        "model": str(model_path),
        "total": len(rows),
        "exact_matches": sum(row["exact_match"] for row in rows),
        "merge_decisions": dict(Counter(row["merge_decision"] for row in rows)),
        "patch_blocked_edits": sum(row["blocked_edit_count"] for row in rows),
        "by_category": dict(sorted(summary.items())),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exact: {report['exact_matches']}/{len(rows)}; decisions: {report['merge_decisions']}")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
