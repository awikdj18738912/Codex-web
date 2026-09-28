#!/usr/bin/env python3
"""Evaluate the numeric gold set with zh-itn alone or the Refiner experiment path."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from postprocess_asr import ModelConfig, TransformersPostprocessor
from test_refiner_zh_itn import (
    DEFAULT_MODEL,
    _find_itn_cli,
    _generate_batches,
    _mask_numeric_spans,
    _restore_markers,
    _run_itn,
    _strip_key_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = PROJECT_ROOT / "experiments" / "fixtures" / "numeric_refinement_gold.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "numeric_refinement_gold_evaluation.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--itn-cli", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--with-refiner", action="store_true")
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
    itn_cli = _find_itn_cli(args.itn_cli)
    sources = [case["input"] for case in cases]
    rule_only = _run_itn(itn_cli, sources)
    masked = sources
    model_outputs = sources
    restored = sources
    model_errors: list[str | None] = [None] * len(cases)
    marker_errors: list[str | None] = [None] * len(cases)

    if args.with_refiner:
        protected = [_mask_numeric_spans(text) for text in sources]
        masked = [text for text, _spans in protected]
        spans = [numeric_spans for _text, numeric_spans in protected]
        processor = TransformersPostprocessor(
            ModelConfig(
                model_path=str(args.model.expanduser().resolve()),
                device_map="auto",
                dtype="auto",
                trust_remote_code=False,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        )
        model_outputs = []
        model_errors = []
        for start in range(0, len(masked), args.batch_size):
            batch_outputs, batch_errors = _generate_batches(
                processor, masked[start : start + args.batch_size], args.batch_size
            )
            model_outputs.extend(batch_outputs)
            model_errors.extend(batch_errors)
            print(f"Refiner: {min(start + args.batch_size, len(masked))}/{len(masked)}", flush=True)
        restored = []
        marker_errors = []
        for source, masked_text, numeric_spans, candidate in zip(
            sources, masked, spans, model_outputs
        ):
            cleaned, metadata_error = _strip_key_metadata(candidate)
            recovered, marker_error = _restore_markers(
                cleaned, masked_text, numeric_spans
            )
            issue = metadata_error or marker_error
            marker_errors.append(issue)
            restored.append(source if issue else recovered)

    final = _run_itn(itn_cli, restored)
    rows = []
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "rule_only_exact": 0, "combined_exact": 0}
    )
    for case, baseline, masked_text, candidate, recovered, output, model_error, marker_error in zip(
        cases, rule_only, masked, model_outputs, restored, final, model_errors, marker_errors
    ):
        category = case["category"]
        rule_match = baseline["output"] == case["expected"]
        final_match = output["output"] == case["expected"]
        by_category[category]["total"] += 1
        by_category[category]["rule_only_exact"] += int(rule_match)
        by_category[category]["combined_exact"] += int(final_match)
        rows.append(
            {
                **case,
                "rule_only_output": baseline["output"],
                "masked_input": masked_text,
                "refiner_output": candidate if args.with_refiner else None,
                "restored_text": recovered,
                "final_output": output["output"],
                "rule_only_exact": rule_match,
                "combined_exact": final_match,
                "refiner_error": model_error,
                "marker_error": marker_error,
            }
        )
    report = {
        "scope": "refiner_masked_then_zh_itn" if args.with_refiner else "zh_itn_only",
        "cases": str(args.cases.resolve()),
        "itn_cli": str(itn_cli),
        "model": str(args.model.resolve()) if args.with_refiner else None,
        "total": len(rows),
        "rule_only_exact": sum(row["rule_only_exact"] for row in rows),
        "combined_exact": sum(row["combined_exact"] for row in rows),
        "refiner_errors": sum(row["refiner_error"] is not None for row in rows),
        "marker_errors": sum(row["marker_error"] is not None for row in rows),
        "by_category": dict(sorted(by_category.items())),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"rule only: {report['rule_only_exact']}/{len(rows)}, "
        f"combined: {report['combined_exact']}/{len(rows)}, "
        f"marker errors: {report['marker_errors']}"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
