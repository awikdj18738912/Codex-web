#!/usr/bin/env python3
"""Run numeric gold cases through the current Web final-text refinement function."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system.protection import EntityProtector
from system.web_app import create_app

DEFAULT_CASES = PROJECT_ROOT / "experiments" / "fixtures" / "numeric_refinement_gold.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "numeric_refinement_gold_current_web.json"
DEFAULT_MODEL = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    app = create_app(
        args.model.expanduser().resolve(),
        args.device,
        "http://127.0.0.1:8766",
        "Chinese",
        args.max_new_tokens,
        None,
        None,
        "auto",
        "tri_state",
        True,
        False,
        "current",
    )
    route = next(route for route in app.routes if getattr(route, "path", None) == "/ws/stream")
    closure = dict(zip(route.endpoint.__code__.co_freevars, route.endpoint.__closure__))
    refine_update = closure["refine_update"].cell_contents

    rows = []
    category_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "exact_matches": 0}
    )
    for index, case in enumerate(cases, 1):
        protector = EntityProtector(
            (), enable_rule_protection=True, protect_numeric_spans=False
        )
        result = refine_update(
            case["input"],
            "Chinese",
            True,
            protector,
            None,
            None,
            confidence_metadata={},
        )
        actual = result["clean_text"]
        exact = actual == case["expected"]
        category_summary[case["category"]]["total"] += 1
        category_summary[case["category"]]["exact_matches"] += int(exact)
        rows.append(
            {
                **case,
                "actual": actual,
                "exact_match": exact,
                "refiner_executed": result["refiner_executed"],
                "refiner_accepted": result["refiner_accepted"],
                "refiner_reject_reasons": result["refiner_reject_reasons"],
                "refinement_gate_decisions": result["refinement_gate_decisions"],
                "numeric_normalizations": result["numeric_normalizations"],
                "numeric_fallbacks": result["numeric_fallbacks"],
                "safe_numeric_repairs": result["safe_numeric_repairs"],
                "placeholder_retry_count": result["placeholder_retry_count"],
            }
        )
        print(f"Current Web refinement: {index}/{len(cases)}", flush=True)

    report = {
        "scope": "web_refine_update_final_text_offline",
        "mode": {
            "refinement_gate_mode": "tri_state",
            "numeric_normalization": False,
            "rule_protection": True,
            "protect_numeric_spans": False,
            "entity_db": None,
            "asr_confidence": None,
        },
        "cases": str(args.cases.resolve()),
        "model": str(args.model.resolve()),
        "total": len(rows),
        "exact_matches": sum(row["exact_match"] for row in rows),
        "refiner_executed": sum(row["refiner_executed"] for row in rows),
        "refiner_rejected": sum(not row["refiner_accepted"] for row in rows),
        "by_category": dict(sorted(category_summary.items())),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"exact: {report['exact_matches']}/{len(rows)}, "
        f"refiner calls: {report['refiner_executed']}, "
        f"rejected: {report['refiner_rejected']}"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
