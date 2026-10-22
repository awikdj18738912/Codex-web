#!/usr/bin/env python3
"""Prepare a fixed 200-case Qwen3.8 teacher comparison set."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "teacher_model_comparison_1000_20261022"
OUTPUT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = SOURCE_DIR / "sample_1000.jsonl"
PREDICTIONS_PATH = SOURCE_DIR / "predictions_qwen3_30b.jsonl"
PROVENANCE_PATH = SOURCE_DIR / "generation_provenance.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {path}:{line_no}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(value)
    return records


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    for path in (SAMPLE_PATH, PREDICTIONS_PATH, PROVENANCE_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_records = read_jsonl(SAMPLE_PATH)
    if len(source_records) < 200:
        raise ValueError(f"Expected at least 200 source records, got {len(source_records)}")
    predictions = read_jsonl(PREDICTIONS_PATH)
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    prediction_by_id: dict[str, dict[str, Any]] = {}
    for row_no, prediction in enumerate(predictions, 1):
        record_id = prediction.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Prediction row {row_no} has no valid id")
        if record_id in prediction_by_id:
            raise ValueError(f"Duplicate prediction id: {record_id}")
        prediction_by_id[record_id] = prediction

    selected = source_records[:200]
    sample_ids: set[str] = set()
    output_records: list[dict[str, Any]] = []
    baseline_model_aliases: set[str] = set()
    for row_no, source in enumerate(selected, 1):
        record_id = source.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Source row {row_no} has no valid id")
        if record_id in sample_ids:
            raise ValueError(f"Duplicate source id among first 200: {record_id}")
        sample_ids.add(record_id)

        prediction = prediction_by_id.get(record_id)
        if prediction is None:
            raise ValueError(f"Missing baseline prediction for id={record_id}")
        if prediction.get("input") != source.get("input"):
            raise ValueError(
                f"Baseline input mismatch for id={record_id}; refusing id-only match"
            )
        if prediction.get("error") not in (None, ""):
            raise ValueError(
                f"Baseline error for id={record_id}: {prediction.get('error')}"
            )
        baseline_target = prediction.get("clean_text")
        if not isinstance(baseline_target, str) or not baseline_target.strip():
            raise ValueError(f"Missing/empty baseline clean_text for id={record_id}")
        model = prediction.get("teacher_model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Missing baseline teacher_model for id={record_id}")
        baseline_model_aliases.add(model)

        for field in ("case_no", "scene", "input"):
            if field not in source:
                raise ValueError(f"Source row {row_no} missing {field}")
        output_records.append(
            {
                "case_no": source["case_no"],
                "id": record_id,
                "scene": source["scene"],
                "input": source["input"],
                "baseline_target": baseline_target,
            }
        )

    if len(baseline_model_aliases) != 1:
        raise ValueError(
            f"Expected one baseline model alias, found: {sorted(baseline_model_aliases)}"
        )
    teacher_model = provenance.get("teacher_model")
    if not isinstance(teacher_model, str) or not teacher_model.strip():
        raise ValueError("Provenance is missing teacher_model")

    system_prompt = provenance.get("generation_system_prompt")
    user_template = provenance.get("generation_user_template")
    if not isinstance(system_prompt, str) or not isinstance(user_template, str):
        raise ValueError("Provenance is missing exact system or user prompt template")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_output = OUTPUT_DIR / "sample_200.jsonl"
    sample_bytes = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in output_records
    ).encode("utf-8")
    sample_output.write_bytes(sample_bytes)

    write_json(
        OUTPUT_DIR / "prompt_snapshot.json",
        {"system_prompt": system_prompt, "user_template": user_template},
    )

    scene_counts = Counter(str(row["scene"]) for row in output_records)
    stratum_counts = Counter(str(row.get("stratum")) for row in selected)
    manifest = {
        "sample_count": len(output_records),
        "selection": "First 200 records in sample_1000.jsonl, matched to baseline by id.",
        "source_sample_file": str(SAMPLE_PATH.relative_to(ROOT)),
        "source_sample_sha256": sha256_file(SAMPLE_PATH),
        "baseline_predictions_file": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "baseline_predictions_sha256": sha256_file(PREDICTIONS_PATH),
        "generation_provenance_file": str(PROVENANCE_PATH.relative_to(ROOT)),
        "generation_provenance_sha256": sha256_file(PROVENANCE_PATH),
        "sample_file": str(sample_output.relative_to(ROOT)),
        "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "baseline_model": teacher_model,
        "baseline_prediction_model_alias": next(iter(baseline_model_aliases)),
        "baseline_target_field": "clean_text",
        "category_distribution": {
            "scene": dict(sorted(scene_counts.items())),
            "stratum": dict(sorted(stratum_counts.items())),
        },
        "evaluation_scope": (
            "These 200 cases are diagnostic samples drawn from the training set, "
            "not an independent final test set; results must not be used to estimate "
            "overall accuracy."
        ),
    }
    write_json(OUTPUT_DIR / "sample_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
