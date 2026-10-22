# -*- coding: utf-8 -*-
"""Full structural acceptance audit for finalized JSONL datasets (no LLM calls)."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.training_quality import validate_audit
from src.utils.numeric_policy import (
    NUMERIC_POLICY,
    prepare_fragment_training_pair,
    prepare_training_pair,
)

MAX_ERROR_EXAMPLES = 50


def _count_values(counter: Counter, value: object) -> None:
    if isinstance(value, list):
        for item in value:
            counter[str(item)] += 1
    elif value is not None:
        counter[str(value)] += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(input_path: Path, report_path: Path, sample_path: Path | None,
          sample_size: int, seed: int, min_records: int = 0,
          trust_assembly_metadata: bool = False) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    counts = Counter()
    errors = Counter()
    examples = []
    scenes, phenomena, fragments, origins = (Counter() for _ in range(4))
    ids = set()
    inputs: dict[str, dict[str, object]] = {}
    input_frequency = Counter()
    reservoir: list[dict] = []
    scene_representatives: dict[str, dict] = {}
    phenomenon_representatives: dict[str, dict] = {}

    def fail(code: str, line_number: int, record_id: object = None, detail: str = "") -> None:
        errors[code] += 1
        if len(examples) < MAX_ERROR_EXAMPLES:
            example = {"line": line_number, "error": code}
            if record_id is not None:
                example["id"] = record_id
            if detail:
                example["detail"] = detail[:300]
            examples.append(example)

    with input_path.open("rb") as stream:
        for line_number, raw_bytes in enumerate(stream, 1):
            counts["lines"] += 1
            try:
                raw_line = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                fail("invalid_utf8", line_number, detail=str(exc))
                continue
            if not raw_line.strip():
                fail("blank_or_empty_record", line_number)
                continue
            try:
                item = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                fail("invalid_json", line_number, detail=str(exc))
                continue
            if not isinstance(item, dict):
                fail("record_not_object", line_number)
                continue

            counts["records"] += 1
            record_id = item.get("id")
            valid_id = type(record_id) in (str, int) and (not isinstance(record_id, str) or bool(record_id.strip()))
            if record_id is None or (isinstance(record_id, str) and not record_id.strip()):
                fail("missing_id", line_number)
            elif not valid_id:
                fail("invalid_id_type", line_number, detail=type(record_id).__name__)
            else:
                try:
                    if record_id in ids:
                        fail("duplicate_id", line_number, record_id)
                    else:
                        ids.add(record_id)
                except TypeError:
                    fail("invalid_id_type", line_number, detail=type(record_id).__name__)

            meta = item.get("meta")
            if not isinstance(meta, dict):
                fail("missing_metadata", line_number, record_id)
                meta = {}
            scene = meta.get("scene") or item.get("scene") or "unknown"
            scenes[str(scene)] += 1
            _count_values(phenomena, item.get("speech_phenomena"))
            if meta.get("speech_phenomena") != item.get("speech_phenomena"):
                _count_values(phenomena, meta.get("speech_phenomena"))
            if type(meta.get("is_fragment")) is not bool:
                fail("missing_or_invalid_fragment_flag", line_number, record_id)
            is_fragment = meta.get("is_fragment") is True
            fragments[str(is_fragment).lower()] += 1
            _count_values(origins, meta.get("target_origin", item.get("target_origin")))

            source = item.get("input")
            output = item.get("output")
            target = output.get("refined_text") if isinstance(output, dict) else None
            if not isinstance(source, str) or not source.strip():
                fail("empty_or_invalid_input", line_number, record_id)
                source = ""
            if not isinstance(target, str) or not target.strip():
                fail("empty_or_invalid_output", line_number, record_id)
                target = ""

            if isinstance(source, str) and source:
                input_frequency[source] += 1
                target_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()
                entry = inputs.setdefault(source, {"targets": set(), "first_line": line_number})
                targets_for_input = entry["targets"]
                had_target = bool(targets_for_input)
                already_seen_target = target_hash in targets_for_input
                targets_for_input.add(target_hash)
                if had_target and not already_seen_target:
                    fail("same_input_different_target", line_number, record_id)

            audit_data = meta.get("quality_audit")
            if audit_data is None:
                fail("missing_quality_audit", line_number, record_id)
            elif (isinstance(audit_data, dict)
                    and audit_data.get("mode") == "upstream-semantic-v1"):
                if type(audit_data.get("consistent")) is not bool:
                    fail("invalid_upstream_semantic_verdict", line_number, record_id)
                elif audit_data["consistent"] is not True:
                    fail("upstream_semantic_qc_failed", line_number, record_id,
                         str(audit_data.get("reason", "")))
            else:
                passed, reason = validate_audit(audit_data, source, target)
                if not passed:
                    fail("quality_audit_failed", line_number, record_id, reason)

            plan = meta.get("generation_plan")
            expected = plan.get("expected_clean") if isinstance(plan, dict) else None
            if is_fragment:
                reference = meta.get("fragment_reference_clean") or expected
                if not isinstance(reference, str) or not reference or not isinstance(item.get("oral_text"), str):
                    fail("invalid_fragment_pair", line_number, record_id)
                try:
                    pair = prepare_fragment_training_pair(
                        source, target, item.get("oral_text", ""), reference,
                        plan if isinstance(plan, dict) else None,
                        observed_prefix=meta.get("fragment_observed_prefix"),
                    )
                except Exception as exc:
                    pair = None
                    fail("prepare_training_pair_exception", line_number, record_id, type(exc).__name__)
            else:
                if not isinstance(expected, str) or not expected or target != expected:
                    fail("target_disagrees_with_plan", line_number, record_id)
                meta_proves_prepared_pair = (
                    trust_assembly_metadata
                    and isinstance(plan, dict)
                    and plan.get("generation_mode") == "upstream-teacher-v1"
                    and expected == target
                    and meta.get("numeric_policy") == NUMERIC_POLICY
                    and meta.get("itn_owner") == "rules"
                    and meta.get("english_entity_policy") == "verbatim-v1"
                )
                if meta_proves_prepared_pair:
                    # Stage 4 already ran prepare_training_pair and stored these
                    # provenance fields. Trust that proof instead of rerunning
                    # the native Chinese ITN engine for every final record.
                    pair = (source, target, {})
                else:
                    try:
                        pair = prepare_training_pair(source, target, plan if isinstance(plan, dict) else None)
                    except Exception as exc:
                        pair = None
                        fail("prepare_training_pair_exception", line_number, record_id, type(exc).__name__)
            if pair is None:
                fail("prepare_training_pair_failed", line_number, record_id)
            elif pair[:2] != (source, target):
                fail("stored_pair_not_rule_normalized", line_number, record_id)

            # Reservoir sampling over every parseable record object.
            index = counts["reservoir_seen"]
            counts["reservoir_seen"] += 1
            if sample_size > 0:
                if len(reservoir) < sample_size:
                    reservoir.append(item)
                else:
                    replacement = rng.randrange(index + 1)
                    if replacement < sample_size:
                        reservoir[replacement] = item

            scene_representatives.setdefault(str(scene), item)
            labels = []
            for label_values in (item.get("speech_phenomena"), meta.get("speech_phenomena")):
                if isinstance(label_values, list):
                    labels.extend(label_values)
                elif label_values:
                    labels.append(label_values)
            if not isinstance(labels, list):
                labels = [labels] if labels else []
            for label in labels:
                phenomenon_representatives.setdefault(str(label), item)

    total_records = counts["records"]
    if total_records == 0:
        fail("empty_dataset", 0)
    if total_records < min_records:
        fail("insufficient_records", 0, detail=f"{total_records} < required {min_records}")
    rare_threshold = max(1, int(total_records * 0.01))
    rare_controls = {
        label: representative
        for label, representative in phenomenon_representatives.items()
        if phenomena[label] <= rare_threshold
    }
    sample_records = []
    seen_sample_keys = set()
    for record in reservoir + list(scene_representatives.values()) + list(rare_controls.values()):
        key = record.get("id")
        if type(key) not in (str, int):
            key = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if key not in seen_sample_keys:
            seen_sample_keys.add(key)
            sample_records.append(record)

    duplicate_input_count = sum(1 for value in inputs.values() if len(value["targets"]) > 1)
    exact_input_repeat_occurrences = sum(count - 1 for count in input_frequency.values() if count > 1)

    report = {
        "passed": sum(errors.values()) == 0,
        "scope": "Structural and stored audit evidence checks; no new semantic judgment or student model evaluation.",
        "normalization_check": (
            "trusted-stage4-metadata" if trust_assembly_metadata else "recomputed-native-rules"
        ),
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "counts": {"lines": counts["lines"], "records": total_records,
                   "reservoir_candidates": counts["reservoir_seen"], "minimum_required": min_records},
        "errors": {"total": sum(errors.values()), "by_type": dict(sorted(errors.items())),
                   "examples_first_50": examples},
        "warnings": {"exact_input_repeat_occurrences": exact_input_repeat_occurrences,
                     "inputs_with_different_targets": duplicate_input_count},
        "statistics": {"scene": dict(sorted(scenes.items())),
                       "speech_phenomena": dict(sorted(phenomena.items())),
                       "fragment": dict(sorted(fragments.items())),
                       "target_origin": dict(sorted(origins.items()))},
        "sampling": {"method": "fixed-seed reservoir sampling over parseable JSON object records; then union with first representative per scene and per rare speech_phenomena label (frequency <= max(1, floor(1% of parseable records)))",
                     "seed": seed, "requested_reservoir_size": sample_size,
                     "reservoir_size": len(reservoir), "representative_scene_count": len(scene_representatives),
                     "rare_control_threshold": rare_threshold, "rare_control_labels": sorted(rare_controls),
                     "output_records_including_representatives": len(sample_records)},
    }
    return report, sample_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit finalized JSONL data without changing records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sample-output")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--min-records", type=int, default=0,
                        help="fail acceptance if fewer than this many records are present")
    parser.add_argument("--trust-assembly-metadata", action="store_true",
                        help="trust Stage 4 normalization provenance for upstream-teacher records instead of rerunning native numeric rules")
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error("--sample-size must be nonnegative")
    if args.min_records < 0:
        parser.error("--min-records must be nonnegative")

    input_path = Path(args.input).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    sample_path = Path(args.sample_output).expanduser().resolve() if args.sample_output else None
    output_paths = [report_path] + ([sample_path] if sample_path else [])
    if any(path == input_path for path in output_paths):
        parser.error("input and output paths must be different")
    if sample_path is not None and sample_path == report_path:
        parser.error("--report and --sample-output paths must be different")
    if not input_path.is_file():
        parser.error(f"input file does not exist: {input_path}")

    report, samples = audit(
        input_path, report_path, sample_path, args.sample_size, args.seed,
        args.min_records, args.trust_assembly_metadata,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if sample_path is not None:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.open("w", encoding="utf-8") as stream:
            for item in samples:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return 0 if report["errors"]["total"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
