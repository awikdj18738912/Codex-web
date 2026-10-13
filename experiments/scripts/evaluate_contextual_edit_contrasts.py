"""Frozen baseline/new model comparison on revised contrasts and old regression."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_location_detector import EditLocationDetector, candidate_spans
from experiments.scripts.train_edit_location_detector import predict, metrics, sha, write, write_rows


def positive_hit(row, candidates):
    positions = {index for candidate in candidates for index in range(candidate["start"], candidate["end"])}
    gaps = {candidate["start"] for candidate in candidates if candidate["start"] == candidate["end"]}
    for edit in row["edits"]:
        alternatives = row.get("offset_alternatives", []) if len(row["edits"]) == 1 else []
        locations = alternatives or [edit]
        found = False
        for location in locations:
            if location["start"] == location["end"]:
                found |= location["start"] in gaps
            else:
                lexical = {index for index in range(location["start"], location["end"])
                           if not row["baseline_text"][index].isspace()
                           and unicodedata.category(row["baseline_text"][index])[0] not in {"P", "Z", "S"}}
                found |= bool((lexical or set(range(location["start"], location["end"]))) & positions)
        if not found:
            return False
    return bool(row["edits"])


def contextual_metrics(rows, predictions, threshold, insertion_supported):
    full = metrics(rows, predictions, threshold, insertion_supported)
    candidate_lists = [candidate_spans(item["chars"], item["gaps"] if insertion_supported else [], threshold)
                       for item in predictions]
    pairs = defaultdict(list)
    families = defaultdict(list)
    scopes = defaultdict(list)
    for row, prediction, candidates in zip(rows, predictions, candidate_lists):
        if row.get("pair_id"):
            pairs[row["pair_id"]].append((row, candidates))
        families[row.get("category", "unknown")].append((row, prediction))
        scopes[row.get("lexical_scope", "legacy")].append((row, prediction))
    eligible = [pair for pair in pairs.values() if len(pair) == 2 and sum(bool(row["edits"]) for row, _ in pair) == 1]
    full["contrast_pairs"] = len(eligible)
    full["both_roles_correct_pairs"] = sum(all(positive_hit(row, candidates) if row["edits"] else not candidates
                                               for row, candidates in pair) for pair in eligible)
    full["by_semantic_family"] = {family: metrics([row for row, _ in values], [item for _, item in values], threshold, insertion_supported)
                                  for family, values in families.items()}
    full["by_surface_scope"] = {scope: metrics([row for row, _ in values], [item for _, item in values], threshold, insertion_supported)
                                for scope, values in scopes.items()}
    return full, candidate_lists


def run(args):
    import torch
    from threadpoolctl import threadpool_limits

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    panels = {"new_validation": args.partitions / "validation.jsonl", "corrected_test_regression": args.partitions / "test.jsonl",
              "old_validation_regression": args.old_partitions / "validation.jsonl", "old_test_regression": args.old_partitions / "test.jsonl"}
    cases = {name: [json.loads(line) for line in path.open()] for name, path in panels.items()}
    models = {"frozen_previous": args.baseline, "context_trained": args.model}
    configuration = dict(panel_sha256={name: sha(path) for name, path in panels.items()},
                         weights_sha256={name: sha(path / "detector.pt") for name, path in models.items()},
                         model_thresholds={name: json.loads((path / "metadata.json").read_text())["threshold"] for name, path in models.items()},
                         thresholds_and_checkpoints_frozen=True, device=args.device,
                         corrected_test_status="revised_consumed_authored_test_not_fresh", natural_audio_gold=0,
                         selection_panel="new_validation_only_used_by_trainer", old_panels_used_for_selection=False)
    write(output / "configuration.json", configuration)
    (output / "executed_code").mkdir()
    (output / "executed_code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    summary = dict(scope="authored_context_contrast_and_consumed_regression_not_audio_acceptance", models={})
    for name, path in models.items():
        detector = EditLocationDetector(path, args.device)
        threshold = detector.metadata["threshold"]
        insertion_supported = detector.metadata["insertion_detection_supported"]
        values = {}
        for panel, rows in cases.items():
            predictions = predict(detector.model, detector.tokenizer, rows, args.device, detector.metadata["max_tokens"])
            result, candidates = contextual_metrics(rows, predictions, threshold, insertion_supported)
            values[panel] = result
            write_rows(output / f"{name}.{panel}.jsonl", [dict(id=row["id"], baseline_text=row["baseline_text"],
                        left_context=row.get("left_context", ""), desired_text=row["desired_text"],
                        pair_id=row.get("pair_id"), category=row.get("category"), lexical_scope=row.get("lexical_scope"),
                        candidates=proposal, edits=row["edits"], meaningful_edit_hit=positive_hit(row, proposal))
                        for row, proposal in zip(rows, candidates)])
        # Actual synchronized single-window GPU API; batch timing is not a
        # proxy for interactive latency. detect() copies scores back to CPU.
        for row in cases["new_validation"][:5]:
            detector.detect(row)
        latencies = []
        for row in cases["new_validation"]:
            start = time.perf_counter()
            result = detector.detect(row)
            latencies.append((time.perf_counter() - start) * 1000)
            if result["status"] == "unsupported" or result["production_skip_allowed"]:
                raise RuntimeError("Unexpected inference contract")
        values["single_window_gpu_latency"] = dict(rows=len(latencies), mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies), p95_ms=sorted(latencies)[int(.95 * (len(latencies) - 1))],
            scope="warm synchronized single-window inference plus tokenization, no ASR/Refiner/file timing")
        summary["models"][name] = values
        del detector
        torch.cuda.empty_cache()
    write(output / "summary.json", summary)
    concise = {model: {panel: {key: result.get(key) for key in ("rows", "true_positive_windows", "needs_review", "false_positive_keep_windows",
                             "keep_windows", "location_precision", "contrast_pairs", "both_roles_correct_pairs")}
                        for panel, result in values.items() if panel != "single_window_gpu_latency"}
               for model, values in summary["models"].items()}
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=Path, default=ROOT / "results/contextual_edit_contrasts_20261013_v3")
    parser.add_argument("--old-partitions", type=Path, default=ROOT / "results/edit_detector_traceable_encoder_20261013")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/edit_detector_traceable_encoder_20261013")
    parser.add_argument("--model", type=Path, default=ROOT / "results/contextual_edit_teacher_20261013")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
