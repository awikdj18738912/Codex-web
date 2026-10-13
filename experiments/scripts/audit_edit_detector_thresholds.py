"""Validation-only operating-point audit; never changes frozen model metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_location_detector import EditLocationDetector
from experiments.scripts.train_edit_location_detector import THRESHOLDS, metrics, predict, sha, write


def recall_constraints(result):
    return (result["window_recall"] is not None and result["window_recall"] >= .95
            and all(item["recall"] >= .95 for item in result["by_task"].values() if item.get("edits")))


def audit_operating_points(rows, predictions, insertion_supported=False):
    """Find the highest feasible score breakpoint by monotone binary search.

    This minimizes KEEP false positives under the stated recall constraints.
    It does not claim the globally best location precision among tied points.
    Scores are still uncalibrated. All search inputs must be validation rows.
    """
    if not any(row["edits"] for row in rows):
        raise ValueError("Recall audit requires positive validation examples")
    events = {0., 1.}
    for item in predictions:
        events.update(1. - values[0] for values in item["chars"])
        if insertion_supported:
            events.update(values[1] for values in item["gaps"])
    thresholds = sorted(events)
    cache = {}

    def evaluate(threshold):
        if threshold not in cache:
            cache[threshold] = metrics(rows, predictions, threshold, insertion_supported)
        return cache[threshold]

    if not recall_constraints(evaluate(0.)):
        raise ValueError("No feasible recall policy even at zero")
    low, high = 0, len(thresholds) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if recall_constraints(evaluate(thresholds[middle])):
            low = middle
        else:
            high = middle - 1
    boundary = thresholds[low]
    coarse = [dict(threshold=value, feasible=recall_constraints(evaluate(value)), metrics=evaluate(value))
              for value in THRESHOLDS]
    return dict(event_count=len(thresholds), boundary_threshold=boundary, boundary_metrics=evaluate(boundary),
                next_event_threshold=thresholds[low + 1] if low + 1 < len(thresholds) else None,
                next_event_metrics=evaluate(thresholds[low + 1]) if low + 1 < len(thresholds) else None,
                fixed_grid=coarse, evaluated_thresholds=len(cache),
                selection="highest validation score breakpoint satisfying >=95% window and per-task meaningful edit recall",
                caveat="validation adaptation only; no probability calibration, no production permission, no test selection")


def run(args):
    import torch
    from threadpoolctl import threadpool_limits

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    with args.validation.open() as stream:
        rows = [json.loads(line) for line in stream]
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("Threshold audit accepts validation only")
    results = {}
    for path in args.models:
        detector = EditLocationDetector(path, args.device)
        predictions = predict(detector.model, detector.tokenizer, rows, args.device, detector.metadata["max_tokens"])
        result = audit_operating_points(rows, predictions, detector.metadata["insertion_detection_supported"])
        result.update(model=str(path.resolve()), weights_sha256=sha(path / "detector.pt"),
                      frozen_threshold=detector.metadata["threshold"],
                      frozen_metrics=metrics(rows, predictions, detector.metadata["threshold"], detector.metadata["insertion_detection_supported"]))
        results[path.name] = result
        write(args.output / (path.name + ".json"), result)
        print(json.dumps(dict(model=path.name, frozen_threshold=result["frozen_threshold"],
              boundary_threshold=result["boundary_threshold"], boundary_metrics=result["boundary_metrics"])), flush=True)
        del detector
        torch.cuda.empty_cache()
    write(args.output / "summary.json", dict(validation_sha256=sha(args.validation), scope="validation_only_threshold_diagnostic",
          runtime_metadata_changed=False, test_evaluated=False, production_eligible=False, models=results))
    (args.output / "executed_code").mkdir()
    (args.output / "executed_code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
