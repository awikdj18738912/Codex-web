"""Frozen-model positive-only diagnostic; unknown positions are not negatives."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_location_detector import EditLocationDetector, encode_input, collate, candidate_spans


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in args.data.open()]
    if any(row["training_eligible"] or row["human_audio_gold"] for row in rows):
        raise ValueError("Expected review-only partial annotation candidates")
    random.Random(20261013).shuffle(rows)
    rows = rows[:args.limit]
    # Selection is persisted before opening model weights or predictions.
    write(output / "selected_cases.json", rows)
    models = {"small": args.small_model, "traceable": args.traceable_model}
    configuration = dict(data_sha256=sha(args.data), selected_ids=[row["id"] for row in rows],
        seed=20261013, device=args.device, source_scope="new_public_spoken_text_positive_only_deletion_diagnostic",
        weights_sha256={name: sha(path / "detector.pt") for name, path in models.items()},
        thresholds={name: json.loads((path / "metadata.json").read_text())["threshold"] for name, path in models.items()},
        no_model_or_threshold_tuning=True, no_KEEP_precision_or_false_positive_claim=True,
        no_audio_evaluation=True, no_document_independence_claim=True)
    write(output / "configuration.json", configuration)
    (output / "executed_code").mkdir()
    (output / "executed_code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    summaries = {}
    for name, path in models.items():
        detector = EditLocationDetector(path, args.device)
        counts, predictions = Counter(), []
        supported = []
        for row in rows:
            try:
                supported.append((row, encode_input(detector.tokenizer, row, detector.metadata["max_tokens"])))
            except ValueError as error:
                counts["unsupported_rows"] += 1
                predictions.append(dict(id=row["id"], status="unsupported", error=str(error), requires_fallback=True))
        counts["selected_rows"] = len(rows)
        counts["supported_rows"] = len(supported)
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(supported), 16):
                batch = supported[start:start + 16]
                chars, gaps = detector.model(**{key: value.to(args.device) for key, value in collate(detector.tokenizer, [encoded for _, encoded in batch]).items()})
                for index, (row, encoded) in enumerate(batch):
                    probabilities = chars[index, :len(row["baseline_text"])].softmax(-1).cpu().tolist()
                    candidates = candidate_spans(probabilities, [], detector.metadata["threshold"])
                    positions = {i for candidate in candidates for i in range(candidate["start"], candidate["end"])}
                    positives = {i for i, label in enumerate(row["positive_only_character_labels"]) if label == 1}
                    counts["annotated_positive_positions"] += len(positives)
                    counts["matched_positive_positions"] += len(positives & positions)
                    counts["proposed_positions"] += len(positions)
                    counts["unlabelled_proposed_positions"] += len(positions - positives)
                    counts["rows_with_any_candidate"] += bool(candidates)
                    located = 0
                    for edit in row["edits"]:
                        relevant = {i for i in range(edit["start"], edit["end"])
                                    if not row["baseline_text"][i].isspace()
                                    and unicodedata.category(row["baseline_text"][i])[0] not in {"P", "Z", "S"}}
                        relevant = relevant or set(range(edit["start"], edit["end"]))
                        hit = bool(relevant & positions)
                        counts["author_deletions"] += 1
                        counts["located_author_deletions"] += hit
                        located += hit
                    predictions.append(dict(id=row["id"], source_line=row["source_line"], baseline_text=row["baseline_text"],
                        author_target=row["prepared_target"], author_edits=row["edits"], candidates=candidates,
                        located_author_deletions=located, status="scored", production_skip_allowed=False))
        elapsed = time.perf_counter() - started
        summaries[name] = dict(counts, author_positive_position_recall=counts["matched_positive_positions"] / counts["annotated_positive_positions"],
            author_deletion_hit_rate=counts["located_author_deletions"] / counts["author_deletions"],
            batch_forward_and_postprocess_seconds=elapsed,
            note="Unlabelled proposed positions are unknown, not proven false positives; target is not project gold")
        (output / f"{name}_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        del detector
        torch.cuda.empty_cache()
    summary = dict(models=summaries, rows=len(rows), label_scope="author standardization positive-only deletion proposals",
        evaluation_status="new_source_partial_diagnostic_consumed_after_this_run", production_eligible=False,
        limitations=["Author deletion intent may not match faithful ASR cleanup", "No negative/gold KEEP labels",
                     "No paired audio or verified document/speaker split", "No overall precision, accuracy or speed claim"])
    write(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/public_edit_candidates_20261013/bounded_deletion_review.jsonl")
    parser.add_argument("--small-model", type=Path, default=ROOT / "results/edit_location_detector_pretrain_20261013_v2")
    parser.add_argument("--traceable-model", type=Path, default=ROOT / "results/edit_detector_traceable_encoder_20261013")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    run(args)
