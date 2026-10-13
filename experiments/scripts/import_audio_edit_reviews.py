"""Validate explicit audio reviews; consumed diagnostics stay out of training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_detection_data import align_edits, attach_review
from experiments.scripts.prepare_audio_edit_review import packet_basis, read_rows, sha, write_jsonl


def validate_review(row, review, directory):
    if review.get("packet_basis_sha256") != packet_basis(row):
        raise ValueError("Stale review packet")
    if review.get("review_kind") != "human_audio_cleaned" or review.get("audio_checked") is not True:
        raise ValueError("Explicit human audio review required")
    if sha(Path(row["audio_path"])) != row["audio_sha256"]:
        raise ValueError("Original review clip changed")
    if sha(directory / row["packaged_audio"]) != row["audio_sha256"]:
        raise ValueError("Packaged review clip changed")
    original = Path(row["audio_provenance"]["original_path"])
    if sha(original) != row["audio_provenance"]["original_sha256"]:
        raise ValueError("Original recording changed")
    normalized = dict(review)
    edits = align_edits(row["baseline_text"], review["desired_text"])["edits"]
    if normalized.get("single_task_for_all_edits") is True and len(normalized["edit_tasks"]) == 1:
        normalized["edit_tasks"] = normalized["edit_tasks"] * len(edits)
    effect = review.get("historical_pipeline_effect")
    if effect not in {"beneficial", "neutral", "harmful", "unassessable", "not_available"}:
        raise ValueError("Explicit historical pipeline effect required")
    if row["historical_output"]["projected_window_text"] is None and effect != "not_available":
        raise ValueError("Missing window proposal cannot have an assessed effect")
    result = attach_review(row, normalized)
    result.update(training_eligible=False, independent_acceptance_eligible=False,
                  allowed_usage="verified_consumed_audio_diagnostic_only",
                  historical_pipeline_effect=effect, isolated_current_refiner_effect=None)
    return result


def run(args):
    manifest = json.loads((args.queue / "manifest.json").read_text())
    if sha(args.queue / "review_queue.jsonl") != manifest["output_sha256"]["review_queue.jsonl"]:
        raise ValueError("Review queue changed")
    rows = {row["id"]: row for row in read_rows(args.queue / "review_queue.jsonl")}
    reviews = read_rows(args.reviews)
    seen, verified, errors = set(), [], []
    for review in reviews:
        identity = review.get("id")
        try:
            if identity not in rows or identity in seen:
                raise ValueError("Unknown or duplicate review identity")
            seen.add(identity)
            verified.append(validate_review(rows[identity], review, args.queue))
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(dict(id=identity, error=str(error)))
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "verified_diagnostics.jsonl", verified)
    write_jsonl(args.output / "rejected_reviews.jsonl", errors)
    summary = dict(reviewed=len(verified), rejected=len(errors), unreviewed=len(rows)-len(verified),
        historical_pipeline_effects=Counter(row["historical_pipeline_effect"] for row in verified),
        queue_sha256=manifest["output_sha256"]["review_queue.jsonl"], reviews_sha256=sha(args.reviews),
        training_eligible=0, independent_acceptance_eligible=0,
        scope="explicitly_reviewed_consumed_audio_diagnostic_not_isolated_current_refiner_gains")
    (args.output / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue",type=Path,required=True)
    parser.add_argument("--reviews",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
