#!/usr/bin/env python3
"""Offline scoring for the five fixed confirmation-long concatenation inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUT_PLAN = OUT / "confirmation_long_inputs" / "plan.json"
COLLECTOR_PLAN = OUT / "confirmation_long_test_plan.json"
REFERENCES = ROOT / "results/entity_modes_dataset_20261013/references.json"
RESULTS = OUT / "confirmation_long_results.jsonl"
PROTOCOL = OUT / "confirmation_long_protocol.json"
sys.path.insert(0, str(ROOT / ".runtime/functional_test_deps"))
sys.path.insert(0, str(OUT / "source_snapshot"))
import opencc  # noqa: E402
from system.entity_audio_repair import normalized  # noqa: E402
from system.zh_itn_bridge import ZhITNNormalizer  # noqa: E402

ARMS = ("old", "new")
cc = opencc.OpenCC("t2s")
itn = ZhITNNormalizer(ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
TIMING_FIELDS = ("total_seconds", "settled_total_seconds", "post_return_cleanup_seconds",
                 "setup_seconds", "stream_seconds", "tail_seconds")


def norm(text):
    return normalized(cc.convert(itn.normalize(str(text)).text))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    def percentile(q):
        position = (len(values) - 1) * q
        low, high = int(position), min(int(position) + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"n": len(values), "mean": sum(values) / len(values),
            "median": percentile(.5), "p90": percentile(.9)}


def make_reference_rows(input_plan, references):
    joined = {}
    for group in input_plan["groups"]:
        components = [item for item in group["components_and_gaps"]
                      if item.get("kind") != "inserted_silence"]
        if len(components) != group["component_count"] or not components:
            raise ValueError(f"bad component manifest for {group['id']}")
        prior_start = -1.0
        texts, gold = [], Counter()
        component_ids = []
        for component in components:
            sid = component["id"]
            if component.get("source_split") != "confirmation":
                raise ValueError(f"non-confirmation component in {group['id']}: {sid}")
            if float(component["start_seconds"]) < prior_start:
                raise ValueError(f"components out of order in {group['id']}")
            prior_start = float(component["end_seconds"])
            if sid not in references:
                raise ValueError(f"reference missing for component {sid}")
            label = references[sid]
            texts.append(str(label["reference"]))
            for entity in label["gold_entities"]:
                key = norm(entity["canonical"])
                if not key:
                    raise ValueError(f"empty normalized gold canonical in {sid}")
                gold[key] += 1
            component_ids.append(sid)
        reference = norm("".join(texts))
        if not reference:
            raise ValueError(f"empty joined reference for {group['id']}")
        if any(reference.count(word) < count for word, count in gold.items()):
            # A concatenated boundary may alter ITN grouping; treat mismatch as
            # a preparation error instead of silently changing gold counts.
            raise ValueError(f"gold Counter exceeds joined normalized reference for {group['id']}")
        joined[group["id"]] = {"reference": reference, "gold": gold,
                                "component_ids": component_ids,
                                "expected_mentions": sum(gold.values())}
    return joined


def row_metrics(row, labels):
    result = row["result"]
    clean = norm(result.get("clean_text", ""))
    raw = norm(result.get("raw_text", ""))
    gold = labels["gold"]
    reference = labels["reference"]
    clean_hits = sum(min(count, clean.count(word)) for word, count in gold.items())
    raw_hits = sum(min(count, raw.count(word)) for word, count in gold.items())
    repair = result.get("entity_audio_repair") or {}
    changes = list(repair.get("changes") or [])
    incremental = repair.get("incremental") or {}
    absent_changes = [change for change in changes if norm(change.get("canonical", "")) not in reference]
    fallbacks = int(incremental.get("legacy_final_fallbacks", 0) or 0)
    fallbacks += int(incremental.get("fallbacks", 0) or 0)
    fallback_active = repair.get("phase") == "fallback" or bool(repair.get("fallback_active"))
    streaming_replay = repair.get("streaming_replay") or {}
    fallback_active = fallback_active or bool(streaming_replay.get("fallback_active"))
    return {"id": row["id"], "arm": row["arm"], "library_key": row["library_key"],
        "expected_mentions": labels["expected_mentions"],
        "clean_hits": clean_hits, "raw_hits": raw_hits,
        "clean_character_errors": Levenshtein.distance(reference, clean),
        "raw_character_errors": Levenshtein.distance(reference, raw),
        "reference_characters": len(reference),
        "replay_checks": int(repair.get("checks", 0) or 0),
        "changes": len(changes), "reference_absent_change_count": len(absent_changes),
        "reference_absent_changes": absent_changes,
        "replay_audio_seconds": float(repair.get("replay_audio_seconds", 0) or 0),
        "incremental_stream_revisions": int(incremental.get("stream_revisions", 0) or 0),
        "incremental_fallbacks": fallbacks, "fallback_active": fallback_active,
        "incremental_errors": int(incremental.get("errors", 0) or 0),
        "refiner_session_logical_calls": int(result.get("session_refiner_call_count", 0) or 0),
        "timing": row.get("timing", {})}


def run(partial=False):
    input_plan_bytes = INPUT_PLAN.read_bytes()
    input_plan = json.loads(input_plan_bytes)
    collector_plan_bytes = COLLECTOR_PLAN.read_bytes()
    collector_plan = json.loads(collector_plan_bytes)
    groups = input_plan.get("groups", [])
    samples = collector_plan.get("samples", [])
    if len(groups) != 5 or len(samples) != 5:
        raise ValueError("both confirmation-long plans must describe exactly five inputs")
    group_by_id = {group["id"]: group for group in groups}
    sample_by_id = {sample["id"]: sample for sample in samples}
    if len(group_by_id) != 5 or len(sample_by_id) != 5 or set(group_by_id) != set(sample_by_id):
        raise ValueError("collector sample IDs do not map one-to-one to concatenation plan")

    expected = {}
    for sid, group in group_by_id.items():
        sample = sample_by_id[sid]
        for field in ("pcm_sha256", "library_key"):
            if sample.get(field) != group.get(field):
                raise ValueError(f"collector plan {field} mismatch for {sid}")
        if float(sample["duration_seconds"]) != float(group["duration_seconds"]):
            raise ValueError(f"collector plan duration mismatch for {sid}")
        path = (ROOT / group["pcm_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        pcm_digest = sha256_bytes(path.read_bytes())
        if pcm_digest != group["pcm_sha256"]:
            raise ValueError(f"actual PCM hash mismatch for {sid}")
        expected[sid] = {"library_key": group["library_key"], "pcm_sha256": pcm_digest,
                         "pcm_path": str(path), "duration_seconds": float(group["duration_seconds"])}

    references = json.loads(REFERENCES.read_text())["samples"]
    labels = make_reference_rows(input_plan, references)
    if set(labels) != set(expected):
        raise ValueError("reference aggregation IDs differ from fixed audio IDs")

    raw = RESULTS.read_bytes() if RESULTS.exists() else b""
    rows = []
    skipped_trailing = False
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith((b"\n", b"\r")):
            if partial and index == len(lines) - 1:
                skipped_trailing = True
                break
            raise ValueError("incomplete trailing result line; use --partial")
        rows.append(json.loads(line))

    by_key = {}
    seen_library_by_id = {}
    for row in rows:
        sid, arm = row.get("id"), row.get("arm")
        key = (sid, arm)
        if sid not in expected or arm not in ARMS or key in by_key:
            raise ValueError(f"unexpected or duplicate result row: {key}")
        want = expected[sid]
        if row.get("pcm_sha256") != want["pcm_sha256"]:
            raise ValueError(f"result PCM hash mismatch for {sid}/{arm}")
        if row.get("library_key") != want["library_key"]:
            raise ValueError(f"result library key mismatch for {sid}/{arm}")
        by_key[key] = row
        previous = seen_library_by_id.setdefault(sid, row["library_key"])
        if previous != row["library_key"]:
            raise ValueError(f"old/new library mismatch for {sid}: {previous} vs {row['library_key']}")
    if not partial and len(by_key) != 10:
        raise AssertionError(f"full score needs 10 distinct sample/arm rows, found {len(by_key)}")

    detail_rows = []
    by_arm = {arm: {"samples": 0, "expected_mentions": 0, "clean_hits": 0, "raw_hits": 0,
        "reference_characters": 0, "clean_character_errors": 0, "raw_character_errors": 0,
        "reference_absent_change_count": 0, "changes": 0, "replay_checks": 0,
        "replay_audio_seconds": 0.0, "incremental_stream_revisions": 0,
        "incremental_fallbacks": 0, "fallback_samples": 0, "incremental_errors": 0,
        "refiner_session_logical_calls": 0,
        "timing": {field: [] for field in TIMING_FIELDS}} for arm in ARMS}
    for (sid, arm), row in by_key.items():
        item = row_metrics(row, labels[sid])
        detail_rows.append(item)
        aggregate = by_arm[arm]
        for key in ("samples", "expected_mentions", "clean_hits", "raw_hits", "reference_characters",
                    "clean_character_errors", "raw_character_errors", "reference_absent_change_count",
                    "changes", "replay_checks", "incremental_stream_revisions", "incremental_fallbacks",
                    "incremental_errors", "refiner_session_logical_calls"):
            aggregate[key] += int(item.get(key, 1 if key == "samples" else 0))
        aggregate["replay_audio_seconds"] += item["replay_audio_seconds"]
        aggregate["fallback_samples"] += int(item["fallback_active"])
        for field in TIMING_FIELDS:
            if field in item["timing"] and item["timing"][field] is not None:
                aggregate["timing"][field].append(float(item["timing"][field]))
    for arm, values in by_arm.items():
        values["clean_entity_recall"] = values["clean_hits"] / values["expected_mentions"] if values["expected_mentions"] else None
        values["raw_entity_recall"] = values["raw_hits"] / values["expected_mentions"] if values["expected_mentions"] else None
        ref_chars = values["reference_characters"]
        values["clean_CER"] = values["clean_character_errors"] / ref_chars if ref_chars else None
        values["raw_CER"] = values["raw_character_errors"] / ref_chars if ref_chars else None
        values["timing"] = {field: distribution(numbers) for field, numbers in values["timing"].items()}

    paired = []
    by_sample_arm = {(item["id"], item["arm"]): item for item in detail_rows}
    for sid in expected:
        old, new = by_sample_arm.get((sid, "old")), by_sample_arm.get((sid, "new"))
        if old is not None and new is not None:
            if old["library_key"] != new["library_key"]:
                raise ValueError(f"old/new library mismatch for paired sample {sid}")
            paired.append({"id": sid, "library_key": old["library_key"],
                "clean_entity_hit_delta_new_minus_old": new["clean_hits"] - old["clean_hits"],
                "raw_entity_hit_delta_new_minus_old": new["raw_hits"] - old["raw_hits"],
                "clean_character_error_delta_new_minus_old": new["clean_character_errors"] - old["clean_character_errors"],
                "raw_character_error_delta_new_minus_old": new["raw_character_errors"] - old["raw_character_errors"]})

    protocol = json.loads(PROTOCOL.read_text()) if PROTOCOL.exists() else {}
    output = {"scope": "Only the five fixed confirmation-long concatenation inputs; references/gold are scorer-only.",
        "partial": partial, "complete": len(by_key) == 10, "result_rows": len(rows),
        "paired_samples": len(paired), "skipped_incomplete_trailing_row": skipped_trailing,
        "reference_policy": "For each fixed input, concatenate component references in plan order and add each component gold canonical mention Counter; never pass these labels to inference.",
        "scoring": "Raw and clean entity hits are canonical substring mentions capped by summed component gold counts. CER is Levenshtein distance over normalized concatenated reference and transcript.",
        "by_arm": by_arm, "samples": detail_rows, "paired_deltas": paired,
        "input_hashes": {"confirmation_long_inputs/plan.json": sha256_bytes(input_plan_bytes),
            "confirmation_long_test_plan.json": sha256_bytes(collector_plan_bytes),
            "references.json": sha256_bytes(REFERENCES.read_bytes()),
            "confirmation_long_results.jsonl": sha256_bytes(raw)},
        "protocol": {"path": str(PROTOCOL.relative_to(ROOT)) if PROTOCOL.exists() else None,
                     "sha256": sha256_bytes(PROTOCOL.read_bytes()) if PROTOCOL.exists() else None,
                     "candidate_sha256": protocol.get("candidate_sha256"),
                     "source_hashes": protocol.get("source_hashes", {})}}
    output_path = OUT / (("partial_" if partial else "") + "confirmation_long_score.json")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": output_path.name, "rows": len(rows), "pairs": len(paired),
        "complete": output["complete"], "by_arm": by_arm}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true", help="score completed rows so far and pair only samples with both arms")
    args = parser.parse_args()
    run(args.partial)
