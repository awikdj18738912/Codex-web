#!/usr/bin/env python3
"""Offline paired scoring for known-dense and independent concatenation groups."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "results/entity_modes_dataset_20261013"
REFERENCES = DATA / "references.json"
sys.path.insert(0, str(ROOT / ".runtime/functional_test_deps"))
sys.path.insert(0, str(OUT / "source_snapshot"))
import opencc  # noqa: E402
from system.entity_audio_repair import normalized  # noqa: E402
from system.zh_itn_bridge import ZhITNNormalizer  # noqa: E402

ARMS = ("old", "new")
SPLITS = {
    "known_dense": {
        "input_plan": OUT / "known_dense_inputs" / "plan.json",
        "collector_plan": OUT / "known_dense_test_plan.json",
        "results": OUT / "known_dense_results.jsonl",
        "protocol": OUT / "known_dense_protocol.json",
    },
    "independent": {
        "input_plan": OUT / "independent_long_inputs" / "plan.json",
        "collector_plan": OUT / "independent_test_plan.json",
        "results": OUT / "independent_results.jsonl",
        "protocol": OUT / "independent_protocol.json",
    },
}
TIMING_FIELDS = ("total_seconds", "settled_total_seconds", "post_return_cleanup_seconds",
                 "setup_seconds", "stream_seconds", "tail_seconds")
COUNTER_FIELDS = ("detections", "admitted", "rejected", "windows", "completed_windows",
                  "stream_revisions", "budget_exhausted", "unmapped", "oversized_windows",
                  "errors", "legacy_final_fallbacks", "cache_confirmed_candidates",
                  "weak_unresolved_kept", "preflight_rejections", "candidates",
                  "low_score_rejected", "ambiguous_rejected", "echo_rejected",
                  "oversized_rejected", "budget_rejected", "queue_rejected",
                  "cache_reused", "late_candidates_kept", "full_audio_fallbacks")
BACKEND_NUMERIC_FIELDS = ("primary_pcm_seconds", "replay_pcm_seconds", "primary_http_calls",
                          "replay_http_calls", "primary_wall_seconds", "replay_wall_seconds",
                          "primary_reported_model_ms", "replay_reported_model_ms",
                          "active_http_requests")
BACKEND_SUM_FIELDS = tuple(field for field in BACKEND_NUMERIC_FIELDS
                           if field != "active_http_requests")
cc = opencc.OpenCC("t2s")
itn = ZhITNNormalizer(ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")


def norm(text: str) -> str:
    return normalized(cc.convert(itn.normalize(str(text)).text))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}

    def percentile(q):
        point = (len(values) - 1) * q
        low, high = int(point), min(int(point) + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (point - low)

    return {"n": len(values), "mean": sum(values) / len(values),
            "median": percentile(.5), "p90": percentile(.9)}


def components_for(group):
    """Accept both current group layouts without assuming a particular source_split."""
    if "components_and_gaps" in group:
        values = group["components_and_gaps"]
    else:
        values = group.get("components", [])
    components = [value for value in values if value.get("kind") != "inserted_silence"]
    if not components or len(components) != int(group.get("component_count", len(components))):
        raise ValueError(f"component manifest mismatch for {group.get('id')}")
    return components


def make_labels(input_plan, references):
    labels = {}
    for group in input_plan.get("groups", []):
        group_id = group["id"]
        pieces = []
        gold = Counter()
        component_ids = []
        source_splits = set()
        prior_end = -math.inf
        for component in components_for(group):
            start = float(component.get("start_seconds", component.get("start_sample", 0)))
            if start < prior_end:
                raise ValueError(f"components out of order for {group_id}")
            if "end_seconds" in component:
                prior_end = float(component["end_seconds"])
            elif "end_sample" in component and group.get("sample_rate_hz"):
                prior_end = float(component["end_sample"]) / float(group["sample_rate_hz"])
            component_id = str(component["id"])
            if component_id not in references:
                raise ValueError(f"fixed reference missing component {component_id} in {group_id}")
            reference = references[component_id]
            pieces.append(str(reference["reference"]))
            for entity in reference.get("gold_entities", []):
                canonical = norm(entity["canonical"])
                if not canonical:
                    raise ValueError(f"empty normalized gold canonical for {component_id}")
                gold[canonical] += 1
            component_ids.append(component_id)
            if component.get("source_split") is not None:
                source_splits.add(str(component["source_split"]))
        joined_reference = norm("".join(pieces))
        if not joined_reference:
            raise ValueError(f"empty normalized joined reference for {group_id}")
        if any(joined_reference.count(word) < count for word, count in gold.items()):
            raise ValueError(f"gold Counter exceeds joined normalized reference for {group_id}")
        labels[group_id] = {
            "reference": joined_reference,
            "gold": gold,
            "expected_mentions": sum(gold.values()),
            "component_ids": component_ids,
            "component_source_splits": sorted(source_splits),
        }
    if not labels:
        raise ValueError("input plan has no groups")
    return labels


def find_evaluation_backend(row):
    """Find evaluation_backend wherever the final worker snapshot placed it."""
    preferred = [row.get("result"), row.get("ready"), row]
    seen = set()

    def walk(value):
        if isinstance(value, dict):
            identity = id(value)
            if identity in seen:
                return None
            seen.add(identity)
            candidate = value.get("evaluation_backend")
            if isinstance(candidate, dict):
                return candidate
            for child in value.values():
                result = walk(child)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = walk(child)
                if result is not None:
                    return result
        return None

    for value in preferred:
        result = walk(value)
        if result is not None:
            return result
    return None


def score_row(row, label):
    result = row.get("result") or {}
    raw_text = norm(result.get("raw_text", ""))
    clean_text = norm(result.get("clean_text", ""))
    reference = label["reference"]
    gold = label["gold"]
    raw_hits = sum(min(count, raw_text.count(word)) for word, count in gold.items())
    clean_hits = sum(min(count, clean_text.count(word)) for word, count in gold.items())
    raw_errors = Levenshtein.distance(reference, raw_text)
    clean_errors = Levenshtein.distance(reference, clean_text)
    repair = result.get("entity_audio_repair") or {}
    changes = list(repair.get("changes") or [])
    absent_changes = [change for change in changes
                      if norm(change.get("canonical", "")) not in reference]
    incremental = repair.get("incremental") or {}
    backend = find_evaluation_backend(row)
    primary_text = None if not backend or backend.get("primary_finish_text") is None else norm(backend["primary_finish_text"])
    primary_hits = None
    primary_errors = None
    if primary_text is not None:
        primary_hits = sum(min(count, primary_text.count(word)) for word, count in gold.items())
        primary_errors = Levenshtein.distance(reference, primary_text)
    raw_vs_firstpass = None
    clean_vs_firstpass = None
    if primary_text is not None:
        raw_vs_firstpass = {
            "final_raw_hits": raw_hits,
            "primary_hits": primary_hits,
            "hit_delta_final_raw_minus_primary": raw_hits - primary_hits,
            "final_raw_character_errors": raw_errors,
            "primary_character_errors": primary_errors,
            "character_error_reduction_primary_minus_final_raw": primary_errors - raw_errors,
            "final_raw_CER": raw_errors / len(reference) if reference else None,
            "primary_CER": primary_errors / len(reference) if reference else None,
        }
        clean_vs_firstpass = {
            "final_clean_hits": clean_hits,
            "primary_hits": primary_hits,
            "hit_delta_final_clean_minus_primary": clean_hits - primary_hits,
            "final_clean_character_errors": clean_errors,
            "primary_character_errors": primary_errors,
            "character_error_reduction_primary_minus_final_clean": primary_errors - clean_errors,
            "final_clean_CER": clean_errors / len(reference) if reference else None,
            "primary_CER": primary_errors / len(reference) if reference else None,
        }
    timing = row.get("timing") or {}
    return {
        "id": row["id"], "arm": row["arm"], "library_key": row.get("library_key"),
        "corpus": row.get("corpus"), "stratum": row.get("stratum"),
        "component_ids": label["component_ids"],
        "component_source_splits": label["component_source_splits"],
        "expected_mentions": label["expected_mentions"],
        "reference_characters": len(reference),
        "raw_hits": raw_hits, "clean_hits": clean_hits,
        "raw_character_errors": raw_errors, "clean_character_errors": clean_errors,
        "changes": len(changes), "reference_absent_change_count": len(absent_changes),
        "reference_absent_changes": absent_changes,
        "reported_replay_audio_seconds": float(repair.get("replay_audio_seconds", 0) or 0),
        "replay_checks": int(repair.get("checks", 0) or 0),
        "repair_phase": repair.get("phase"), "repair_reason": repair.get("reason"),
        "incremental_counters": {key: int(incremental.get(key, 0) or 0) for key in COUNTER_FIELDS},
        "incremental_preflight_reason": incremental.get("preflight_reason"),
        "evaluation_backend_available": backend is not None,
        "primary_finish_text_available": primary_text is not None,
        "primary_hits": primary_hits, "primary_character_errors": primary_errors,
        "primary_reference_characters": len(reference) if primary_text is not None else None,
        "primary_CER": (primary_errors / len(reference) if primary_errors is not None and reference else None),
        "raw_vs_firstpass": raw_vs_firstpass,
        "clean_vs_firstpass": clean_vs_firstpass,
        "primary_metrics_missing_reason": (None if primary_text is not None else
            ("snapshot.evaluation_backend absent" if backend is None else "primary_finish_text absent")),
        "backend_counters": {
            key: (float(backend[key]) if backend and backend.get(key) is not None else None)
            for key in BACKEND_NUMERIC_FIELDS
        },
        "refiner_session_logical_calls": int(result.get("session_refiner_call_count", 0) or 0),
        "timing": timing,
        "pcm_sha256": row.get("pcm_sha256"),
    }


def read_complete_rows(path, partial):
    data = path.read_bytes() if path.exists() else b""
    rows = []
    skipped_trailing = False
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith((b"\n", b"\r")):
            if partial and index == len(lines) - 1:
                skipped_trailing = True
                break
            raise ValueError(f"incomplete trailing JSONL row in {path.name}; use --partial")
        rows.append(json.loads(line))
    return rows, skipped_trailing, data


def verify_protocol(split_name, paths):
    protocol_path = paths["protocol"]
    if not protocol_path.exists():
        raise FileNotFoundError(f"missing run protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    candidate = OUT / "candidate.py"
    experiment_worker = OUT / "experiment_worker.py"
    live_core = OUT / "live_core.py"
    expected_candidate = protocol.get("candidate_sha256")
    if not expected_candidate or sha256_file(candidate) != expected_candidate:
        raise ValueError("protocol candidate_sha256 does not match candidate.py")
    worker_hash = protocol.get("experiment_worker_sha256")
    if worker_hash is None:
        raise ValueError("protocol is missing experiment_worker_sha256")
    if sha256_file(experiment_worker) != worker_hash:
        raise ValueError("protocol experiment_worker_sha256 does not match experiment_worker.py")
    expected_live_core = protocol.get("live_core_sha256")
    if not expected_live_core:
        raise ValueError("protocol is missing live_core_sha256")
    if sha256_file(live_core) != expected_live_core:
        raise ValueError("protocol live_core_sha256 does not match live_core.py")
    worker_path = OUT / "worker.json"
    if not worker_path.is_file():
        raise FileNotFoundError(f"missing worker snapshot: {worker_path}")
    worker = json.loads(worker_path.read_text())
    worker_sources = worker.get("source_hashes")
    protocol_sources = protocol.get("source_hashes")
    if not worker_sources or protocol_sources != worker_sources:
        raise ValueError("protocol source_hashes do not match worker.json")
    source_hashes = protocol.get("source_hashes") or {}
    if not source_hashes:
        raise ValueError("protocol contains no source_hashes")
    for relative, expected in source_hashes.items():
        path = OUT / "source_snapshot" / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source snapshot hash mismatch: {relative}")
    return {
        "path": str(protocol_path.relative_to(ROOT)),
        "sha256": sha256_file(protocol_path),
        "candidate_sha256": expected_candidate,
        "experiment_worker_sha256": worker_hash,
        "live_core_sha256": expected_live_core,
        "source_hash_count": len(source_hashes),
        "source_hashes": source_hashes,
    }


def run(split_name: str, partial: bool):
    paths = SPLITS[split_name]
    plan_bytes = paths["input_plan"].read_bytes()
    input_plan = json.loads(plan_bytes)
    collector_plan_bytes = paths["collector_plan"].read_bytes()
    collector_plan = json.loads(collector_plan_bytes)
    groups = input_plan.get("groups", [])
    samples = collector_plan.get("samples", [])
    group_by_id = {group["id"]: group for group in groups}
    sample_by_id = {sample["id"]: sample for sample in samples}
    if not group_by_id or len(group_by_id) != len(groups):
        raise ValueError("input plan group IDs are empty or duplicated")
    if len(sample_by_id) != len(samples) or set(sample_by_id) != set(group_by_id):
        raise ValueError("collector plan IDs do not map one-to-one to input plan groups")

    expected = {}
    for group_id, group in group_by_id.items():
        sample = sample_by_id[group_id]
        for field in ("pcm_sha256", "library_key"):
            if sample.get(field) != group.get(field):
                raise ValueError(f"collector plan {field} mismatch for {group_id}")
        if float(sample["duration_seconds"]) != float(group["duration_seconds"]):
            raise ValueError(f"collector plan duration mismatch for {group_id}")
        # Hash the collector's actual upload path. Some copied group plans keep
        # the original source experiment's path while their collector plan maps
        # the same PCM into this new experiment directory.
        pcm_path = Path(sample.get("pcm_path", group["pcm_path"]))
        if not pcm_path.is_absolute():
            pcm_path = ROOT / pcm_path
        pcm_path = pcm_path.resolve()
        if not pcm_path.is_file():
            raise FileNotFoundError(pcm_path)
        actual_pcm_hash = sha256_file(pcm_path)
        if actual_pcm_hash != group["pcm_sha256"]:
            raise ValueError(f"input PCM hash mismatch for {group_id}")
        expected[group_id] = {
            "library_key": group["library_key"], "pcm_sha256": actual_pcm_hash,
            "pcm_path": str(pcm_path), "duration_seconds": float(group["duration_seconds"]),
        }

    # References are read only for component IDs present in this fixed plan.
    references_all = json.loads(REFERENCES.read_text())["samples"]
    used_component_ids = {
        str(component["id"])
        for group in groups for component in components_for(group)
    }
    references = {component_id: references_all[component_id]
                  for component_id in used_component_ids if component_id in references_all}
    missing_refs = sorted(used_component_ids - references.keys())
    if missing_refs:
        raise ValueError(f"fixed references missing planned component IDs: {missing_refs[:8]}")
    labels = make_labels(input_plan, references)
    if set(labels) != set(expected):
        raise ValueError("reference aggregation IDs differ from planned audio IDs")

    rows, skipped_trailing, result_bytes = read_complete_rows(paths["results"], partial)
    by_key = {}
    for row in rows:
        group_id, arm = row.get("id"), row.get("arm")
        key = (group_id, arm)
        if group_id not in expected or arm not in ARMS or key in by_key:
            raise ValueError(f"unexpected or duplicate result row: {key}")
        want = expected[group_id]
        if row.get("pcm_sha256") != want["pcm_sha256"]:
            raise ValueError(f"result PCM hash mismatch for {group_id}/{arm}")
        if row.get("library_key") != want["library_key"]:
            raise ValueError(f"result library_key mismatch for {group_id}/{arm}")
        collector_sample = sample_by_id[group_id]
        for source_field in ("corpus", "stratum", "split"):
            if source_field in collector_sample and row.get(source_field) != collector_sample[source_field]:
                raise ValueError(f"result {source_field} mismatch for {group_id}/{arm}")
        by_key[key] = row
    if not partial and len(by_key) != 2 * len(expected):
        raise AssertionError(f"full score requires {2 * len(expected)} rows, found {len(by_key)}")

    details = []
    by_arm = {
        arm: {
            "samples": 0, "expected_mentions": 0,
            "raw_hits": 0, "clean_hits": 0, "reference_characters": 0,
            "raw_character_errors": 0, "clean_character_errors": 0,
            "changes": 0, "reference_absent_change_count": 0,
            "reported_replay_audio_seconds": 0.0, "replay_checks": 0,
            "incremental_counter_totals": Counter(),
            "preflight_reason_counts": Counter(), "repair_reason_counts": Counter(),
            "repair_phase_counts": Counter(),
            "refiner_session_logical_calls": 0,
            "timing": {field: [] for field in TIMING_FIELDS},
            "evaluation_backend_rows": 0, "primary_finish_text_rows": 0,
            "primary_expected_mentions": 0, "primary_hits": 0,
            "primary_reference_characters": 0, "primary_character_errors": 0,
            "final_raw_hits_on_primary_rows": 0,
            "final_raw_character_errors_on_primary_rows": 0,
            "final_clean_hits_on_primary_rows": 0,
            "final_clean_character_errors_on_primary_rows": 0,
            "evaluation_backend_numeric": {field: {"n": 0, "sum": 0.0}
                                           for field in BACKEND_SUM_FIELDS},
            "active_http_requests_at_snapshot": [],
        } for arm in ARMS
    }
    for (group_id, arm), row in by_key.items():
        item = score_row(row, labels[group_id])
        details.append(item)
        aggregate = by_arm[arm]
        aggregate["samples"] += 1
        for key in ("expected_mentions", "raw_hits", "clean_hits", "reference_characters",
                    "raw_character_errors", "clean_character_errors", "changes",
                    "reference_absent_change_count", "replay_checks",
                    "refiner_session_logical_calls"):
            aggregate[key] += int(item[key])
        aggregate["reported_replay_audio_seconds"] += item["reported_replay_audio_seconds"]
        aggregate["repair_phase_counts"][str(item["repair_phase"] or "<missing>")] += 1
        aggregate["repair_reason_counts"][str(item["repair_reason"] or "<none>")] += 1
        if item["incremental_preflight_reason"]:
            aggregate["preflight_reason_counts"][str(item["incremental_preflight_reason"])] += 1
        for counter_name, value in item["incremental_counters"].items():
            aggregate["incremental_counter_totals"][counter_name] += int(value)
        for field in TIMING_FIELDS:
            value = item["timing"].get(field)
            if value is not None:
                aggregate["timing"][field].append(float(value))
        if item["evaluation_backend_available"]:
            aggregate["evaluation_backend_rows"] += 1
            for field, value in item["backend_counters"].items():
                if value is not None:
                    if field == "active_http_requests":
                        aggregate["active_http_requests_at_snapshot"].append(float(value))
                    else:
                        aggregate["evaluation_backend_numeric"][field]["n"] += 1
                        aggregate["evaluation_backend_numeric"][field]["sum"] += float(value)
        if item["primary_finish_text_available"]:
            aggregate["primary_finish_text_rows"] += 1
            aggregate["primary_expected_mentions"] += item["expected_mentions"]
            aggregate["primary_hits"] += item["primary_hits"]
            aggregate["primary_reference_characters"] += item["primary_reference_characters"]
            aggregate["primary_character_errors"] += item["primary_character_errors"]
            aggregate["final_raw_hits_on_primary_rows"] += item["raw_hits"]
            aggregate["final_raw_character_errors_on_primary_rows"] += item["raw_character_errors"]
            aggregate["final_clean_hits_on_primary_rows"] += item["clean_hits"]
            aggregate["final_clean_character_errors_on_primary_rows"] += item["clean_character_errors"]

    for aggregate in by_arm.values():
        ref_chars = aggregate["reference_characters"]
        aggregate["raw_entity_recall"] = (aggregate["raw_hits"] / aggregate["expected_mentions"]
                                          if aggregate["expected_mentions"] else None)
        aggregate["clean_entity_recall"] = (aggregate["clean_hits"] / aggregate["expected_mentions"]
                                            if aggregate["expected_mentions"] else None)
        aggregate["raw_CER"] = aggregate["raw_character_errors"] / ref_chars if ref_chars else None
        aggregate["clean_CER"] = aggregate["clean_character_errors"] / ref_chars if ref_chars else None
        aggregate["timing"] = {field: distribution(values) for field, values in aggregate["timing"].items()}
        primary_expected = aggregate["primary_expected_mentions"]
        primary_chars = aggregate["primary_reference_characters"]
        aggregate["primary_entity_recall"] = (aggregate["primary_hits"] / primary_expected
                                               if primary_expected else None)
        aggregate["primary_CER"] = (aggregate["primary_character_errors"] / primary_chars
                                     if primary_chars else None)
        if aggregate["primary_finish_text_rows"]:
            raw_errors = aggregate["final_raw_character_errors_on_primary_rows"]
            clean_errors = aggregate["final_clean_character_errors_on_primary_rows"]
            primary_errors = aggregate["primary_character_errors"]
            primary_chars = aggregate["primary_reference_characters"]
            aggregate["raw_vs_firstpass"] = {
                "interpretation": "Same-session final raw vs primary_finish_text; raw-level entity-module signal.",
                "hit_delta_final_raw_minus_primary": aggregate["final_raw_hits_on_primary_rows"] - aggregate["primary_hits"],
                "character_error_reduction_primary_minus_final_raw": primary_errors - raw_errors,
                "primary_CER": primary_errors / primary_chars if primary_chars else None,
                "final_raw_CER": raw_errors / primary_chars if primary_chars else None,
            }
            aggregate["clean_vs_firstpass"] = {
                "interpretation": "Same-session final clean vs primary_finish_text; full Web pipeline including refinement.",
                "hit_delta_final_clean_minus_primary": aggregate["final_clean_hits_on_primary_rows"] - aggregate["primary_hits"],
                "character_error_reduction_primary_minus_final_clean": primary_errors - clean_errors,
                "primary_CER": primary_errors / primary_chars if primary_chars else None,
                "final_clean_CER": clean_errors / primary_chars if primary_chars else None,
            }
        else:
            aggregate["raw_vs_firstpass"] = None
            aggregate["clean_vs_firstpass"] = None
        aggregate["active_http_requests_at_snapshot"] = distribution(aggregate["active_http_requests_at_snapshot"])
        aggregate["timing_backend_note"] = (
            "primary/replay HTTP calls count backend requests, not model calls. primary/replay wall seconds measure completed request-handler time and may omit unfinished requests. primary/replay reported_model_ms are ASR-payload-reported cumulative model-stage times, including buffering and function work; they are not GPU hardware timers and are not model call counts. active_http_requests is the snapshot value."
        )
        aggregate["incremental_counter_totals"] = dict(aggregate["incremental_counter_totals"])
        aggregate["preflight_reason_counts"] = dict(aggregate["preflight_reason_counts"])
        aggregate["repair_reason_counts"] = dict(aggregate["repair_reason_counts"])
        aggregate["repair_phase_counts"] = dict(aggregate["repair_phase_counts"])
        aggregate["evaluation_backend_numeric"] = {
            key: ({"n": value["n"], "sum": value["sum"]} if value["n"] else None)
            for key, value in aggregate["evaluation_backend_numeric"].items()
        }

    detail_by_id_arm = {(item["id"], item["arm"]): item for item in details}
    paired = []
    for group_id in expected:
        old = detail_by_id_arm.get((group_id, "old"))
        new = detail_by_id_arm.get((group_id, "new"))
        if old is None or new is None:
            continue
        for field in ("library_key", "corpus", "stratum", "component_ids", "pcm_sha256"):
            if old.get(field) != new.get(field):
                raise ValueError(f"old/new pair source mismatch for {group_id}: {field}")
        paired.append({
            "id": group_id, "library_key": old["library_key"],
            "clean_hit_delta_new_minus_old": new["clean_hits"] - old["clean_hits"],
            "raw_hit_delta_new_minus_old": new["raw_hits"] - old["raw_hits"],
            "clean_character_error_delta_new_minus_old": new["clean_character_errors"] - old["clean_character_errors"],
            "raw_character_error_delta_new_minus_old": new["raw_character_errors"] - old["raw_character_errors"],
            "primary_hit_delta_new_minus_old": (new["primary_hits"] - old["primary_hits"]
                if new["primary_hits"] is not None and old["primary_hits"] is not None else None),
            "primary_character_error_delta_new_minus_old": (new["primary_character_errors"] - old["primary_character_errors"]
                if new["primary_character_errors"] is not None and old["primary_character_errors"] is not None else None),
            "raw_hit_delta_already_present_in_primary": (
                new["raw_hits"] - old["raw_hits"] == new["primary_hits"] - old["primary_hits"]
                if new["primary_hits"] is not None and old["primary_hits"] is not None else None),
            "raw_error_delta_already_present_in_primary": (
                new["raw_character_errors"] - old["raw_character_errors"] ==
                new["primary_character_errors"] - old["primary_character_errors"]
                if new["primary_character_errors"] is not None and old["primary_character_errors"] is not None else None),
        })

    protocol = verify_protocol(split_name, paths)
    refs_hash = sha256_file(REFERENCES)
    report = {
        "split": split_name,
        "partial": partial,
        "complete": len(by_key) == 2 * len(expected),
        "result_rows": len(rows),
        "planned_groups": len(expected),
        "paired_groups": len(paired),
        "skipped_incomplete_trailing_row": skipped_trailing,
        "scope": "Only component IDs listed in the fixed input plan are scored. References/gold are scorer-only and never added to inference inputs.",
        "source_split_values_in_plan": sorted({
            split for group in groups for component in components_for(group)
            if (split := component.get("source_split")) is not None
        }),
        "scoring": "Component references are concatenated in plan order; canonical gold mention Counters are added across components. Raw and clean hits are capped substring counts; CER is Levenshtein on normalized joined reference and text.",
        "same_session_firstpass": "When result.entity_audio_repair.evaluation_backend.primary_finish_text exists, raw_vs_firstpass compares same-session final raw to first-pass ASR and is the entity-module raw-level signal. clean_vs_firstpass compares final clean to first-pass ASR and includes the full Web refinement pipeline. Paired deltas show whether final-raw cross-arm hit/error differences were already present in primary_finish_text. Missing fields remain null. HTTP calls are backend request counts, not model calls. primary/replay reported_model_ms are ASR-payload-reported cumulative model-stage times including buffering and function work, not GPU hardware timing or model invocation counts. Wall seconds cover completed measured HTTP handlers and may omit unfinished requests.",
        "by_arm": by_arm,
        "samples": details,
        "paired_deltas": paired,
        "input_hashes": {
            "input_plan": sha256_bytes(plan_bytes),
            "collector_plan": sha256_bytes(collector_plan_bytes),
            "references_json": refs_hash,
            "results_jsonl": sha256_bytes(result_bytes),
        },
        "protocol_hashes": protocol,
    }
    output_path = OUT / (("partial_" if partial else "") + f"{split_name}_score.json")
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": output_path.name, "rows": len(rows), "paired_groups": len(paired),
        "complete": report["complete"], "by_arm": by_arm}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--partial", action="store_true",
                        help="score complete rows available now; pair only groups with both arms")
    args = parser.parse_args()
    run(args.split, args.partial)
