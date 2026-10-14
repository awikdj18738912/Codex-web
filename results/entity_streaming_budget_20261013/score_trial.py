#!/usr/bin/env python3
"""Offline paired scorer for the frozen entity-streaming budget trial."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "results/entity_modes_dataset_20261013"
sys.path.insert(0, str(ROOT / ".runtime/functional_test_deps"))
sys.path.insert(0, str(OUT / "source_snapshot"))
import opencc
from rapidfuzz.distance import Levenshtein

from system.entity_audio_repair import normalized  # noqa: E402
from system.zh_itn_bridge import ZhITNNormalizer  # noqa: E402

CONDITIONS = ("old", "new")
cc = opencc.OpenCC("t2s")
itn = ZhITNNormalizer(ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def norm(text: str) -> str:
    return normalized(cc.convert(itn.normalize(str(text)).text))


def distribution(values):
    if not values:
        return {"mean": None, "median": None, "p90": None}
    values = sorted(float(v) for v in values)
    def percentile(q):
        point = (len(values) - 1) * q
        lo, hi = int(point), min(int(point) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (point - lo)
    return {"mean": sum(values) / len(values), "median": percentile(.5), "p90": percentile(.9)}


def summarize_rows(rows, refs):
    by_arm = {arm: {"samples": 0, "expected_mentions": 0, "hits": 0,
                    "raw_hits": 0, "reference_characters": 0, "character_errors": 0,
                    "raw_character_errors": 0,
                    "reviewed_samples": 0, "checks": 0, "changes": 0,
                    "reference_absent_canonical_changes": 0,
                    "replay_audio_seconds": 0.0, "budget_timeout_samples": 0,
                    "budget_timeouts": 0,
                    "model_calls": {"session_refiner": 0, "session_refiner_retries": 0,
                                    "repetition_review": 0, "total": 0},
                    "timing": {key: [] for key in ("total_seconds", "setup_seconds", "stream_seconds",
                        "tail_seconds", "settled_total_seconds", "post_return_cleanup_seconds")}}
               for arm in CONDITIONS}
    type_stats = defaultdict(lambda: {a: Counter() for a in CONDITIONS})
    stratum_stats = defaultdict(lambda: {a: Counter() for a in CONDITIONS})
    by_id = defaultdict(dict)
    for row in rows:
        arm, sid = row["arm"], row["id"]
        reference = norm(refs[sid]["reference"])
        if not reference:
            raise ValueError(f"empty normalized reference for {sid}")
        gold = Counter(norm(entity["canonical"]) for entity in refs[sid]["gold_entities"])
        if any(not word or reference.count(word) < count for word, count in gold.items()):
            raise ValueError(f"gold entity missing from reference: {sid}")
        result = row["result"]
        text = norm(result["clean_text"])
        raw_text = norm(result["raw_text"])
        hits = sum(min(count, text.count(word)) for word, count in gold.items())
        raw_hits = sum(min(count, raw_text.count(word)) for word, count in gold.items())
        errors = Levenshtein.distance(reference, text)
        raw_errors = Levenshtein.distance(reference, raw_text)
        repair = result.get("entity_audio_repair") or {}
        changes = list(repair.get("changes") or [])
        stats = by_arm[arm]
        stats["samples"] += 1
        stats["expected_mentions"] += sum(gold.values())
        stats["hits"] += hits
        stats["raw_hits"] += raw_hits
        stats["reference_characters"] += len(reference)
        stats["character_errors"] += errors
        stats["raw_character_errors"] += raw_errors
        stats["reviewed_samples"] += int(int(repair.get("checks", 0)) > 0)
        stats["checks"] += int(repair.get("checks", 0))
        stats["changes"] += len(changes)
        stats["reference_absent_canonical_changes"] += sum(norm(c["canonical"]) not in reference for c in changes)
        stats["replay_audio_seconds"] += float(repair.get("replay_audio_seconds", 0) or 0)
        incremental = repair.get("incremental") or {}
        timeout_count = int(incremental.get("budget_exhausted", 0) or 0)
        stats["budget_timeouts"] += timeout_count
        stats["budget_timeout_samples"] += int(timeout_count > 0)
        refiner_calls = int(result.get("session_refiner_call_count", 0) or 0)
        retry_calls = int(result.get("session_refiner_retry_call_count", 0) or 0)
        repetition_calls = int(result.get("repetition_review_model_calls", 0) or 0)
        stats["model_calls"]["session_refiner"] += refiner_calls
        stats["model_calls"]["session_refiner_retries"] += retry_calls
        stats["model_calls"]["repetition_review"] += repetition_calls
        # Session refiner call_count is the canonical total; repetition-review
        # telemetry is retained separately and may overlap that aggregate.
        stats["model_calls"]["total"] += refiner_calls
        timing = row["timing"]
        for output_key, source_key in (("total_seconds", "total_seconds"),
                                       ("setup_seconds", "setup_seconds"),
                                       ("stream_seconds", "stream_seconds"),
                                       ("tail_seconds", "tail_seconds"),
                                       ("settled_total_seconds", "settled_total_seconds"),
                                       ("post_return_cleanup_seconds", "post_return_cleanup_seconds")):
            if source_key in timing and timing[source_key] is not None:
                stats["timing"].setdefault(output_key, []).append(float(timing[source_key]))
        gold_by_type = defaultdict(Counter)
        for entity in refs[sid]["gold_entities"]:
            gold_by_type[entity["entity_type"]][norm(entity["canonical"])] += 1
        for entity_type, typed_gold in gold_by_type.items():
            type_stats[entity_type][arm]["expected"] += sum(typed_gold.values())
            type_stats[entity_type][arm]["hits"] += sum(min(n, text.count(word))
                                                            for word, n in typed_gold.items())
        stratum_stats[row["stratum"]][arm]["samples"] += 1
        stratum_stats[row["stratum"]][arm]["expected_mentions"] += sum(gold.values())
        stratum_stats[row["stratum"]][arm]["hits"] += hits
        stratum_stats[row["stratum"]][arm]["reference_characters"] += len(reference)
        stratum_stats[row["stratum"]][arm]["character_errors"] += errors
        by_id[sid][arm] = {"hits": hits, "expected_mentions": sum(gold.values()),
                           "character_errors": errors, "timing": timing}

    for arm, stats in by_arm.items():
        stats["entity_recall"] = stats["hits"] / stats["expected_mentions"] if stats["expected_mentions"] else None
        stats["CER"] = stats["character_errors"] / stats["reference_characters"] if stats["reference_characters"] else None
        stats["raw_entity_recall"] = stats["raw_hits"] / stats["expected_mentions"] if stats["expected_mentions"] else None
        stats["raw_CER"] = stats["raw_character_errors"] / stats["reference_characters"] if stats["reference_characters"] else None
        stats["timing"] = {key: {"n": len(values), **distribution(values)}
                            for key, values in stats["timing"].items()}
    paired = Counter()
    deltas = {"entity_hit_delta_new_minus_old": [], "character_error_delta_new_minus_old": []}
    for sid, values in by_id.items():
        if set(values) != set(CONDITIONS):
            continue
        old, new = values["old"], values["new"]
        hit_delta = new["hits"] - old["hits"]
        error_delta = new["character_errors"] - old["character_errors"]
        paired["pairs"] += 1
        paired["entity_mentions_improved_samples"] += int(hit_delta > 0)
        paired["entity_mentions_regressed_samples"] += int(hit_delta < 0)
        paired["entity_mentions_equal_samples"] += int(hit_delta == 0)
        paired["entity_mentions_net_delta"] += hit_delta
        paired["character_errors_improved_samples"] += int(error_delta < 0)
        paired["character_errors_regressed_samples"] += int(error_delta > 0)
        paired["character_errors_equal_samples"] += int(error_delta == 0)
        paired["character_errors_net_delta"] += error_delta
        deltas["entity_hit_delta_new_minus_old"].append(hit_delta)
        deltas["character_error_delta_new_minus_old"].append(error_delta)
    type_result = {}
    for typ, arms in type_stats.items():
        type_result[typ] = {}
        for arm, counts in arms.items():
            expected, hits = counts["expected"], counts["hits"]
            type_result[typ][arm] = {"expected_mentions": expected, "hits": hits,
                                     "recall": hits / expected if expected else None}
    stratum_result = {}
    for stratum, arms in stratum_stats.items():
        stratum_result[stratum] = {}
        for arm, counts in arms.items():
            counts = dict(counts)
            counts["entity_recall"] = counts["hits"] / counts["expected_mentions"] if counts["expected_mentions"] else None
            counts["CER"] = counts["character_errors"] / counts["reference_characters"] if counts["reference_characters"] else None
            stratum_result[stratum][arm] = counts
    return {"by_arm": by_arm, "by_entity_type": type_result, "by_stratum": stratum_result,
            "paired": dict(paired), "paired_delta_distributions": {k: distribution(v) for k, v in deltas.items()}}


def load_complete_rows(path, partial):
    raw = path.read_bytes()
    rows = []
    lines = raw.splitlines(keepends=True)
    skipped_trailing = False
    for index, line in enumerate(lines):
        if not line.endswith((b"\n", b"\r")):
            if partial and index == len(lines) - 1:
                skipped_trailing = True
                break
            raise ValueError(f"incomplete result line {index + 1}; use --partial to skip only trailing partial row")
        rows.append(__import__("json").loads(line))
    return rows, skipped_trailing, raw


def run(split: str, partial: bool):
    plan_path = OUT / "frozen_test_plan.json"
    plan_bytes = plan_path.read_bytes()
    plan = __import__("json").loads(plan_bytes)
    expected_ids = [x["id"] for x in plan["samples"] if x.get("split") == split]
    expected_n = plan["development_count"] if split == "development" else plan["confirmation_count"]
    if len(expected_ids) != expected_n or len(set(expected_ids)) != expected_n:
        raise ValueError(f"frozen plan mismatch for {split}: expected {expected_n}, found {len(expected_ids)}")
    source = OUT / f"{split}_results.jsonl"
    if not source.exists():
        archived = OUT / f"v2_{split}" / f"{split}_results.jsonl"
        if archived.exists():
            source = archived
    rows, skipped_trailing, result_bytes = load_complete_rows(source, partial)
    by_key = {}
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"wrong split in {source.name}: {row.get('split')}")
        if row.get("id") not in set(expected_ids):
            raise ValueError(f"unexpected ID in results: {row.get('id')}")
        key = (row["id"], row["arm"])
        if key in by_key:
            raise ValueError(f"duplicate result row: {key}")
        if row["arm"] not in CONDITIONS:
            raise ValueError(f"unexpected arm {row['arm']}")
        by_key[key] = row
    if not partial:
        if len(by_key) != expected_n * len(CONDITIONS):
            raise AssertionError(f"full score requires {expected_n * 2} rows, found {len(by_key)}")
        if {sid for sid, arm in by_key} != set(expected_ids):
            raise AssertionError("full score missing planned sample IDs")
        for sid in expected_ids:
            if any((sid, arm) not in by_key for arm in CONDITIONS):
                raise AssertionError(f"full score missing paired arms for {sid}")
    refs_path = DATA / "references.json"
    refs = __import__("json").loads(refs_path.read_bytes())["samples"]
    if any(sid not in refs for sid in expected_ids):
        raise ValueError("reference missing a frozen plan ID")
    if any(row["pcm_sha256"] != next(s["pcm_sha256"] for s in plan["samples"] if s["id"] == row["id"])
           for row in rows):
        raise ValueError("PCM hash does not match frozen plan")

    scores = summarize_rows(rows, refs)
    run_dir = source.parent
    protocol_path = run_dir / f"{split}_protocol.json"
    if not protocol_path.exists():
        protocol_path = OUT / f"{split}_protocol.json"
    protocol = __import__("json").loads(protocol_path.read_text())
    source_hashes = dict(protocol.get("source_hashes", {}))
    source_hashes["candidate.py (protocol)" ] = protocol.get("candidate_sha256")
    local_code = {}
    for name in ("candidate.py", "collect.py", "experiment_worker.py", "start_worker.py"):
        path = run_dir / name
        if path.exists():
            local_code[name] = digest_file(path)
    local_code["score_trial.py"] = digest_file(OUT / "score_trial.py")
    snapshot_checks = {}
    for name, expected in protocol.get("source_hashes", {}).items():
        path = OUT / "source_snapshot" / name
        actual = digest_file(path) if path.exists() else None
        snapshot_checks[name] = {"expected": expected, "actual": actual, "matches": actual == expected}
    if protocol.get("candidate_sha256") and "candidate.py" in local_code:
        snapshot_checks["candidate.py"] = {"expected": protocol["candidate_sha256"],
                                           "actual": local_code["candidate.py"],
                                           "matches": local_code["candidate.py"] == protocol["candidate_sha256"]}
    report = {
        "split": split, "partial": partial, "complete": (not partial and len(by_key) == expected_n * 2),
        "scored_rows": len(rows), "scored_samples_by_arm": {a: scores["by_arm"][a]["samples"] for a in CONDITIONS},
        "paired_samples": scores["paired"].get("pairs", 0), "planned_samples": expected_n,
        "skipped_incomplete_trailing_row": skipped_trailing,
        "normalization": "Frozen benchmark formula: zh-itn normalization, OpenCC t2s, then entity_audio_repair.normalized (NFKC/casefold, strip Unicode P/Z/C).",
        "scoring": "Canonical substring hits capped by annotated reference mentions; CER is Levenshtein distance of normalized clean_text against normalized reference.",
        "scores": scores,
        "input_hashes": {"frozen_test_plan.json": digest_bytes(plan_bytes),
                         "references.json": digest_file(refs_path),
                         source.name: digest_bytes(result_bytes),
                         "planned_inputs": plan.get("input_source_sha256", {})},
        "source_code_hashes": {"protocol_source_snapshot": source_hashes,
                               "local_experiment_code": local_code,
                               "protocol_snapshot_verification": snapshot_checks},
        "notes": "Offline scoring only. A partial report may omit unfinished rows or unmatched arms; confirmation scoring does not feed threshold selection."
    }
    out = OUT / (("partial_" if partial else "") + f"{split}_score.json")
    out.write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(__import__("json").dumps({"output": out.name, "split": split, "partial": partial,
          "rows": len(rows), "pairs": scores["paired"].get("pairs", 0),
          "by_arm": scores["by_arm"], "paired": scores["paired"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "confirmation"), default="development")
    parser.add_argument("--partial", action="store_true")
    arguments = parser.parse_args()
    run(arguments.split, arguments.partial)
