#!/usr/bin/env python3
"""Offline scoring for three sparse concatenation stresses and one Li latency case."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REFS_PATH = ROOT / "results/entity_local_replay_20261012/sparse_development/references.json"
sys.path.insert(0, str(ROOT / ".runtime/functional_test_deps"))
sys.path.insert(0, str(OUT / "source_snapshot"))
import opencc  # noqa: E402
from system.entity_audio_repair import normalized  # noqa: E402
from system.zh_itn_bridge import ZhITNNormalizer  # noqa: E402

cc = opencc.OpenCC("t2s")
itn = ZhITNNormalizer(ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
ARMS = ("old", "new")
LI_ID = "li_feiyu_322s_web_development_audio"
LI_TARGET = "厉飞宇"


def norm(text):
    return normalized(cc.convert(itn.normalize(str(text)).text))


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    def percentile(q):
        pos = (len(values) - 1) * q
        lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"n": len(values), "mean": sum(values) / len(values),
            "median": percentile(.5), "p90": percentile(.9)}


def first_seen(updates, field, target):
    for update in updates:
        text = update.get(field)
        if text and target in norm(text):
            return {"elapsed_seconds": update.get("elapsed_seconds"),
                    "event": update.get("event"), "acked_frames": update.get("acked_frames")}
    return None


def run(partial=False):
    plan_path = OUT / "long_test_plan.json"
    results_path = OUT / "long_results.jsonl"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    samples = plan["samples"]
    ids = [sample["id"] for sample in samples]
    if len(ids) != 4 or len(set(ids)) != 4 or LI_ID not in ids:
        raise ValueError("long_test_plan must contain three sparse cases and the single Li case")
    refs = json.loads(REFS_PATH.read_text())
    sparse = {sample["id"] for sample in samples if sample["id"] != LI_ID}
    if sparse != set(refs):
        raise ValueError(f"sparse plan/reference IDs differ: {sorted(sparse ^ set(refs))}")
    raw_results = results_path.read_bytes() if results_path.exists() else b""
    rows, skipped_trailing = [], False
    for index, line in enumerate(raw_results.splitlines(keepends=True)):
        if not line.endswith((b"\n", b"\r")):
            if partial and index == len(raw_results.splitlines(keepends=True)) - 1:
                skipped_trailing = True
                break
            raise ValueError("incomplete final long result row; use --partial")
        rows.append(json.loads(line))
    by_key = {}
    for row in rows:
        key = (row["id"], row["arm"])
        if row["id"] not in set(ids) or row["arm"] not in ARMS or key in by_key:
            raise ValueError(f"unexpected or duplicate long result row: {key}")
        expected = next(s for s in samples if s["id"] == row["id"])
        if row.get("pcm_sha256") != expected["pcm_sha256"]:
            raise ValueError(f"PCM hash mismatch for {row['id']}")
        by_key[key] = row
    if not partial and len(by_key) != len(ids) * len(ARMS):
        raise AssertionError(f"full long score requires 8 rows, found {len(by_key)}")

    sparse_rows = []
    li_rows = []
    for sid in ids:
        for arm in ARMS:
            row = by_key.get((sid, arm))
            if row is None:
                continue
            result = row["result"]
            timing = row["timing"]
            if sid == LI_ID:
                target = norm(LI_TARGET)
                raw = result.get("raw_text", "")
                clean = result.get("clean_text", "")
                li_rows.append({"id": sid, "arm": arm,
                    "target": LI_TARGET,
                    "target_in_final_raw_text": target in norm(raw),
                    "target_in_final_clean_text": target in norm(clean),
                    "first_stream_raw": first_seen(row.get("updates", []), "raw_text", target),
                    "first_stream_clean": first_seen(row.get("updates", []), "clean_text", target),
                    "timing": timing, "audio_seconds": row.get("audio_seconds"),
                    "stream_update_count": len(row.get("updates", []))})
                continue
            label = refs[sid]
            reference = norm(label["reference"])
            canonical = norm(label["canonical"])
            if not reference or not canonical:
                raise ValueError(f"invalid sparse reference for {sid}")
            clean = norm(result.get("clean_text", ""))
            raw = norm(result.get("raw_text", ""))
            expected_mentions = reference.count(canonical)
            if expected_mentions < 1:
                raise ValueError(f"target canonical absent from normalized reference for {sid}")
            clean_hits = min(expected_mentions, clean.count(canonical))
            raw_hits = min(expected_mentions, raw.count(canonical))
            sparse_rows.append({"id": sid, "arm": arm, "canonical": label["canonical"],
                "expected_mentions": expected_mentions, "clean_hits": clean_hits,
                "raw_hits": raw_hits, "clean_character_errors": Levenshtein.distance(reference, clean),
                "raw_character_errors": Levenshtein.distance(reference, raw),
                "reference_characters": len(reference), "timing": row["timing"],
                "replay_audio_seconds": float((result.get("entity_audio_repair") or {}).get("replay_audio_seconds", 0) or 0),
                "changes": len((result.get("entity_audio_repair") or {}).get("changes") or [])})

    by_arm = {}
    for arm in ARMS:
        values = [row for row in sparse_rows if row["arm"] == arm]
        expected_mentions = sum(v["expected_mentions"] for v in values)
        ref_chars = sum(v["reference_characters"] for v in values)
        clean_errors = sum(v["clean_character_errors"] for v in values)
        raw_errors = sum(v["raw_character_errors"] for v in values)
        timing_fields = ("total_seconds", "settled_total_seconds", "post_return_cleanup_seconds",
                         "setup_seconds", "stream_seconds", "tail_seconds")
        timing = {field: distribution([v["timing"][field] for v in values if field in v["timing"]])
                  for field in timing_fields}
        by_arm[arm] = {"sparse_samples": len(values), "expected_mentions": expected_mentions,
            "clean_hits": sum(v["clean_hits"] for v in values),
            "clean_entity_recall": sum(v["clean_hits"] for v in values) / expected_mentions if expected_mentions else None,
            "raw_hits": sum(v["raw_hits"] for v in values),
            "raw_entity_recall": sum(v["raw_hits"] for v in values) / expected_mentions if expected_mentions else None,
            "clean_character_errors": clean_errors,
            "clean_CER": clean_errors / ref_chars if ref_chars else None,
            "raw_character_errors": raw_errors,
            "raw_CER": raw_errors / ref_chars if ref_chars else None,
            "replay_audio_seconds": sum(v["replay_audio_seconds"] for v in values),
            "changes": sum(v["changes"] for v in values), "timing": timing}
    paired = []
    for sid in ids:
        old, new = (next((v for v in sparse_rows if v["id"] == sid and v["arm"] == arm), None) for arm in ARMS)
        if old and new:
            paired.append({"id": sid, "entity_hit_delta_new_minus_old": new["clean_hits"] - old["clean_hits"],
                           "clean_character_error_delta_new_minus_old": new["clean_character_errors"] - old["clean_character_errors"]})

    # The Li case has no complete annotation. Report only target presence,
    # first streaming occurrence and elapsed timings; never compute CER.
    li_by_arm = {arm: [value for value in li_rows if value["arm"] == arm] for arm in ARMS}
    li_timing = {}
    for arm, values in li_by_arm.items():
        fields = ("total_seconds", "settled_total_seconds", "post_return_cleanup_seconds",
                  "setup_seconds", "stream_seconds", "tail_seconds")
        li_timing[arm] = {field: distribution([v["timing"][field] for v in values if field in v["timing"]])
                          for field in fields}
    protocol_path = OUT / "long_protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    report = {"scope": "Three sparse concatenated development stresses with references, plus one Li development latency/target-presence case without complete reference.",
        "partial": partial, "complete": len(by_key) == len(ids) * len(ARMS),
        "result_rows": len(rows), "paired_ids": len(paired), "skipped_incomplete_trailing_row": skipped_trailing,
        "sparse_reference_policy": "Only sparse_development/references.json is used; target canonical mention counts and CER are reported for its three listed concatenations.",
        "li_reference_policy": "No full reference available. No CER, WER, or general accuracy score is computed for Li; only final target presence, first stream update containing 厉飞宇, and timing are reported.",
        "sparse_by_arm": by_arm, "sparse_samples": sparse_rows, "sparse_paired_deltas": paired,
        "li_target": LI_TARGET, "li_by_arm": li_by_arm, "li_timing_by_arm": li_timing,
        "input_hashes": {"long_test_plan.json": digest_bytes(plan_bytes),
                         "sparse_development/references.json": digest_bytes(REFS_PATH.read_bytes()),
                         "long_results.jsonl": digest_bytes(raw_results)},
        "source_code_hashes": {"candidate_sha256": protocol.get("candidate_sha256"),
                               "protocol_source_hashes": protocol.get("source_hashes", {})}}
    output = OUT / (("partial_" if partial else "") + "long_score.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": output.name, "result_rows": len(rows), "paired_ids": len(paired),
                      "complete": report["complete"], "sparse_by_arm": by_arm,
                      "li_by_arm": li_by_arm, "li_timing_by_arm": li_timing}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true", help="score complete rows available so far; keep unmatched arms unpaired")
    args = parser.parse_args()
    run(args.partial)
