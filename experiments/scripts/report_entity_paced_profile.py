#!/usr/bin/env python3
"""Summarize the fixed eight-sample, real-time paced entity backend profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.report_entity_backend_comparison import norm, score  # noqa: E402

RUN_DEFAULT = ROOT / "results/entity_backend_comparison_20261015"
EXPECTED_ARMS = ("qwen_raw", "qwen_ctc_sidecar", "native_hotword", "native_broad_hotword")
EXPECTED_MODE = "realtime_160ms_wallclock_paced"
EXPECTED_SAMPLES = 8
SCHEDULE_TOLERANCE_SECONDS = 0.002


def read_jsonl(path: Path, allow_partial: bool) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    out: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            obj = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            is_tail = index == len(lines) - 1 and not line.endswith((b"\n", b"\r"))
            if allow_partial and is_tail:
                break
            raise ValueError(f"invalid JSONL record at {path}:{index+1}")
        if not isinstance(obj, dict):
            raise ValueError(f"expected JSON object at {path}:{index+1}")
        out.append(obj)
    return out


def selection_hash(selection: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(selection, sort_keys=True).encode("utf-8")).hexdigest()


def finite(v: Any, label: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"{label} must be a finite number, got {v!r}")
    return float(v)


def percentile_summary(values: list[float], *, include_max: bool = True) -> dict[str, Any]:
    if not values:
        out: dict[str, Any] = {"n": 0, "median_seconds": None}
        if include_max:
            out["max_seconds"] = None
        return out
    out = {"n": len(values), "median_seconds": statistics.median(values)}
    if include_max:
        out["max_seconds"] = max(values)
    return out


def get_arm_text(arm: str, trace: dict[str, Any]) -> str:
    key = {"qwen_raw": "final_text", "qwen_ctc_sidecar": "final_fused_text",
           "native_hotword": "text", "native_broad_hotword": "text"}[arm]
    text = trace.get(key)
    if not isinstance(text, str):
        raise ValueError(f"{arm} lacks trace.{key}")
    return text


def validate_record(row: dict[str, Any], main_hash: str, broad_hash: str) -> None:
    sid, arm = str(row.get("id") or ""), str(row.get("arm") or "")
    if arm not in EXPECTED_ARMS:
        raise ValueError(f"unexpected arm {arm!r} for {sid}")
    if row.get("profile_mode") != EXPECTED_MODE:
        raise ValueError(f"{sid}/{arm}: expected profile_mode={EXPECTED_MODE!r}")
    if not row.get("broad_arm_enabled"):
        raise ValueError(f"{sid}/{arm}: broad arm was not enabled")
    selection = row.get("selection") or {}
    if selection.get("selection_sha256") != main_hash:
        raise ValueError(f"{sid}/{arm}: main selection hash mismatch")
    if selection.get("broad_selection_sha256") != broad_hash:
        raise ValueError(f"{sid}/{arm}: broad selection hash mismatch")
    if float(selection.get("native_hotword_score")) != 2.0 or float(selection.get("hybrid_minimum_gap")) != -2.0 or float(selection.get("hybrid_minimum_margin")) != 0.1:
        raise ValueError(f"{sid}/{arm}: main parameters differ from the selected comparison settings")

    trace = row.get("trace")
    if not isinstance(trace, dict):
        raise ValueError(f"{sid}/{arm}: trace missing")
    if arm in ("qwen_raw", "qwen_ctc_sidecar") and trace.get("start_context_applied") is not False:
        raise ValueError(f"{sid}/{arm}: Qwen start_context_applied must be false")
    audio = finite(trace.get("audio_seconds"), f"{sid}/{arm}.audio_seconds")
    schedule = trace.get("chunk_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(f"{sid}/{arm}: empty chunk_schedule")
    for event in schedule:
        available = finite(event.get("audio_available_seconds"), f"{sid}/{arm}.audio_available_seconds")
        submitted = finite(event.get("submit_wall_seconds"), f"{sid}/{arm}.submit_wall_seconds")
        if submitted + SCHEDULE_TOLERANCE_SECONDS < available:
            raise ValueError(f"{sid}/{arm}: chunk submitted before audio was available ({submitted:.6f} < {available:.6f})")
    last = finite(trace.get("last_audio_submit_wall_seconds"), f"{sid}/{arm}.last_audio_submit_wall_seconds")
    last_sched = finite(schedule[-1].get("submit_wall_seconds"), f"{sid}/{arm}.last_schedule_submit_wall_seconds")
    if abs(last - last_sched) > SCHEDULE_TOLERANCE_SECONDS:
        raise ValueError(f"{sid}/{arm}: last_audio_submit does not match schedule tail")
    total = finite(trace.get("total_wall_seconds"), f"{sid}/{arm}.total_wall_seconds")
    tail = finite(trace.get("finish_tail_seconds"), f"{sid}/{arm}.finish_tail_seconds")
    if total + SCHEDULE_TOLERANCE_SECONDS < last or tail < 0:
        raise ValueError(f"{sid}/{arm}: invalid total/finish tail")

    if arm == "qwen_raw":
        if abs(tail - max(0.0, total - last)) > SCHEDULE_TOLERANCE_SECONDS:
            raise ValueError(f"{sid}/{arm}: finish_tail is inconsistent with total wall and last submit")
    elif arm == "qwen_ctc_sidecar":
        final_fusion = finite(trace.get("final_fusion_completed_wall_seconds"), f"{sid}/{arm}.final_fusion_completed_wall_seconds")
        sidecar_finish = finite(trace.get("sidecar_finish_wall_seconds"), f"{sid}/{arm}.sidecar_finish_wall_seconds")
        expected_tail = max(0.0, final_fusion - last)
        if abs(total - final_fusion) > SCHEDULE_TOLERANCE_SECONDS or abs(total - sidecar_finish) > SCHEDULE_TOLERANCE_SECONDS:
            raise ValueError(f"{sid}/{arm}: total/sidecar finish must include final fusion completion")
        if abs(tail - expected_tail) > SCHEDULE_TOLERANCE_SECONDS:
            raise ValueError(f"{sid}/{arm}: finish_tail must include final fusion after last audio submit")
        finite(trace.get("ctc_compute_seconds"), f"{sid}/{arm}.ctc_compute_seconds")
        finite(trace.get("final_fusion_compute_seconds"), f"{sid}/{arm}.final_fusion_compute_seconds")
        if "final_fused_text" not in trace:
            raise ValueError(f"{sid}/{arm}: final fusion output missing")
    else:
        if abs(total - finite(trace.get("finish_wall_seconds"), f"{sid}/{arm}.finish_wall_seconds")) > SCHEDULE_TOLERANCE_SECONDS:
            raise ValueError(f"{sid}/{arm}: total_wall differs from finish_wall")
        finite(trace.get("compute_wall_seconds"), f"{sid}/{arm}.compute_wall_seconds")
        finite(trace.get("flush_compute_seconds"), f"{sid}/{arm}.flush_compute_seconds")


def metric_stats(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    traces = [row["trace"] for row in records]
    first = [finite(t["first_text_wall_seconds"], f"{row['id']}/{arm}.first_text_wall_seconds")
             for row, t in zip(records, traces) if t.get("first_text_wall_seconds") is not None]
    tails = [finite(t["finish_tail_seconds"], f"{row['id']}/{arm}.finish_tail_seconds") for row, t in zip(records, traces)]
    lags = [finite(t["total_wall_seconds"], f"{row['id']}/{arm}.total_wall_seconds") - finite(t["audio_seconds"], f"{row['id']}/{arm}.audio_seconds") for row, t in zip(records, traces)]
    out = {
        "samples": len(records),
        "first_text_wall_seconds": percentile_summary(first, include_max=False),
        "tail_after_last_audio_submit_seconds": percentile_summary(tails),
        "total_wall_minus_audio_seconds": percentile_summary(lags),
    }
    if arm == "qwen_ctc_sidecar":
        out["ctc_compute_seconds"] = percentile_summary([finite(t["ctc_compute_seconds"], "ctc compute") for t in traces])
        out["final_fusion_compute_seconds"] = percentile_summary([finite(t["final_fusion_compute_seconds"], "final fusion compute") for t in traces])
        out["incremental_event_fusion_compute_seconds_sum"] = percentile_summary([
            sum(finite(ev["fusion_compute_seconds"], "event fusion compute") for ev in t.get("events", []) if ev.get("fusion_compute_seconds") is not None)
            for t in traces
        ])
    if arm in ("native_hotword", "native_broad_hotword"):
        out["native_compute_wall_seconds"] = percentile_summary([finite(t["compute_wall_seconds"], "native compute") for t in traces])
        out["flush_compute_seconds"] = percentile_summary([finite(t["flush_compute_seconds"], "native flush compute") for t in traces])
    if arm in ("qwen_raw", "qwen_ctc_sidecar"):
        http = [finite(t["http_seconds"], "qwen http seconds") for t in traces if t.get("http_seconds") is not None]
        out["qwen_http_seconds"] = percentile_summary(http)
    return out


def quality_diagnostics(records_by_arm: dict[str, list[dict[str, Any]]], refs: dict[str, Any]) -> dict[str, Any]:
    arms_out: dict[str, Any] = {}
    per_arm: dict[str, list[dict[str, Any]]] = {}
    for arm, records in records_by_arm.items():
        scored = []
        for row in records:
            sid, trace = row["id"], row["trace"]
            refobj = refs.get(sid)
            if not isinstance(refobj, dict):
                raise ValueError(f"references missing profile id {sid}")
            scored.append(score(str(refobj.get("reference", "")), get_arm_text(arm, trace), refobj.get("gold_entities", [])))
        hits = sum(x["entity_hits"] for x in scored)
        mentions = sum(x["entity_mentions"] for x in scored)
        errors = sum(x["cer_errors"] for x in scored)
        chars = sum(x["reference_chars"] for x in scored)
        arms_out[arm] = {"samples": len(scored), "entity_hits": hits, "gold_mentions": mentions,
                         "entity_recall": hits / mentions if mentions else None,
                         "cer_errors": errors, "reference_chars": chars, "cer": errors / chars if chars else None}
        per_arm[arm] = scored
    return {"scope": "Eight fixed samples only; exploratory diagnostics, not a broad quality conclusion.", "arms": arms_out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=RUN_DEFAULT)
    ap.add_argument("--allow-partial", action="store_true", help="allow incomplete profile records for interim API validation")
    args = ap.parse_args()
    run = args.run_dir.resolve()
    profile_path = run / "paced_profile.jsonl"
    main_selection_path, broad_selection_path = run / "selection.json", run / "broad_vocab" / "selection.json"
    for path in (profile_path, main_selection_path, broad_selection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    records = read_jsonl(profile_path, args.allow_partial)
    if not records:
        raise ValueError("paced profile has no complete records")
    main_selection = json.loads(main_selection_path.read_text(encoding="utf-8"))
    broad_selection = json.loads(broad_selection_path.read_text(encoding="utf-8"))
    main_hash, broad_hash = selection_hash(main_selection), selection_hash(broad_selection)
    if (float(main_selection.get("native_hotword_score")) != 2.0 or
        float(main_selection.get("hybrid_minimum_gap")) != -2.0 or
        float(main_selection.get("hybrid_minimum_margin")) != 0.1):
        raise ValueError("main selection does not match frozen 2.0 / -2.0 / 0.1 parameters")

    expected_ids: list[str] | None = None
    fixed_meta: list[dict[str, Any]] | None = None
    for row in records:
        row_meta = row.get("fixed_eight_inputs")
        if not isinstance(row_meta, list) or len(row_meta) != EXPECTED_SAMPLES:
            raise ValueError(f"{row.get('id')}/{row.get('arm')}: fixed_eight_inputs must contain exactly eight IDs")
        row_ids = [str(x.get("id") or "") for x in row_meta]
        if len(set(row_ids)) != EXPECTED_SAMPLES or any(not x for x in row_ids):
            raise ValueError("fixed_eight_inputs IDs must be eight unique nonempty IDs")
        if expected_ids is None:
            expected_ids, fixed_meta = row_ids, row_meta
        elif row_ids != expected_ids or row_meta != fixed_meta:
            raise ValueError("fixed_eight_inputs metadata differs between profile records")
        validate_record(row, main_hash, broad_hash)
        if str(row.get("id")) not in set(expected_ids):
            raise ValueError(f"record ID {row.get('id')} is outside the fixed eight")

    assert expected_ids is not None
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        key = (str(row["id"]), str(row["arm"]))
        if key in rows_by_key:
            raise ValueError(f"duplicate profile ID/arm record: {key}")
        rows_by_key[key] = row
    missing = [(sid, arm) for sid in expected_ids for arm in EXPECTED_ARMS if (sid, arm) not in rows_by_key]
    extra_arms = sorted({str(row.get("arm")) for row in records} - set(EXPECTED_ARMS))
    if extra_arms:
        raise ValueError(f"unexpected profile arms: {extra_arms}")
    if missing and not args.allow_partial:
        raise ValueError(f"formal profile requires 8 IDs x 4 arms = 32 unique records; missing {len(missing)} cells, examples={missing[:10]}")

    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (sid, arm), row in rows_by_key.items():
        by_id[sid][arm] = row
    expected_pcm = {str(x["id"]): str(x["pcm_sha256"]) for x in (fixed_meta or [])}
    expected_audio = {str(x["id"]): finite(x["duration_seconds"], "fixed input duration") for x in (fixed_meta or [])}
    for sid, arm_rows in by_id.items():
        for row in arm_rows.values():
            if str(row.get("pcm_sha256")) != expected_pcm[sid]:
                raise ValueError(f"{sid}/{row['arm']}: PCM hash differs from fixed input manifest")
            if abs(finite(row["trace"].get("audio_seconds"), f"{sid}/{row['arm']}.audio_seconds") - expected_audio[sid]) > 1e-6:
                raise ValueError(f"{sid}/{row['arm']}: decoded audio_seconds differs from fixed input manifest")
        present = set(arm_rows)
        if len(present) > 1 and len({str(x.get("pcm_sha256")) for x in arm_rows.values()}) != 1:
            raise ValueError(f"{sid}: PCM hashes differ across arms")

    records_by_arm = {arm: [rows_by_key[(sid, arm)] for sid in expected_ids if (sid, arm) in rows_by_key] for arm in EXPECTED_ARMS}
    arm_stats = {arm: metric_stats(records_by_arm[arm], arm) for arm in EXPECTED_ARMS}
    paired_tail: list[dict[str, Any]] = []
    paired_lags: list[float] = []
    for sid in expected_ids:
        raw = rows_by_key.get((sid, "qwen_raw"))
        hybrid = rows_by_key.get((sid, "qwen_ctc_sidecar"))
        if raw and hybrid:
            raw_tail = finite(raw["trace"]["finish_tail_seconds"], "raw tail")
            hybrid_tail = finite(hybrid["trace"]["finish_tail_seconds"], "hybrid tail")
            diff = hybrid_tail - raw_tail
            paired_lags.append(diff)
            paired_tail.append({"id": sid, "qwen_raw_tail_seconds": raw_tail,
                                "hybrid_tail_seconds": hybrid_tail,
                                "hybrid_minus_raw_tail_seconds": diff})
    paired_summary = {"paired_samples": len(paired_lags), "hybrid_minus_raw_tail_seconds": percentile_summary(paired_lags),
                      "per_sample": paired_tail}

    refs_path = run / "references.json"
    quality = None
    if refs_path.is_file():
        refdoc = json.loads(refs_path.read_text(encoding="utf-8"))
        quality = quality_diagnostics(records_by_arm, refdoc.get("samples", {}))
    completeness = len(rows_by_key) == EXPECTED_SAMPLES * len(EXPECTED_ARMS) and not missing
    out = {
        "status": "complete" if completeness else "partial",
        "profile_path": str(profile_path), "profile_mode": EXPECTED_MODE,
        "required_ids": EXPECTED_SAMPLES, "required_arms": list(EXPECTED_ARMS),
        "record_count": len(rows_by_key), "unique_ids_present": len(by_id),
        "complete_id_arm_cells": len(rows_by_key), "missing_id_arm_cells": missing,
        "fixed_eight_inputs": fixed_meta,
        "selection_hashes": {"main_selection_sha256": main_hash, "broad_selection_sha256": broad_hash},
        "validation": {"all_pcm_hashes_match_fixed_manifest": True, "all_selection_hashes_match": True,
                       "all_records_realtime_paced": True, "all_chunks_submitted_after_audio_available_within_2ms": True,
                       "qwen_context_not_applied": True, "hybrid_total_and_finish_tail_include_final_fusion": True},
        "latency_by_arm": arm_stats,
        "paired_hybrid_vs_raw_tail": paired_summary,
        "quality_diagnostics_small_n": quality,
        "interpretation": ["These are eight paced samples and descriptive medians/maxima; they do not establish a service guarantee or p95 latency.",
                           "Hybrid finish tail and total wall include final fusion completion. CTC and fusion compute are separately recorded and are not interchangeable with wall latency.",
                           "The pacing schedule attempts to submit chunks only when their audio is available; small timing deviations up to 2 ms are accepted.",
                           "Qwen start context was not applied in either Qwen-backed profile arm.",
                           "Quality metrics, when references are available, are diagnostics for this fixed eight only."],
    }
    (run / "paced_profile_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Paced entity backend profile", "", f"Status: **{out['status']}**; {len(rows_by_key)}/32 records across {len(by_id)}/8 fixed IDs.",
             "", "The run used `realtime_160ms_wallclock_paced`. Every arm was checked for identical per-ID PCM hashes and selected-parameter hashes. Chunk submits were checked against audio availability with a 2 ms numeric tolerance. Qwen context had to remain unapplied.",
             "", "| Arm | N | First text wall median (s) | Tail after last audio submit median / max (s) | Total wall − audio median (s) |", "|---|---:|---:|---:|---:|"]
    for arm in EXPECTED_ARMS:
        x = arm_stats[arm]
        first = x["first_text_wall_seconds"]["median_seconds"]
        tail = x["tail_after_last_audio_submit_seconds"]
        lag = x["total_wall_minus_audio_seconds"]["median_seconds"]
        lines.append(f"| {arm} | {x['samples']} | {fmt(first)} | {fmt(tail['median_seconds'])} / {fmt(tail['max_seconds'])} | {fmt(lag)} |")
    h = paired_summary["hybrid_minus_raw_tail_seconds"]
    lines += ["", "## Hybrid versus raw Qwen paired finish tail", "", f"Paired samples: {paired_summary['paired_samples']}; hybrid minus raw tail median: {fmt(h['median_seconds'])} s; maximum absolute delta: {fmt(max((abs(x['hybrid_minus_raw_tail_seconds']) for x in paired_tail), default=0.0))} s.",
              "", "## Compute and quality diagnostics", "", "CTC and fusion compute summaries are in `paced_profile_summary.json`; quality metrics, if references were available, are marked small-sample diagnostics.",
              "", "Eight samples are descriptive only. Their medians/maxima do not establish a service guarantee or a p95 latency. The hybrid total wall and finish tail include final fusion. No quality or latency result from this profile alone supports a broad deployment conclusion.", ""]
    (run / "PACED_PROFILE.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
