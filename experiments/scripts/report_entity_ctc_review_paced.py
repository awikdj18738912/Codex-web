#!/usr/bin/env python3
"""Strict summary for the fixed eight-pair wallclock-paced CTC profile."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.profile_entity_ctc_review_combination import (
    ARMS, BASE, DEFAULT_OUTPUT, OLD, PACKET_BYTES, PACKET_SECONDS, short_test_rows,
)
from experiments.scripts.report_entity_backend_comparison import score

RUN = ROOT / "results/entity_ctc_live_integration_20261016"


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(row)
    return rows


def finite(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Missing or invalid {key}: {row.get('id')} {row.get('arm')}")
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}: {row.get('id')} {row.get('arm')}")
    return value


def aggregate_quality(items):
    hits = sum(item["entity_hits"] for item in items)
    mentions = sum(item["entity_mentions"] for item in items)
    errors = sum(item["cer_errors"] for item in items)
    chars = sum(item["reference_chars"] for item in items)
    return dict(samples=len(items), entity_hits=hits, entity_mentions=mentions,
                entity_recall=hits / mentions if mentions else 0., cer_errors=errors,
                reference_chars=chars, cer=errors / chars if chars else (0. if not errors else 1.))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--old-dir", type=Path, default=OLD)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--results", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base, old, run, result_path = (args.base_dir.resolve(), args.old_dir.resolve(),
                                   args.run_dir.resolve(), args.results.resolve())

    _inputs, expected_ids = short_test_rows(base, old)
    records = read_jsonl(result_path)
    indexed = {}
    for row in records:
        key = (str(row.get("id", "")), str(row.get("arm", "")))
        if key in indexed:
            raise ValueError(f"Duplicate profile row: {key}")
        indexed[key] = row
    expected_keys = {(sid, arm) for sid in expected_ids for arm in ARMS}
    if set(indexed) != expected_keys:
        missing, extra = expected_keys - set(indexed), set(indexed) - expected_keys
        raise ValueError(f"Expected exactly 8 IDs x off/on (16 rows); missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}")

    pairs = {}
    pacing_packets = 0
    for sid in expected_ids:
        pair = {arm: indexed[(sid, arm)] for arm in ARMS}
        pairs[sid] = pair
        if any(row.get("split") != "test" for row in pair.values()):
            raise ValueError(f"Non-test result in selected cohort: {sid}")
        if not pair["off"].get("pcm_sha256") or pair["off"].get("pcm_sha256") != pair["on"].get("pcm_sha256"):
            raise ValueError(f"Off/on PCM hash mismatch: {sid}")
        if pair["off"].get("raw_primary_text") != pair["on"].get("raw_primary_text"):
            raise ValueError(f"Off/on actual Qwen primary text mismatch: {sid}")
        if any(row.get("arm") != arm for arm, row in pair.items()):
            raise ValueError(f"Arm label mismatch: {sid}")

        for arm, row in pair.items():
            clock = finite(row, "audio_clock_wall_seconds")
            audio_seconds = finite(row, "audio_seconds")
            packets = row.get("chunk_submissions")
            if not isinstance(packets, list) or not packets:
                raise ValueError(f"Missing per-packet submit timestamps: {sid} {arm}")
            expected_packets = math.ceil(audio_seconds / PACKET_SECONDS)
            if len(packets) != expected_packets:
                raise ValueError(f"Packet count mismatch: {sid} {arm}: {len(packets)} != {expected_packets}")
            previous_audio = 0.
            for i, packet in enumerate(packets):
                available = float(packet["audio_seconds"])
                # The collector records submit_wall as `wall_seconds`.
                submitted = float(packet.get("submit_wall_seconds", packet.get("wall_seconds")))
                if not math.isfinite(available) or not math.isfinite(submitted):
                    raise ValueError(f"Non-finite packet timestamp: {sid} {arm} packet {i}")
                if available <= previous_audio or submitted < clock + available - .002:
                    raise ValueError(f"Packet was submitted early or has invalid audio position: {sid} {arm} packet {i}")
                previous_audio = available
                pacing_packets += 1
            if abs(previous_audio - audio_seconds) > .002:
                raise ValueError(f"Last packet does not cover full audio: {sid} {arm}")

    stats = {}
    paired_tail_delta = []
    for arm in ARMS:
        rows = [pairs[sid][arm] for sid in expected_ids]
        first = [float(row["first_nonempty_text_wall_seconds"])
                 for row in rows if row.get("first_nonempty_text_wall_seconds") is not None]
        tails = [finite(row, "finish_tail_seconds") for row in rows]
        stats[arm] = dict(samples=len(rows), first_nonempty_samples=len(first),
            first_nonempty_text_wall_median_seconds=statistics.median(first) if first else None,
            finish_tail_median_seconds=statistics.median(tails),
            finish_tail_max_seconds=max(tails),
            audio_review_checks=sum(int(row.get("audio_review_checks", 0) or 0) for row in rows),
            ctc_streaming_review_skips=sum(int((row.get("ctc") or {}).get("streaming_review_skips", 0) or 0) for row in rows),
            ctc_committed=sum(int((row.get("ctc") or {}).get("committed", 0) or 0) for row in rows),
            ctc_errors=sum(int((row.get("ctc") or {}).get("errors", 0) or 0) for row in rows))
    for sid in expected_ids:
        paired_tail_delta.append(finite(pairs[sid]["on"], "finish_tail_seconds") -
                                 finite(pairs[sid]["off"], "finish_tail_seconds"))

    # Small post-hoc quality diagnostic, strictly limited to the selected eight.
    refsdoc = json.loads((base / "references.json").read_text(encoding="utf-8"))
    refs = refsdoc.get("samples", {})
    quality_rows = {arm: [] for arm in ARMS}
    for sid in expected_ids:
        if sid not in refs:
            raise ValueError(f"Missing frozen reference for diagnostic ID {sid}")
        ref = refs[sid]
        for arm in ARMS:
            quality_rows[arm].append(score(str(ref.get("reference", "")),
                str(pairs[sid][arm].get("final_text", "")), ref.get("gold_entities", [])))
    quality = {arm: aggregate_quality(items) for arm, items in quality_rows.items()}

    summary = dict(cohort=dict(samples=8, rows=16, ids=expected_ids,
        selection="8 lowest SHA256('entity-combo-paced-v1:'+id) among 230 old short-finish/test IDs; selection used input IDs only"),
        verification=dict(exact_id_arm_coverage=True, paired_pcm_hashes_match=True,
            actual_primary_text_matches_between_arms=True, all_splits_test=True,
            pacing_tolerance_ms=2, pacing_packets_checked=pacing_packets,
            every_packet_met_audio_clock_plus_audio_position=True),
        arms=stats,
        paired_on_minus_off_finish_tail=dict(samples=len(paired_tail_delta),
            median_seconds=statistics.median(paired_tail_delta),
            per_id_seconds={sid: paired_tail_delta[i] for i, sid in enumerate(expected_ids)}),
        posthoc_quality_diagnostic=quality,
        interpretation="Eight paired recordings are a small diagnostic sample, not a service p95 estimate. Quality metrics are post-hoc diagnostics on these eight references only and do not establish generalization.")
    (run / "paired_paced_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# CTC + 原音复核 paced 配对诊断", "",
        "固定 8 个 short-finish/test ID，每条各跑 off 与 on；ID 按输入-only SHA256 排序选择。严格核对 16 条完整覆盖、split、配对 PCM SHA256、两臂实际 Qwen raw 主流文本一致，并逐包验证提交时间不早于 `audio_clock_wall + audio_seconds - 2 ms`。", "",
        "| Arm | First nonempty text 中位数 | Finish tail 中位数 | Finish tail 最大值 | 原音复核 checks | CTC review skips | CTC committed | CTC errors |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        x = stats[arm]
        first = "NA" if x["first_nonempty_text_wall_median_seconds"] is None else f"{x['first_nonempty_text_wall_median_seconds']:.3f}s"
        lines.append(f"| {arm} | {first} ({x['first_nonempty_samples']}/8) | {x['finish_tail_median_seconds']:.3f}s | {x['finish_tail_max_seconds']:.3f}s | {x['audio_review_checks']} | {x['ctc_streaming_review_skips']} | {x['ctc_committed']} | {x['ctc_errors']} |")
    lines += ["", f"配对 on−off finish-tail 差中位数：{statistics.median(paired_tail_delta):.3f}s。", "",
        "## 八条样本的事后质量诊断", "", "| Arm | Entity hits / mentions | Recall | CER |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        x = quality[arm]
        lines.append(f"| {arm} | {x['entity_hits']} / {x['entity_mentions']} | {x['entity_recall']:.2%} | {x['cer']:.2%} |")
    lines += ["", "这些质量数字只描述这 8 条已选录音，不用于估计总体精度或推广到其他语料。该 8 对小样本也不是服务 p95；finish tail 是最后一块音频提交到最终文本返回的 wall time。逐样本、逐包核验与完整统计见 `paired_paced_summary.json`。", ""]
    (run / "PAIRED_PACED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"PAIRED_PACED_REPORT_COMPLETE ids=8 rows=16 packets={pacing_packets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
