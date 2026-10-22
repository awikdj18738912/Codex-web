#!/usr/bin/env python3
"""Compare cached backend hypotheses with existing entity-audio-review outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Reuse exactly the report scorer and its single ITN normalizer instance.
from experiments.scripts.report_entity_backend_comparison import norm, score, read_jsonl  # noqa: E402

BACKEND_DEFAULT = ROOT / "results/entity_backend_comparison_20261015"
OLD_REVIEW_DEFAULT = ROOT / "results/entity_challenge_scene_eval_20261014_v2"
SHORT_FINISH_DEFAULT = ROOT / "results/entity_short_finish_20261015_v1/evaluation"
EXPECTED_TEST = 1829
EXPECTED_SHORT_TOTAL = 315
EXPECTED_SHORT_TEST = 230
CURRENT_ARMS = {
    "qwen": "qwen_raw",
    "nativehotword": "native_hotword",
    "hybrid": "qwen_ctc_sidecar",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def unique_index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("id") or "")
        if not sid:
            raise ValueError(f"{label} has a row without id")
        if sid in out:
            raise ValueError(f"{label} contains duplicate id {sid}")
        out[sid] = row
    return out


def csv_text(row: dict[str, str], arm: str) -> str | None:
    value = row.get(arm + "_text")
    if value not in (None, ""):
        return value
    # Distinguish an empty scored transcription from a missing arm via the
    # report's numeric score columns, which are empty only when no hypothesis.
    if row.get(arm + "_entity_hits") not in (None, ""):
        return ""
    return None


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"observations": 0, "sum_seconds": 0.0, "mean_seconds": None,
                "p50_seconds": None, "p95_seconds": None, "max_seconds": None}
    ordered = sorted(values)
    return {"observations": len(values), "sum_seconds": sum(values),
            "mean_seconds": statistics.fmean(values), "p50_seconds": statistics.median(values),
            "p95_seconds": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
            "max_seconds": ordered[-1]}


def observed_review_timings(results: dict[str, dict[str, Any]], ids: set[str], *, short: bool) -> dict[str, Any]:
    fields = {
        "module_elapsed_seconds": ("result", "entity_audio_repair", "elapsed_seconds"),
        "incremental_detection_seconds": ("result", "entity_audio_repair", "incremental", "detection_seconds"),
        "incremental_review_wall_seconds": ("result", "entity_audio_repair", "incremental", "review_wall_seconds"),
        "web_request_total_seconds": ("timing", "total_seconds"),
        "web_request_settled_total_seconds": ("timing", "settled_total_seconds"),
    }
    if short:
        fields.update({
            "short_finish_wall_seconds": ("result", "entity_audio_repair", "incremental", "short_finish_wall_seconds"),
            "short_finish_replay_audio_seconds": ("result", "entity_audio_repair", "incremental", "short_finish_replay_seconds"),
        })
    observations: dict[str, list[float]] = {key: [] for key in fields}
    reused_count = 0
    for sid in sorted(ids):
        row = results.get(sid)
        if row is None:
            continue
        reused_count += bool(row.get("reused_from"))
        for key, path in fields.items():
            obj: Any = row
            for part in path:
                obj = obj.get(part) if isinstance(obj, dict) else None
            if isinstance(obj, (int, float)) and math.isfinite(float(obj)) and float(obj) >= 0:
                observations[key].append(float(obj))
    return {
        "sample_ids_with_result": sum(sid in results for sid in ids),
        "samples_marked_reused_from": reused_count,
        "fields": {key: numeric_summary(vals) for key, vals in observations.items()},
        "scope_note": (
            "Observed durations from prior Web/module result records for these exact IDs. module_elapsed_seconds and incremental review fields describe the entity-review path; web_request_* are whole request/settled totals including the surrounding service path. short_finish_replay_audio_seconds measures replayed audio duration, not wall time. None are native CPU timings or measured concurrent backend latency."
        ),
    }


def compare_pair(ids: list[str], scored: dict[str, dict[str, dict[str, Any]]],
                 texts: dict[str, dict[str, str]], groups: dict[str, str],
                 candidate: str, baseline: str) -> dict[str, Any]:
    def summarize(selected: list[str]) -> dict[str, Any]:
        rec = lost = 0
        chits = bhits = mentions = cerr = berr = chars = 0
        cer_better = cer_worse = cer_equal = entity_better = entity_worse = entity_equal = 0
        for sid in selected:
            cs, bs = scored[candidate][sid], scored[baseline][sid]
            ctext, btext = norm(texts[candidate][sid]), norm(texts[baseline][sid])
            for canonical, count in cs["gold_counts"].items():
                delta = min(count, ctext.count(canonical)) - min(count, btext.count(canonical))
                rec += max(delta, 0)
                lost += max(-delta, 0)
            chits += cs["entity_hits"]; bhits += bs["entity_hits"]
            mentions += cs["entity_mentions"]
            cerr += cs["cer_errors"]; berr += bs["cer_errors"]
            chars += cs["reference_chars"]
            entity_better += cs["entity_hits"] > bs["entity_hits"]
            entity_worse += cs["entity_hits"] < bs["entity_hits"]
            entity_equal += cs["entity_hits"] == bs["entity_hits"]
            cer_better += cs["cer_errors"] < bs["cer_errors"]
            cer_worse += cs["cer_errors"] > bs["cer_errors"]
            cer_equal += cs["cer_errors"] == bs["cer_errors"]
        ccer, bcer = (cerr / chars if chars else None), (berr / chars if chars else None)
        crecall, brecall = (chits / mentions if mentions else None), (bhits / mentions if mentions else None)
        return {
            "paired_samples": len(selected), "entity_recovered_mentions": rec, "entity_lost_mentions": lost,
            "entity_net_hits": chits - bhits, "entity_hits_candidate": chits, "entity_hits_baseline": bhits,
            "gold_mentions_denominator": mentions, "candidate_entity_recall": crecall,
            "baseline_entity_recall": brecall,
            "entity_recall_difference": (crecall - brecall) if crecall is not None and brecall is not None else None,
            "entity_improved_samples": entity_better, "entity_degraded_samples": entity_worse,
            "entity_equal_samples": entity_equal,
            "candidate_cer_errors": cerr, "baseline_cer_errors": berr,
            "reference_chars_denominator": chars, "candidate_cer": ccer, "baseline_cer": bcer,
            "cer_net_difference": (ccer - bcer) if ccer is not None and bcer is not None else None,
            "cer_net_error_count": cerr - berr,
            "cer_improved_samples": cer_better, "cer_degraded_samples": cer_worse,
            "cer_equal_samples": cer_equal,
        }

    per_group: dict[str, list[str]] = defaultdict(list)
    for sid in ids:
        per_group[groups[sid]].append(sid)
    return {"candidate_arm": candidate, "baseline_arm": baseline,
            "overall": summarize(ids), "source_group_count": len(per_group),
            "by_source_group": {group: summarize(sorted(group_ids)) for group, group_ids in sorted(per_group.items())}}


def arm_summary(ids: list[str], scored: dict[str, dict[str, dict[str, Any]]], arm: str) -> dict[str, Any]:
    samples = [scored[arm][sid] for sid in ids]
    hits = sum(x["entity_hits"] for x in samples)
    mentions = sum(x["entity_mentions"] for x in samples)
    errors = sum(x["cer_errors"] for x in samples)
    chars = sum(x["reference_chars"] for x in samples)
    return {"samples": len(samples), "entity_hits": hits, "gold_mentions": mentions,
            "entity_recall": hits / mentions if mentions else None,
            "cer_errors": errors, "reference_chars": chars,
            "cer": errors / chars if chars else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend-dir", type=Path, default=BACKEND_DEFAULT)
    ap.add_argument("--old-review-dir", type=Path, default=OLD_REVIEW_DEFAULT)
    ap.add_argument("--short-finish-dir", type=Path, default=SHORT_FINISH_DEFAULT)
    args = ap.parse_args()
    backend, old_dir, short_dir = (x.resolve() for x in (args.backend_dir, args.old_review_dir, args.short_finish_dir))
    needed = [backend / "sample_comparison.csv", backend / "inputs.jsonl", backend / "references.json",
              backend / "summary.json", backend / "qwen_primary_streams.jsonl",
              old_dir / "comparisons.jsonl", old_dir / "results.jsonl",
              short_dir / "comparisons.jsonl", short_dir / "results.jsonl"]
    for path in needed:
        if not path.is_file():
            raise FileNotFoundError(path)

    inputs_rows = read_jsonl(backend / "inputs.jsonl")
    inputs = unique_index(inputs_rows, "backend inputs")
    refs_doc = json.loads((backend / "references.json").read_text(encoding="utf-8"))
    refs = refs_doc.get("samples", {})
    if set(inputs) != set(refs):
        raise ValueError("backend inputs and references IDs differ")
    test_ids = {sid for sid, row in inputs.items() if row.get("split") == "test"}
    if len(test_ids) != EXPECTED_TEST:
        raise ValueError(f"expected exactly {EXPECTED_TEST} test input IDs, found {len(test_ids)}")

    backend_rows = read_csv(backend / "sample_comparison.csv")
    backend_samples = unique_index(backend_rows, "backend sample_comparison.csv")
    missing_backend = sorted(test_ids - set(backend_samples))
    if missing_backend:
        raise ValueError(f"backend comparison must cover all test IDs; missing {len(missing_backend)} (examples {missing_backend[:10]})")
    for sid in test_ids:
        if backend_samples[sid].get("split") != "test":
            raise ValueError(f"backend sample split mismatch for {sid}")

    qwen_rows = unique_index(read_jsonl(backend / "qwen_primary_streams.jsonl"), "backend Qwen streams")
    if not test_ids <= set(qwen_rows):
        raise ValueError(f"cached Qwen primary missing {len(test_ids-set(qwen_rows))} test IDs")
    unexpected_qwen = set(qwen_rows) - set(inputs)
    if unexpected_qwen:
        raise ValueError(f"cached Qwen has IDs outside backend inputs: {sorted(unexpected_qwen)[:10]}")
    # Keep all split IDs so the short-finish baseline check covers its dev rows too.
    qwen_texts = {sid: str(row.get("final_payload", {}).get("text", "")) for sid, row in qwen_rows.items()}

    old_comp = unique_index(read_jsonl(old_dir / "comparisons.jsonl"), "old scene comparisons")
    if len(old_comp) != 2000 or not test_ids <= set(old_comp):
        raise ValueError(f"old scene comparisons must provide the full 2000-sample set including all test IDs; found {len(old_comp)} rows")
    old_results = unique_index(read_jsonl(old_dir / "results.jsonl"), "old scene results")
    short_comp = unique_index(read_jsonl(short_dir / "comparisons.jsonl"), "short-finish comparisons")
    short_results = unique_index(read_jsonl(short_dir / "results.jsonl"), "short-finish results")
    short_ids = set(short_comp)
    if len(short_ids) != EXPECTED_SHORT_TOTAL or set(short_results) != short_ids:
        raise ValueError(f"short-finish evaluation must remain exactly {EXPECTED_SHORT_TOTAL} matched result/comparison IDs; got comparisons={len(short_ids)}, results={len(short_results)}")
    if not short_ids <= test_ids | {sid for sid, row in inputs.items() if row.get("split") == "dev"}:
        raise ValueError("short-finish IDs contain samples outside the backend dev/test cohort")
    short_test_ids = short_ids & test_ids
    if len(short_test_ids) != EXPECTED_SHORT_TEST:
        raise ValueError(f"expected {EXPECTED_SHORT_TEST} short-finish test intersections, found {len(short_test_ids)}")

    # The archived primary stream is authoritative for the legacy comparison.
    # Assert exact text identity rather than trusting a prior aggregate report.
    baseline_checks = {"old_scene_common": 0, "short_finish_common": 0}
    old_common_ids = set(old_comp) & set(qwen_texts)
    if len(old_common_ids) != 2000:
        raise ValueError(f"expected 2000 cached-Qwen/old-scene common IDs, found {len(old_common_ids)}")
    for sid in sorted(old_common_ids):
        arms = old_comp[sid].get("arms", {})
        for old_arm in ("paired_primary", "saved_baseline"):
            old_text = str((arms.get(old_arm) or {}).get("text", ""))
            if qwen_texts[sid] != old_text:
                raise ValueError(f"cached Qwen text differs from old {old_arm} for id {sid}")
        baseline_checks["old_scene_common"] += 1
    for sid in sorted(short_ids):
        if sid not in test_ids and inputs[sid].get("split") != "dev":
            raise ValueError(f"short-finish common ID has unexpected split: {sid}")
        arms = short_comp[sid].get("arms", {})
        for old_arm in ("paired_primary", "saved_baseline"):
            old_text = str((arms.get(old_arm) or {}).get("text", ""))
            if qwen_texts.get(sid) != old_text:
                raise ValueError(f"cached Qwen text differs from short-finish {old_arm} for id {sid}")
        baseline_checks["short_finish_common"] += 1

    old_source_groups = {sid: str(inputs[sid].get("source_group") or "unknown") for sid in test_ids}
    short_source_groups = {sid: str(inputs[sid].get("source_group") or "unknown") for sid in short_test_ids}
    old_texts: dict[str, dict[str, str]] = {"qwen": dict(qwen_texts), "existing_review": {}}
    old_qwen_scores: dict[str, dict[str, Any]] = {}
    old_module_scores: dict[str, dict[str, Any]] = {}
    for sid in sorted(test_ids):
        rec = old_comp[sid]
        ref_obj = refs[sid]
        ref, gold = str(ref_obj.get("reference", "")), ref_obj.get("gold_entities", [])
        if ref != str(rec.get("reference", "")) or gold != rec.get("gold_entities", []):
            raise ValueError(f"reference/gold mismatch between backend and old scene run for {sid}")
        old_texts["existing_review"][sid] = str(rec["arms"]["entity_audio_raw"]["text"])
        old_qwen_scores[sid] = score(ref, qwen_texts[sid], gold)
        old_module_scores[sid] = score(ref, old_texts["existing_review"][sid], gold)
    old_scores = {"qwen": old_qwen_scores, "existing_review": old_module_scores}
    old_pairs = {
        "qwen_vs_existing_review": compare_pair(sorted(test_ids), old_scores, old_texts, old_source_groups, "qwen", "existing_review"),
        "nativehotword_vs_existing_review": None,
        "hybrid_vs_existing_review": None,
    }
    old_arm_scores: dict[str, dict[str, Any]] = {"qwen": old_qwen_scores, "existing_review": old_module_scores}
    old_arm_texts: dict[str, dict[str, str]] = {"qwen": dict(qwen_texts), "existing_review": dict(old_texts["existing_review"])}
    for label, csv_arm in CURRENT_ARMS.items():
        texts: dict[str, str] = {}
        scores: dict[str, Any] = {}
        for sid in sorted(test_ids):
            text = csv_text(backend_samples[sid], csv_arm)
            if text is None:
                raise ValueError(f"backend arm {csv_arm} is missing test hypothesis for {sid}")
            texts[sid] = text
            scores[sid] = score(str(refs[sid].get("reference", "")), text, refs[sid].get("gold_entities", []))
        old_arm_texts[label] = texts
        old_arm_scores[label] = scores
        if label in ("nativehotword", "hybrid"):
            old_pairs[f"{label}_vs_existing_review"] = compare_pair(sorted(test_ids), old_arm_scores, old_arm_texts, old_source_groups, label, "existing_review")
    old_arm_summaries = {arm: arm_summary(sorted(test_ids), old_arm_scores, arm) for arm in old_arm_scores}

    short_names = ("qwen", "current_short_finish", "nativehotword", "hybrid", "existing_review")
    short_texts: dict[str, dict[str, str]] = {name: {} for name in short_names}
    short_scores: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in short_names}
    for sid in sorted(short_test_ids):
        ref_obj = refs[sid]
        ref, gold = str(ref_obj.get("reference", "")), ref_obj.get("gold_entities", [])
        short_rec = short_comp[sid]
        old_rec = old_comp[sid]
        for rec, label in ((short_rec, "short-finish"), (old_rec, "old scene")):
            if ref != str(rec.get("reference", "")) or gold != rec.get("gold_entities", []):
                raise ValueError(f"reference/gold mismatch between backend and {label} run for {sid}")
        arm_texts = {
            "qwen": qwen_texts[sid],
            "current_short_finish": str(short_rec["arms"]["entity_audio_raw"]["text"]),
            "nativehotword": csv_text(backend_samples[sid], CURRENT_ARMS["nativehotword"]),
            "hybrid": csv_text(backend_samples[sid], CURRENT_ARMS["hybrid"]),
            "existing_review": str(old_rec["arms"]["entity_audio_raw"]["text"]),
        }
        for arm, text in arm_texts.items():
            if text is None:
                raise ValueError(f"backend arm {arm} missing short-finish test hypothesis for {sid}")
            short_texts[arm][sid] = text
            short_scores[arm][sid] = score(ref, text, gold)
    short_arm_summaries = {arm: arm_summary(sorted(short_test_ids), short_scores, arm) for arm in short_names}
    short_pairs: dict[str, Any] = {}
    for i, candidate in enumerate(short_names):
        for baseline in short_names[:i]:
            short_pairs[f"{candidate}_vs_{baseline}"] = compare_pair(sorted(short_test_ids), short_scores, short_texts, short_source_groups, candidate, baseline)

    old_scene_timing = observed_review_timings(old_results, test_ids, short=False)
    short_timing = observed_review_timings(short_results, short_test_ids, short=True)
    old_reused_count = sum(bool(old_results.get(sid, {}).get("reused_from")) for sid in test_ids)
    summary = json.loads((backend / "summary.json").read_text(encoding="utf-8"))
    out = {
        "status": "complete", "backend_run_dir": str(backend),
        "dev_selected_parameters": summary.get("selection", {}),
        "normalization": "All hypotheses were rescored with report_entity_backend_comparison.norm/score (ZhITN, OpenCC t2s, NFKC, casefold, remove Unicode P/Z/C; clipped canonical mention counts and normalized character CER). No stored percentages were mixed.",
        "baseline_identity_checks": {
            "cached_qwen_source": str(qwen_rows[next(iter(sorted(test_ids)))].get("source", "")),
            "old_scene_common_ids_exactly_matched_both_paired_primary_and_saved_baseline": baseline_checks["old_scene_common"],
            "short_finish_common_ids_exactly_matched_both_paired_primary_and_saved_baseline": baseline_checks["short_finish_common"],
        },
        "old_scene_test_1829": {
            "sample_count": len(test_ids), "old_scene_full_unique_id_count": len(old_comp),
            "old_scene_test_result_rows_reused_from_prior_runs": old_reused_count,
            "arms": old_arm_summaries, "comparisons_vs_existing_review": old_pairs,
            "comparison_scope": "Existing scene-evaluation entity_audio_raw output versus cached Qwen and final selected nativehotword/hybrid on exactly the 1829 backend test IDs.",
        },
        "short_finish_test_230": {
            "short_finish_total_existing_ids": len(short_ids), "short_finish_test_intersection": len(short_test_ids),
            "short_finish_dev_intersection": len(short_ids - short_test_ids),
            "expected_short_finish_test_intersection": EXPECTED_SHORT_TEST,
            "arms": short_arm_summaries, "all_pairwise_comparisons": short_pairs,
            "comparison_scope": "Only the existing 230 short-finish test IDs; includes cached Qwen, current short-finish entity_audio_raw output, final selected nativehotword/hybrid, and older scene entity_audio_raw. This partial subset is not representative of all 1829 test samples.",
        },
        "observed_review_timings": {
            "old_scene_test_1829": old_scene_timing,
            "short_finish_test_230": short_timing,
            "comparison_warning": "These are observed durations recorded by the earlier Web/entity-review evaluations. They describe that path's recorded scope; they are reported separately and are not native CPU compute or concurrent end-to-end backend latency.",
        },
        "limits": ["The short-finish run remains partial: 315 total IDs (85 dev and 230 test), so its test subset cannot represent the full 1829-sample test set.",
                  "Stored historical score percentages are not reused; all compared arms are rescored from their available texts with one normalization/scoring implementation.",
                  "Existing module timing values are retained with their recorded time scope and must not be compared as native inference latency."],
    }
    (backend / "existing_review_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Existing entity review comparison", "", "All text arms were rescored with the backend report's shared normalizer/scorer. The cached Qwen text was required to match the old evaluation's `paired_primary` and `saved_baseline` exactly for every comparison ID.",
             "", "## Existing scene review versus backend test", "", f"The existing scene review has {len(old_comp)} IDs; this comparison uses its exact intersection with the full backend test set ({len(test_ids)} IDs). `existing_review` is the historical `entity_audio_raw` output.",
             "", "| Arm | Samples | Entity hits / mentions | Recall | CER |", "|---|---:|---:|---:|---:|"]
    for arm, metrics in old_arm_summaries.items():
        lines.append(f"| {arm} | {metrics['samples']} | {metrics['entity_hits']} / {metrics['gold_mentions']} | {fmt(metrics['entity_recall'])} | {fmt(metrics['cer'])} |")
    lines += ["", "Paired differences against `existing_review` (candidate minus review):", "", "| Candidate | Recovered | Lost | Entity recall difference | CER difference | CER improved / degraded samples |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ("qwen", "nativehotword", "hybrid"):
        pair = old_pairs[f"{arm}_vs_existing_review"]["overall"]
        lines.append(f"| {arm} | {pair['entity_recovered_mentions']} | {pair['entity_lost_mentions']} | {fmt(pair['entity_recall_difference'])} | {fmt(pair['cer_net_difference'])} | {pair['cer_improved_samples']} / {pair['cer_degraded_samples']} |")
    lines += ["", "## Short-finish shared test subset", "", f"The short-finish module has {len(short_ids)} recorded comparisons overall, of which {len(short_test_ids)} overlap the backend test set ({len(short_ids-short_test_ids)} are dev). The table below is limited to those {len(short_test_ids)} test IDs and must not be generalized to all 1829 test examples.",
              "", "| Arm | Samples | Entity hits / mentions | Recall | CER |", "|---|---:|---:|---:|---:|---:|"]
    for arm, metrics in short_arm_summaries.items():
        lines.append(f"| {arm} | {metrics['samples']} | {metrics['entity_hits']} / {metrics['gold_mentions']} | {fmt(metrics['entity_recall'])} | {fmt(metrics['cer'])} |")
    lines += ["", "## Recorded entity-review timings", "", "Timing fields from historical `results.jsonl` are summarized in the JSON under `observed_review_timings`. Module review elapsed/wall fields and whole Web request totals retain their recorded scope. Replayed-audio seconds are audio duration, not elapsed time. These values are not native CPU timing or concurrent backend latency.",
              "", "The short-finish measurements represent only the 230 common test samples. The older scene timing sample may include rows marked `reused_from`; the JSON reports their count.", ""]
    (backend / "EXISTING_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2%}"


if __name__ == "__main__":
    raise SystemExit(main())
