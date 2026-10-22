#!/usr/bin/env python3
"""Offline paired scorer for the 2026-10-14 entity challenge.

Inference is expected to run without references. This script is the separate,
offline scoring step and writes reports only beneath --run-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from system.zh_itn_bridge import ZhITNNormalizer

DATASET = ROOT / "results/entity_challenge_20261014_scene_v2/dataset"
EXPECTED = 2000
OPENCC = OpenCC("t2s")
ITN = ZhITNNormalizer(ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")


@lru_cache(maxsize=100000)
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", OPENCC.convert(ITN.normalize(text).text)).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] not in {"P", "Z", "C"})


def levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def cer(ref: str, hyp: str) -> tuple[int, int, float]:
    a, b = list(normalize(ref)), list(normalize(hyp))
    errors = levenshtein(a, b)
    return errors, len(a), errors / len(a) if a else (0.0 if not errors else 1.0)


def entity_counter(text: str, canonicals: list[str]) -> Counter[str]:
    haystack = normalize(text)
    counts: Counter[str] = Counter()
    for canonical in canonicals:
        needle = normalize(canonical)
        if needle:
            counts[canonical] = haystack.count(needle)
    return counts


def score_entities(text: str, gold: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_gold = Counter(normalize(str(e.get("canonical") or e.get("surface") or "")) for e in gold)
    normalized_gold.pop("", None)
    predicted = entity_counter(text, list(normalized_gold))
    hits = sum(min(n, predicted[k]) for k, n in normalized_gold.items())
    expected = sum(normalized_gold.values())
    predicted_n = sum(predicted.values())
    return {"hits": hits, "expected": expected, "predicted_gold_lexicon_mentions": predicted_n,
            "missed": dict(normalized_gold - predicted), "extra": dict(predicted - normalized_gold),
            "recall": hits / expected if expected else 0.0}


def rate(errors: int, units: int) -> float:
    return errors / units if units else (0.0 if errors == 0 else 1.0)


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"samples": len(items), "entity_hits": 0, "entity_expected": 0,
              "cer_errors": 0, "cer_reference_chars": 0}
    for x in items:
        totals["entity_hits"] += x["entity"]["hits"]
        totals["entity_expected"] += x["entity"]["expected"]
        totals["cer_errors"] += x["cer_errors"]
        totals["cer_reference_chars"] += x["cer_reference_chars"]
    totals["entity_recall"] = rate(totals["entity_hits"], totals["entity_expected"])
    totals["cer"] = rate(totals["cer_errors"], totals["cer_reference_chars"])
    return totals


def get_path(obj: Any, *path: str, default: Any = None) -> Any:
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def jsonl(path: Path, *, ignore_incomplete_tail: bool = False) -> list[dict[str, Any]]:
    rows = []
    with path.open("rb") as f:
        data = f.read()
    if ignore_incomplete_tail and data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1] if b"\n" in data else b""
    for line_no, raw in enumerate(data.splitlines(), 1):
        line = raw.decode("utf-8")
        if line.strip():
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("row is not an object")
                rows.append(item)
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def add_text_score(ref: str, text: str, gold: list[dict[str, Any]]) -> dict[str, Any]:
    e, n, c = cer(ref, text)
    return {"text": text, "entity": score_entities(text, gold),
            "cer_errors": e, "cer_reference_chars": n, "cer": c}


@lru_cache(maxsize=128)
def load_library_names(path: str) -> tuple[str, ...]:
    import sqlite3
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return tuple(str(row[0]) for row in conn.execute(
            "SELECT DISTINCT canonical_text FROM protected_entities WHERE enabled=1"))


def audit_names(after: str, before: str | None, changes: Any, library: Path | None, ref: str) -> list[str]:
    if not library or not library.is_file() or before is None:
        return []
    h, b, r = normalize(after), normalize(before), normalize(ref)
    candidates: set[str] = set()
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("canonical", "canonical_text", "canonical_name"):
                if value.get(key): candidates.add(str(value[key]))
            for v in value.values(): visit(v)
        elif isinstance(value, list):
            for v in value: visit(v)
    visit(changes)
    if isinstance(changes, list) and changes:
        for change in changes:
            if isinstance(change, dict):
                for key in ("observed", "before", "source", "old_text"):
                    if change.get(key): candidates.add(str(change[key]))
    # Only use that sample's library. Metadata usually identifies the changed canonical;
    # scanning the library is a safe fallback for formats that omit it.
    if not candidates:
        candidates.update(load_library_names(str(library.resolve())))
    allowed = set(load_library_names(str(library.resolve())))
    return sorted({name for name in candidates if name in allowed
                   if (needle := normalize(name)) and h.count(needle) > b.count(needle) and needle not in r})


def mention_change(before: str, after: str, gold: list[dict[str, Any]]) -> tuple[int, int]:
    gold_counts: Counter[str] = Counter(normalize(str(e.get("canonical") or e.get("surface") or "")) for e in gold)
    gold_counts.pop("", None)
    old, new = normalize(before), normalize(after)
    recovered = lost = 0
    for key, n in gold_counts.items():
        delta = min(n, new.count(key)) - min(n, old.count(key))
        recovered += max(0, delta)
        lost += max(0, -delta)
    return recovered, lost


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path, help="directory containing results.jsonl; reports are written here")
    ap.add_argument("--dataset", type=Path, default=DATASET, help="frozen dataset to score; defaults to the original challenge")
    ap.add_argument("--allow-partial", action="store_true", help="write explicitly partial reports when outputs are missing")
    args = ap.parse_args()
    dataset = args.dataset.resolve()
    refs_file, base_file, sample_file = dataset / "references.json", dataset / "baseline_raw.jsonl", dataset / "samples.jsonl"
    refs_doc = json.loads(refs_file.read_text(encoding="utf-8"))
    references = refs_doc.get("samples") if isinstance(refs_doc, dict) else None
    if not isinstance(references, dict):
        raise ValueError("references.json must have a top-level samples object keyed by sample id")
    sample_rows_raw = jsonl(sample_file)
    samples = {str(x["id"]): x for x in sample_rows_raw}
    baseline_rows = jsonl(base_file)
    baseline = {str(x["id"]): x for x in baseline_rows}
    run_rows = jsonl(args.run_dir / "results.jsonl", ignore_incomplete_tail=True)
    run: dict[str, dict[str, Any]] = {}
    duplicate_run: list[str] = []
    for row in run_rows:
        sid = str(row.get("id", ""))
        if sid in run:
            duplicate_run.append(sid)
        run[sid] = row
    expected_ids = set(references)
    issues = {
        "reference_count": len(references), "dataset_sample_count": len(samples), "dataset_sample_row_count": len(sample_rows_raw),
        "baseline_count": len(baseline), "baseline_row_count": len(baseline_rows),
        "run_row_count": len(run_rows), "run_unique_count": len(run), "duplicate_run_ids": sorted(set(duplicate_run)),
        "missing_reference_ids": sorted(set(samples) - expected_ids),
        "missing_dataset_ids": sorted(expected_ids - set(samples)),
        "missing_baseline_ids": sorted(expected_ids - set(baseline)),
        "extra_baseline_ids": sorted(set(baseline) - expected_ids),
        "missing_run_ids": sorted(expected_ids - set(run)), "extra_run_ids": sorted(set(run) - expected_ids),
    }
    dataset_ok = (len(expected_ids) == EXPECTED and len(samples) == EXPECTED and len(baseline) == EXPECTED
                  and len(sample_rows_raw) == EXPECTED and len(baseline_rows) == EXPECTED
                  and not issues["missing_reference_ids"] and not issues["missing_dataset_ids"]
                  and not issues["missing_baseline_ids"] and not issues["extra_baseline_ids"])
    if not dataset_ok:
        raise ValueError(f"Fixed dataset integrity check failed: {json.dumps(issues, ensure_ascii=False)}")
    if issues["extra_run_ids"] or issues["duplicate_run_ids"]:
        raise ValueError("run output has duplicate or non-dataset IDs; these are always rejected: "
                         + json.dumps({k: issues[k] for k in ("extra_run_ids", "duplicate_run_ids")}, ensure_ascii=False))
    partial = bool(issues["missing_run_ids"])
    if partial and not args.allow_partial:
        raise ValueError("run output must contain exactly 2000 unique dataset IDs; pass --allow-partial for explicitly partial reports: "
                         + json.dumps({k: issues[k] for k in ("missing_run_ids", "extra_run_ids", "duplicate_run_ids")}, ensure_ascii=False))

    arms: dict[str, list[dict[str, Any]]] = {k: [] for k in ("saved_baseline", "paired_primary", "entity_audio_raw", "saved_baseline_rules", "entity_rules_auto")}
    comparisons: list[dict[str, Any]] = []
    domain_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    sample_rows: list[dict[str, Any]] = []
    drift = {"comparable": 0, "raw_text_different": 0, "cer_errors_changed": 0, "entity_hits_changed": 0}
    audio = {"samples_with_check": 0, "checks": 0, "samples_with_changes": 0, "changes": 0, "seconds": 0.0,
             "recovered_mentions": 0, "lost_mentions": 0}
    incremental_totals = Counter()
    module_errors: list[dict[str, Any]] = []
    false_edit_candidates: list[dict[str, Any]] = []

    for sid in sorted(expected_ids & set(run)):
        ref_obj, row = references[sid], run[sid]
        sample = samples[sid]
        reference = str(ref_obj.get("reference", sample.get("reference", "")))
        gold = ref_obj.get("gold_entities", sample.get("gold_entities", []))
        if not isinstance(gold, list):
            raise ValueError(f"gold_entities for {sid} must be an array")
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        repair = result.get("entity_audio_repair") if isinstance(result.get("entity_audio_repair"), dict) else {}
        primary = get_path(repair, "evaluation_backend", "primary_finish_text")
        if primary is None:
            primary = get_path(result, "evaluation_backend", "primary_finish_text")
        if primary is None:
            raise ValueError(f"Missing evaluation_backend.primary_finish_text for {sid}")
        texts = {
            "saved_baseline": str(baseline[sid].get("raw_text", "")),
            "paired_primary": str(primary),
            "entity_audio_raw": str(result.get("raw_text", "")),
            "saved_baseline_rules": result.get("saved_baseline_rules_text", row.get("saved_baseline_rules_text")),
            "entity_rules_auto": result.get("entity_rules_auto_text", row.get("entity_rules_auto_text")),
        }
        scored = {arm: add_text_score(reference, text, gold) for arm, text in texts.items() if text is not None}
        for arm in arms:
            if arm in scored:
                arms[arm].append(scored[arm])
                domain_rows[str(row.get("domain") or sample.get("domain") or "unknown")][arm].append(scored[arm])
        if primary is not None:
            drift["comparable"] += 1
            drift["raw_text_different"] += texts["paired_primary"] != texts["saved_baseline"]
            drift["cer_errors_changed"] += scored["paired_primary"]["cer_errors"] != scored["saved_baseline"]["cer_errors"]
            drift["entity_hits_changed"] += scored["paired_primary"]["entity"]["hits"] != scored["saved_baseline"]["entity"]["hits"]

        stored = baseline[sid].get("score", {})
        saved = scored["saved_baseline"]["entity"]
        baseline_checks = {"entity_hits_match": saved["hits"] == stored.get("hits"),
                           "entity_expected_match": saved["expected"] == stored.get("expected_mentions"),
                           "cer_errors_match": scored["saved_baseline"]["cer_errors"] == stored.get("errors"),
                           "cer_reference_chars_match": scored["saved_baseline"]["cer_reference_chars"] == stored.get("reference_chars")}
        if not all(baseline_checks.values()):
            raise ValueError(f"Recomputed baseline differs from stored baseline score for {sid}: {baseline_checks}; stored={stored}")

        check = get_path(repair, "evaluation_backend", "checks")
        if check is None:
            check = repair.get("checks", [])
        changes = repair.get("changes", [])
        if isinstance(check, int):
            audio["samples_with_check"] += int(check > 0)
            audio["checks"] += check
        elif isinstance(check, (list, dict)) and check:
            audio["samples_with_check"] += 1
            audio["checks"] += len(check)
        if isinstance(changes, (list, dict)) and changes:
            audio["samples_with_changes"] += 1
            audio["changes"] += len(changes)
        timing = get_path(row, "timing", "total_seconds", default=0)
        try:
            if math.isfinite(float(timing)):
                audio["seconds"] += float(timing)
        except (TypeError, ValueError):
            pass
        backend = repair.get("evaluation_backend", {})
        for key in ("replay_pcm_seconds", "replay_wall_seconds"):
            try: audio[key] = audio.get(key, 0.0) + float(backend.get(key, 0) or 0)
            except (TypeError, ValueError): pass
        incremental_errors = get_path(repair, "incremental", "errors", default=0)
        if isinstance(incremental_errors, list):
            audio["incremental_errors"] = audio.get("incremental_errors", 0) + len(incremental_errors)
        else:
            try: audio["incremental_errors"] = audio.get("incremental_errors", 0) + int(incremental_errors or 0)
            except (TypeError, ValueError): pass
        err = row.get("module_errors", result.get("module_errors", row.get("error", result.get("error"))))
        rules_error = result.get("entity_rules_error", row.get("entity_rules_error"))
        if rules_error:
            err = {"module_error": err, "entity_rules_error": rules_error} if err else {"entity_rules_error": rules_error}
        if err:
            module_errors.append({"id": sid, "errors": err})

        b, p, raw = (scored[x] for x in ("saved_baseline", "paired_primary", "entity_audio_raw"))
        comparison = {"id": sid, "domain": row.get("domain") or sample.get("domain"), "library_path": row.get("library_path"),
                      "reference": reference, "gold_entities": gold, "arms": scored,
                      "paired_primary_vs_saved": {"entity_hit_delta": p["entity"]["hits"] - b["entity"]["hits"],
                                                   "cer_error_delta": p["cer_errors"] - b["cer_errors"],
                                                   "raw_text_different": texts["paired_primary"] != texts["saved_baseline"]},
                      "audio_repair": {"checks": check, "changes": changes, "timing_total_seconds": timing},
                      "module_error": err}
        comparison["baseline_recomputed_matches_stored"] = baseline_checks
        comparisons.append(comparison)
        mention_delta = raw["entity"]["hits"] - p["entity"]["hits"]
        cer_delta = raw["cer_errors"] - p["cer_errors"]
        recovered, lost = mention_change(texts["paired_primary"], texts["entity_audio_raw"], gold)
        audio["recovered_mentions"] += recovered
        audio["lost_mentions"] += lost
        rules = scored.get("entity_rules_auto")
        rules_delta = (rules["cer_errors"] - b["cer_errors"]) if rules else None
        lib_value = row.get("library_path") or result.get("library_path")
        library = Path(str(lib_value)) if lib_value else None
        if library and not library.is_file():
            candidate = dataset / "libraries" / library.name
            library = candidate if candidate.is_file() else library
        stage_audits = []
        repair_changes = repair.get("changes", [])
        entity_rule_changes = result.get("entity_rules_changes", result.get("rule_changes", []))
        saved_rule_changes = result.get("saved_baseline_rules_changes", [])
        all_names: list[str] = []
        for stage, before_text, after_text, change_list in (
                ("audio_raw_vs_paired_primary", texts["paired_primary"], texts["entity_audio_raw"], repair_changes),
                ("rules_auto_vs_audio_raw", texts["entity_audio_raw"], texts["entity_rules_auto"], entity_rule_changes),
                ("saved_rules_vs_saved_baseline", texts["saved_baseline"], texts["saved_baseline_rules"], saved_rule_changes)):
            if after_text is not None:
                names = audit_names(after_text, before_text, change_list, library, reference)
                if names:
                    all_names.extend(names)
                    stage_audits.append({"stage": stage, "unsupported_added_canonical_names": names})
        if stage_audits:
            false_edit_candidates.append({"id": sid, "domain": comparison["domain"], "stages": stage_audits,
                                          "note": "Conservative automatic candidates; no human review claimed."})

        inc = repair.get("incremental", {})
        for key in ("detections", "candidates", "low_score_rejected", "ambiguous_rejected", "oversized_rejected",
                    "budget_rejected", "queue_rejected", "unmapped", "late_candidates_kept", "completed_windows",
                    "cache_reused"):
            try: incremental_totals[key] += int(inc.get(key, 0) or 0)
            except (TypeError, ValueError): pass
        sample_rows.append({"id": sid, "domain": comparison["domain"],
            "saved_baseline_entity_hits": b["entity"]["hits"], "saved_baseline_entity_expected": b["entity"]["expected"], "saved_baseline_cer": b["cer"],
            "paired_primary_entity_hits": p["entity"]["hits"], "paired_primary_entity_expected": p["entity"]["expected"], "paired_primary_cer": p["cer"],
            "entity_audio_raw_entity_hits": raw["entity"]["hits"], "entity_audio_raw_cer": raw["cer"],
            "saved_baseline_rules_entity_hits": scored.get("saved_baseline_rules", {}).get("entity", {}).get("hits"), "saved_baseline_rules_cer": scored.get("saved_baseline_rules", {}).get("cer"),
            "entity_rules_auto_entity_hits": rules["entity"]["hits"] if rules else None, "entity_rules_auto_cer": rules["cer"] if rules else None,
            "audio_entity_hit_delta": mention_delta, "audio_cer_error_delta": cer_delta,
            "audio_recovered_mentions": recovered, "audio_lost_mentions": lost,
            "audio_entity_improved": mention_delta > 0, "audio_entity_degraded": mention_delta < 0,
            "audio_cer_improved": cer_delta < 0, "audio_cer_degraded": cer_delta > 0,
            "rules_vs_saved_baseline_cer_error_delta": rules_delta,
            "rules_unsupported_canonical_names": "|".join(all_names)})

    summaries = {arm: aggregate(arms[arm]) for arm in arms}
    domain_summary = {domain: {arm: aggregate(vals[arm]) for arm in arms} for domain, vals in sorted(domain_rows.items())}
    def write_json(name: str, value: Any) -> None:
        (args.run_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": str(dataset), "partial": partial, "scored_unique_samples": len(comparisons), "expected_samples": EXPECTED,
               "arms": summaries, "paired_changes": {
                   "primary_vs_saved_drift": drift,
                   "entity_audio_raw_vs_primary_entity_improved_samples": sum(x["audio_entity_improved"] for x in sample_rows),
                   "entity_audio_raw_vs_primary_entity_degraded_samples": sum(x["audio_entity_degraded"] for x in sample_rows),
                   "entity_audio_raw_vs_primary_cer_improved_samples": sum(x["audio_cer_improved"] for x in sample_rows),
                   "entity_audio_raw_vs_primary_cer_degraded_samples": sum(x["audio_cer_degraded"] for x in sample_rows),
                   "entity_rules_auto_coverage_samples": len(arms["entity_rules_auto"]),
                   "entity_rules_auto_unavailable_samples": len(comparisons) - len(arms["entity_rules_auto"]),
                   "entity_rules_auto_vs_saved_cer_improved_samples": sum((x["rules_vs_saved_baseline_cer_error_delta"] or 0) < 0 for x in sample_rows if x["rules_vs_saved_baseline_cer_error_delta"] is not None),
                   "entity_rules_auto_vs_saved_cer_degraded_samples": sum(x["rules_vs_saved_baseline_cer_error_delta"] > 0 for x in sample_rows if x["rules_vs_saved_baseline_cer_error_delta"] is not None)},
               "audio_review": audio, "incremental_totals": dict(incremental_totals), "module_error_count": len(module_errors), "false_edit_audit": {
                   "candidate_sample_count": len(false_edit_candidates), "candidates": false_edit_candidates,
               "method": "For each sample and each stage, names come only from row.library_path. Prefer canonical values recorded in changes/rule_changes; fallback scans that sample's library. Listed only when the stage output adds an occurrence relative to its before text and the canonical is absent from the standard reference. Conservative automatic candidates; no human review claimed."},
               "integrity": issues}
    write_json("summary.json", summary)
    write_json("scoring_verification.json", {"status": "partial" if partial else "complete", "dataset_integrity": dataset_ok,
         "expected_unique_ids": EXPECTED, "scored_unique_ids": len(comparisons), "duplicate_run_ids": issues["duplicate_run_ids"],
         "missing_ids": issues["missing_run_ids"], "extra_ids": issues["extra_run_ids"],
         "references_schema": "references.json: {schema_version, samples: {id: {reference, gold_entities}}}",
         "normalization": "ZhITNNormalizer(ROOT/zh-itn/zh-itn/build/libzh_itn_bridge.so), OpenCC(t2s), NFKC, casefold, remove Unicode P/Z/C",
         "metric": "Gold canonical mentions normalized then merged; count occurrences with str.count (non-overlapping) and clip hits by the gold count. CER is character edit distance on normalized text."})
    with (args.run_dir / "domain_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["domain", "arm", "samples", "entity_hits", "entity_expected", "entity_recall", "cer_errors", "cer_reference_chars", "cer"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for domain, armvals in domain_summary.items():
            for arm, values in armvals.items(): w.writerow({"domain": domain, "arm": arm, **values})
    with (args.run_dir / "sample_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0]) if sample_rows else ["id", "domain"]); w.writeheader(); w.writerows(sample_rows)
    with (args.run_dir / "comparisons.jsonl").open("w", encoding="utf-8") as f:
        for item in comparisons: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    primary_recall = summaries["paired_primary"]["entity_recall"]
    audio_recall = summaries["entity_audio_raw"]["entity_recall"]
    recall_pp = (audio_recall - primary_recall) * 100
    report = ["# 实体挑战评分报告", "", f"状态：**{'部分结果' if partial else '完整结果'}**。本次评分 {len(comparisons)}/{EXPECTED} 个唯一 ID。",
              "", "这是离线评分。文本归一化顺序为 ZhITNNormalizer、OpenCC t2s、NFKC、casefold、删除 Unicode P/Z/C。先归一化并合并相同 gold canonical，再以非重叠 str.count 计数，命中数按 gold 次数截顶；CER 为归一化字符串上的字符编辑距离。", "",
              "| 评分臂 | 覆盖样本 | 实体命中 / gold | 实体召回率 | CER |", "|---|---:|---:|---:|---:|"]
    arm_labels = {"saved_baseline": "保存的基线 ASR", "paired_primary": "配对首轮 ASR", "entity_audio_raw": "实体音频修复结果",
                  "saved_baseline_rules": "保存基线的 prepare_entity_segment 规则诊断", "entity_rules_auto": "原音修复后的 prepare_entity_segment 规则诊断"}
    for arm, s in summaries.items(): report.append(f"| {arm_labels.get(arm, arm)} | {s['samples']} | {s['entity_hits']} / {s['entity_expected']} | {s['entity_recall']:.2%} | {s['cer']:.2%} |")
    report += ["", f"音频修复配对首轮结果：实体召回率从 {primary_recall:.2%} 到 {audio_recall:.2%}，变化 {recall_pp:+.2f} 个百分点；按 gold 次数截顶后新增恢复 {audio['recovered_mentions']} 个提及、丢失 {audio['lost_mentions']} 个提及。改善/退化样本数分别为实体 {summary['paired_changes']['entity_audio_raw_vs_primary_entity_improved_samples']}/{summary['paired_changes']['entity_audio_raw_vs_primary_entity_degraded_samples']}、CER {summary['paired_changes']['entity_audio_raw_vs_primary_cer_improved_samples']}/{summary['paired_changes']['entity_audio_raw_vs_primary_cer_degraded_samples']}。",
               f"覆盖：音频主臂 {summaries['entity_audio_raw']['samples']}/{len(comparisons)}；保存规则诊断 {summaries['saved_baseline_rules']['samples']}/{len(comparisons)}；当前规则诊断 {summaries['entity_rules_auto']['samples']}/{len(comparisons)}。规则诊断结果来自 prepare_entity_segment 输出，仅作独立规则诊断；未运行 Refiner 或完整 Web，不能视为完整流程质量。",
               f"primary 与保存 ASR 文本不同的样本 {drift['raw_text_different']}/{drift['comparable']}。音频复核检查 {audio['checks']} 次（{audio['samples_with_check']} 个样本），修改 {audio['changes']} 次（{audio['samples_with_changes']} 个样本）；row.timing.total_seconds 合计 {audio['seconds']:.2f} 秒。PCM replay 合计 {audio.get('replay_pcm_seconds', 0):.2f} 秒，replay wall 合计 {audio.get('replay_wall_seconds', 0):.2f} 秒；incremental.errors 合计 {audio.get('incremental_errors', 0)}。模块错误记录 {len(module_errors)}。",
               "", "增量音频复核计数：" + "、".join(f"{k}={v}" for k, v in sorted(incremental_totals.items())),
               f"分阶段保守误改候选：{len(false_edit_candidates)} 个样本。候选基于输出新增的库 canonical 且标准 reference 不支持；这只是自动筛查，没有人工复核结论。", "",
               "详细结果见 `domain_summary.csv`、`sample_comparison.csv`、`comparisons.jsonl`、`summary.json` 和 `scoring_verification.json`。", ""]
    (args.run_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
