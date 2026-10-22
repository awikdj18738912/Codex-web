#!/usr/bin/env python3
"""Compare the full short-finish evaluation with the previous entity run.

This is an offline reporting step. It consumes scorer-produced comparisons and
collector results; it does not run ASR or modify either evaluation directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OLD_DIR = ROOT / "results/entity_challenge_scene_eval_20261014_v2"
DEFAULT_NEW_DIR = ROOT / "results/entity_short_finish_20261015_v1/evaluation"
EXPECTED = 2000
sys.path.insert(0, str(ROOT))

try:
    from opencc import OpenCC
    from system.zh_itn_bridge import ZhITNNormalizer

    _OPENCC = OpenCC("t2s")
    _ITN = ZhITNNormalizer(ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")

    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", _OPENCC.convert(_ITN.normalize(text).text)).casefold()
        return "".join(c for c in text if unicodedata.category(c)[0] not in {"P", "Z", "C"})
except Exception as exc:  # Keep a clear failure rather than silently scoring differently.
    _NORMALIZE_IMPORT_ERROR = exc

    def normalize(text: str) -> str:
        raise RuntimeError("The scorer's OpenCC/ZhITN normalization dependencies are required") from _NORMALIZE_IMPORT_ERROR


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_incomplete_tail: bool = False) -> list[dict[str, Any]]:
    rows = []
    raw = path.read_bytes()
    physical = raw.splitlines(keepends=True)
    for line_no, item in enumerate(physical, 1):
        complete = item.endswith((b"\n", b"\r"))
        try:
            line = item.decode("utf-8")
        except UnicodeDecodeError:
            if allow_incomplete_tail and line_no == len(physical) and not complete:
                break
            raise ValueError(f"Invalid UTF-8 in JSONL at {path}:{line_no}")
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if allow_incomplete_tail and line_no == len(physical) and not complete:
                break
            raise ValueError(f"Invalid JSONL at {path}:{line_no}")
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        rows.append(value)
    return rows


def unique_index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    duplicates = []
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id:
            raise ValueError(f"Missing id in {label}")
        if sample_id in result:
            duplicates.append(sample_id)
        result[sample_id] = row
    if duplicates:
        raise ValueError(f"Duplicate IDs in {label}: {duplicates[:10]} ({len(duplicates)} total)")
    return result


def gold_counter(row: dict[str, Any]) -> Counter[str]:
    gold = row.get("gold_entities") or []
    counts: Counter[str] = Counter()
    for entity in gold:
        canonical = normalize(str(entity.get("canonical") or entity.get("surface") or ""))
        if canonical:
            counts[canonical] += 1
    return counts


def gold_mentions(text: str, gold: Counter[str]) -> Counter[str]:
    normalized_text = normalize(text)
    return Counter({term: min(count, normalized_text.count(term)) for term, count in gold.items()})


def get_metrics(row: dict[str, Any], arm: str) -> dict[str, Any]:
    item = (row.get("arms") or {}).get(arm)
    if not isinstance(item, dict):
        raise ValueError(f"Comparison row {row.get('id')} lacks arm {arm}")
    entity = item.get("entity") or {}
    expected = int(entity.get("expected", 0))
    hits = int(entity.get("hits", 0))
    errors = int(item.get("cer_errors", 0))
    chars = int(item.get("cer_reference_chars", 0))
    return dict(hits=hits, mentions=expected, recall=hits / expected if expected else 0.,
                cer_errors=errors, cer_reference_chars=chars,
                cer=errors / chars if chars else (0. if not errors else 1.))


def sum_metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    values = [get_metrics(row, arm) for row in rows]
    hits = sum(x["hits"] for x in values)
    mentions = sum(x["mentions"] for x in values)
    errors = sum(x["cer_errors"] for x in values)
    chars = sum(x["cer_reference_chars"] for x in values)
    return dict(samples=len(rows), hits=hits, mentions=mentions,
                recall=hits / mentions if mentions else 0., cer_errors=errors,
                cer_reference_chars=chars, cer=errors / chars if chars else (0. if not errors else 1.))


def sample_group(seconds: float) -> str:
    if seconds <= 4.:
        return "<=4秒"
    if seconds <= 8.:
        return "(4,8]秒"
    return ">8秒"


def get_incremental(result_row: dict[str, Any]) -> dict[str, Any]:
    result = result_row.get("result") or {}
    repair = result.get("entity_audio_repair") or {}
    return repair.get("incremental") or {}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return dict(count=0, sum=0., mean=None, p50=None, p95=None, max=None)
    return dict(count=len(values), sum=sum(values), mean=statistics.fmean(values),
                p50=percentile(values, .5), p95=percentile(values, .95), max=max(values))


def short_finish_stats(result_rows: dict[str, dict[str, Any]], sample_ids: list[str]) -> dict[str, Any]:
    incrementals = {sample_id: get_incremental(result_rows[sample_id]) if sample_id in result_rows else {}
                    for sample_id in sample_ids}
    counter_names = ("short_finish_checks", "short_finish_changes", "short_finish_timed_out",
                     "short_finish_errors", "short_finish_wall_seconds",
                     "short_finish_candidates", "short_finish_replay_seconds")
    counters = {key: sum(float(item.get(key, 0) or 0) for item in incrementals.values())
                for key in counter_names}
    configs = {}
    for key in ("short_finish_max_seconds", "short_finish_wait_seconds"):
        values = sorted({float(item[key]) for item in incrementals.values()
                         if isinstance(item.get(key), (int, float))})
        configs[key] = values[0] if len(values) == 1 else values
    checks = {sample_id for sample_id, item in incrementals.items()
              if float(item.get("short_finish_checks", 0) or 0) > 0}
    walls_all = [float(item.get("short_finish_wall_seconds", 0) or 0) for item in incrementals.values()]
    walls_checked = [float(incrementals[sample_id].get("short_finish_wall_seconds", 0) or 0)
                     for sample_id in checks]
    result_coverage = sum(sample_id in result_rows for sample_id in sample_ids)
    return dict(sample_count=len(sample_ids), result_coverage_samples=result_coverage,
                finalcheck_coverage_samples=len(checks),
                timed_out_samples=sum(float(x.get("short_finish_timed_out", 0) or 0) > 0 for x in incrementals.values()),
                error_samples=sum(float(x.get("short_finish_errors", 0) or 0) > 0 for x in incrementals.values()),
                counters=counters, configuration=configs,
                wall_seconds_all_samples=distribution(walls_all),
                wall_seconds_checks_gt_zero=distribution(walls_checked))


def false_audit(summary: dict[str, Any]) -> dict[str, Any]:
    audit = summary.get("false_edit_audit") or {}
    candidates = audit.get("candidates") or []
    ids = sorted({str(x.get("id", "")) for x in candidates if x.get("id")})
    return dict(candidate_sample_count=int(audit.get("candidate_sample_count", len(ids)) or 0),
                candidate_count=len(candidates), candidate_sample_ids=ids,
                method=audit.get("method"))


def make_sample_rows(ids: list[str], old_comp: dict[str, dict[str, Any]],
                     new_comp: dict[str, dict[str, Any]], old_results: dict[str, dict[str, Any]],
                     new_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for sample_id in ids:
        old, new = old_comp[sample_id], new_comp[sample_id]
        gold = gold_counter(old)
        old_primary = str((old.get("arms") or {}).get("paired_primary", {}).get("text", ""))
        new_primary = str((new.get("arms") or {}).get("paired_primary", {}).get("text", ""))
        old_text = str((old.get("arms") or {}).get("entity_audio_raw", {}).get("text", ""))
        new_text = str((new.get("arms") or {}).get("entity_audio_raw", {}).get("text", ""))
        old_counts = gold_mentions(old_text, gold)
        new_counts = gold_mentions(new_text, gold)
        recovered = sum(max(0, new_counts[t] - old_counts[t]) for t in gold)
        lost = sum(max(0, old_counts[t] - new_counts[t]) for t in gold)
        old_audio = get_metrics(old, "entity_audio_raw")
        new_audio = get_metrics(new, "entity_audio_raw")
        old_primary_metrics = get_metrics(old, "paired_primary")
        new_primary_metrics = get_metrics(new, "paired_primary")
        new_result = new_results.get(sample_id, {})
        seconds = float((new_result.get("result") or {}).get("entity_audio_repair", {}).get("source_audio_seconds")
                        or new_result.get("audio_seconds") or old_results[sample_id].get("audio_seconds") or 0.)
        new_inc = get_incremental(new_result)
        output.append(dict(id=sample_id, audio_seconds=seconds, duration_group=sample_group(seconds),
                           primary_text_drift=old_primary != new_primary,
                           old_primary_text=old_primary, new_primary_text=new_primary,
                           old_audio_hits=old_audio["hits"], new_audio_hits=new_audio["hits"],
                           entity_hit_delta=new_audio["hits"] - old_audio["hits"],
                           old_audio_mentions=old_audio["mentions"], new_audio_mentions=new_audio["mentions"],
                           old_audio_recall=old_audio["recall"], new_audio_recall=new_audio["recall"],
                           old_audio_cer=old_audio["cer"], new_audio_cer=new_audio["cer"],
                           cer_error_delta=new_audio["cer_errors"] - old_audio["cer_errors"],
                           entity_improved=new_audio["hits"] > old_audio["hits"],
                           entity_degraded=new_audio["hits"] < old_audio["hits"],
                           cer_improved=new_audio["cer_errors"] < old_audio["cer_errors"],
                           cer_degraded=new_audio["cer_errors"] > old_audio["cer_errors"],
                           recovered_mentions=recovered, lost_mentions=lost,
                           primary_old_hits=old_primary_metrics["hits"], primary_new_hits=new_primary_metrics["hits"],
                           short_finish_checks=new_inc.get("short_finish_checks", 0),
                           short_finish_changes=new_inc.get("short_finish_changes", 0),
                           short_finish_timed_out=new_inc.get("short_finish_timed_out", 0),
                           short_finish_errors=new_inc.get("short_finish_errors", 0),
                           short_finish_wall_seconds=new_inc.get("short_finish_wall_seconds", 0.)))
    return output


def paired_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    same = [r for r in rows if not r["primary_text_drift"]]
    return dict(samples=len(same), audio_old=sum_metrics([r["old"] for r in same], "entity_audio_raw") if same else sum_metrics([], "entity_audio_raw"),
                audio_new=sum_metrics([r["new"] for r in same], "entity_audio_raw") if same else sum_metrics([], "entity_audio_raw"))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    old_comp_rows = read_jsonl(args.old_run_dir / "comparisons.jsonl")
    old_result_rows = read_jsonl(args.old_run_dir / "results.jsonl")
    old_summary = read_json(args.old_run_dir / "summary.json")
    new_comp_rows = read_jsonl(args.run_dir / "comparisons.jsonl", allow_incomplete_tail=args.allow_partial)
    new_result_rows = read_jsonl(args.run_dir / "results.jsonl", allow_incomplete_tail=args.allow_partial)
    new_summary = read_json(args.run_dir / "summary.json")

    old_comp = unique_index(old_comp_rows, "old comparisons")
    old_results = unique_index(old_result_rows, "old results")
    new_comp = unique_index(new_comp_rows, "new comparisons")
    new_results = unique_index(new_result_rows, "new results")
    if len(old_comp) != EXPECTED:
        raise ValueError(f"Old run must contain {EXPECTED} unique comparison rows, got {len(old_comp)}")
    if set(old_comp) != set(old_results):
        raise ValueError("Old comparisons/results ID sets differ")
    old_ids = set(old_comp)
    for label, mapping in (("new comparisons", new_comp), ("new results", new_results)):
        extra = set(mapping) - old_ids
        if extra:
            raise ValueError(f"{label} contains IDs absent from old run: {sorted(extra)[:10]}")
    if not args.allow_partial and set(new_comp) != set(new_results):
        raise ValueError("New comparisons/results ID sets differ")
    if not args.allow_partial and (len(new_comp) != EXPECTED or set(new_comp) != old_ids):
        raise ValueError(f"New run must contain the same {EXPECTED} unique IDs; use --allow-partial for an interim common-ID report")
    ids = sorted(old_ids & set(new_comp))
    if not ids:
        raise ValueError("No common sample IDs")
    for sample_id in ids:
        if gold_counter(old_comp[sample_id]) != gold_counter(new_comp[sample_id]):
            raise ValueError(f"Gold annotations differ for {sample_id}")

    sample_rows = make_sample_rows(ids, old_comp, new_comp, old_results, new_results)
    row_by_id = {r["id"]: r for r in sample_rows}
    groups: dict[str, list[str]] = {name: [] for name in ("全量", "<=4秒", "(4,8]秒", ">8秒")}
    for sample_id in ids:
        row = row_by_id[sample_id]
        groups["全量"].append(sample_id)
        groups[row["duration_group"]].append(sample_id)

    metrics = {}
    for group, group_ids in groups.items():
        selected_old = [old_comp[sid] for sid in group_ids]
        selected_new = [new_comp[sid] for sid in group_ids]
        metrics[group] = {"saved_baseline": sum_metrics(selected_new, "saved_baseline"),
                          "old_entity_audio_raw": sum_metrics(selected_old, "entity_audio_raw"),
                          "new_entity_audio_raw": sum_metrics(selected_new, "entity_audio_raw"),
                          "new_paired_primary": sum_metrics(selected_new, "paired_primary")}

    drift_count = sum(r["primary_text_drift"] for r in sample_rows)
    same_rows = [r for r in sample_rows if not r["primary_text_drift"]]
    same_ids = [r["id"] for r in same_rows]
    same_primary = {"samples": len(same_ids),
                    "old_entity_audio_raw": sum_metrics([old_comp[sid] for sid in same_ids], "entity_audio_raw"),
                    "new_entity_audio_raw": sum_metrics([new_comp[sid] for sid in same_ids], "entity_audio_raw")}

    improved = sum(r["entity_improved"] for r in sample_rows)
    degraded = sum(r["entity_degraded"] for r in sample_rows)
    cer_improved = sum(r["cer_improved"] for r in sample_rows)
    cer_degraded = sum(r["cer_degraded"] for r in sample_rows)
    short_stats = short_finish_stats(new_results, ids)
    summary = dict(old_run_dir=str(args.old_run_dir), new_run_dir=str(args.run_dir),
                   partial=len(ids) != EXPECTED,
                   comparison_scope="仅使用旧/新 comparisons 的共同 ID；partial 运行允许 results 与 comparisons 覆盖不同，计数以 comparison 共同 ID 为准",
                   expected_unique_samples=EXPECTED, common_samples=len(ids),
                   old_unique_samples=len(old_ids), new_unique_samples=len(new_comp),
                   new_results_unique_samples=len(new_results),
                   comparison_ids_missing_new_results=len(set(new_comp) - set(new_results)),
                   primary_text_drift_samples=drift_count,
                   primary_text_same_subset=same_primary,
                   metrics_by_duration=metrics,
                   old_to_new_audio_sample_changes=dict(entity_improved=improved, entity_degraded=degraded,
                       cer_improved=cer_improved, cer_degraded=cer_degraded,
                       recovered_gold_mentions=sum(r["recovered_mentions"] for r in sample_rows),
                       lost_gold_mentions=sum(r["lost_mentions"] for r in sample_rows)),
                   short_finish=short_stats,
                   false_edit_audit=dict(old=false_audit(old_summary), new=false_audit(new_summary)),
                   interpretation=dict(full_evaluation=True, same_fixed_known_glossary=True,
                       incremental_wall_is_browser_latency=False, production_service_restarted=False))
    write_outputs(args.run_dir, summary, sample_rows)
    return summary


def write_outputs(run_dir: Path, summary: dict[str, Any], sample_rows: list[dict[str, Any]]) -> None:
    json_path = run_dir / "short_finish_comparison.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = run_dir / "short_finish_sample_comparison.csv"
    fields = list(sample_rows[0]) if sample_rows else ["id"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample_rows)
    md_path = run_dir / "SHORT_FINISH_COMPARISON.md"
    lines = ["# 短音频收尾复核对照", "",
             f"报告状态：{'partial（中期子集）' if summary['partial'] else 'full'}。", "",
             f"旧 run：`{summary['old_run_dir']}`  ", f"新 run：`{summary['new_run_dir']}`  ",
             f"共同样本：{summary['common_samples']}/{EXPECTED}；新旧 primary 文本不同：{summary['primary_text_drift_samples']} 条。",
             "全量比较要求严格 2000 个唯一 ID；`--allow-partial` 只允许新 run 是旧 ID 的子集，仍拒绝重复或额外 ID。", "",
             "全量运行是重新执行的评测，使用同一固定已知词库；它不是独立 blind 测试。primary 文本存在漂移时，下面同时报告相同 primary 文本子集，便于区分复核收益和首轮 ASR 差异。", "",
             "| 时长组 | 样本 | 保存基线命中/提及 | 保存基线召回 | 保存基线 CER | 旧 audio_raw 命中/提及 | 旧召回 | 旧 CER | 新 audio_raw 命中/提及 | 新召回 | 新 CER |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group, item in summary["metrics_by_duration"].items():
        b, old, new = item["saved_baseline"], item["old_entity_audio_raw"], item["new_entity_audio_raw"]
        lines.append(f"| {group} | {new['samples']} | {b['hits']}/{b['mentions']} | {b['recall']:.2%} | {b['cer']:.2%} | {old['hits']}/{old['mentions']} | {old['recall']:.2%} | {old['cer']:.2%} | {new['hits']}/{new['mentions']} | {new['recall']:.2%} | {new['cer']:.2%} |")
    same = summary["primary_text_same_subset"]
    stats = summary["short_finish"]
    lines += ["", f"primary 文本相同的配对子集为 {same['samples']} 条：旧 audio_raw {same['old_entity_audio_raw']['hits']}/{same['old_entity_audio_raw']['mentions']} 命中、CER {same['old_entity_audio_raw']['cer']:.2%}；新 audio_raw {same['new_entity_audio_raw']['hits']}/{same['new_entity_audio_raw']['mentions']} 命中、CER {same['new_entity_audio_raw']['cer']:.2%}。", "",
              "相对旧 audio_raw 的逐样本变化：实体改善/退化样本 {entity_improved}/{entity_degraded}，CER 改善/退化样本 {cer_improved}/{cer_degraded}；gold 逐词截顶后恢复/丢失 {recovered_gold_mentions}/{lost_gold_mentions} 次。".format(**summary["old_to_new_audio_sample_changes"]), "",
              f"新 run 收尾检查覆盖 {stats['finalcheck_coverage_samples']} 条；超时样本 {stats['timed_out_samples']}，错误样本 {stats['error_samples']}。只累计检查/变化/超时/错误/耗时/candidate/replay 计数：`{json.dumps(stats['counters'], ensure_ascii=False)}`；配置值单列：`{json.dumps(stats['configuration'], ensure_ascii=False)}`。新 results 覆盖共同 comparison ID {stats['result_coverage_samples']}/{stats['sample_count']}。", "",
              "收尾耗时分布按 snapshot.incremental.short_finish_wall_seconds 汇总：", "",
              f"- 全部共同样本：sum {stats['wall_seconds_all_samples']['sum']:.3f}s，mean {fmt(stats['wall_seconds_all_samples']['mean'])}s，p50 {fmt(stats['wall_seconds_all_samples']['p50'])}s，p95 {fmt(stats['wall_seconds_all_samples']['p95'])}s，max {fmt(stats['wall_seconds_all_samples']['max'])}s。",
              f"- checks > 0 样本：sum {stats['wall_seconds_checks_gt_zero']['sum']:.3f}s，mean {fmt(stats['wall_seconds_checks_gt_zero']['mean'])}s，p50 {fmt(stats['wall_seconds_checks_gt_zero']['p50'])}s，p95 {fmt(stats['wall_seconds_checks_gt_zero']['p95'])}s，max {fmt(stats['wall_seconds_checks_gt_zero']['max'])}s。", "",
              "false edit audit：旧 run {old} 条候选样本/{old_count} 个候选，新 run {new} 条候选样本/{new_count} 个候选。候选是自动审计结果，不代表人工确认。".format(old=summary["false_edit_audit"]["old"]["candidate_sample_count"], old_count=summary["false_edit_audit"]["old"]["candidate_count"], new=summary["false_edit_audit"]["new"]["candidate_sample_count"], new_count=summary["false_edit_audit"]["new"]["candidate_count"]), "",
              "这里统计的是收尾步骤 wall time，不等于浏览器端用户延迟；正式服务尚未重启。明细见 `short_finish_comparison.json` 和 `short_finish_sample_comparison.csv`。", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-dir", type=Path, default=OLD_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_NEW_DIR)
    parser.add_argument("--allow-partial", action="store_true",
                        help="report on common IDs when the new run is an incomplete subset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    needed = [args.old_run_dir / "comparisons.jsonl", args.old_run_dir / "results.jsonl",
              args.old_run_dir / "summary.json", args.run_dir / "comparisons.jsonl",
              args.run_dir / "results.jsonl", args.run_dir / "summary.json"]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        print("Evaluation inputs not ready; no outputs generated:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2
    summary = build_report(args)
    print(f"Wrote comparison for {summary['common_samples']} common samples under {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
