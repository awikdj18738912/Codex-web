#!/usr/bin/env python3
"""Strict paired report for the actual Qwen + live CTC + audio-review run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.report_entity_backend_comparison import norm, score

BASE = ROOT / "results/entity_backend_comparison_20261015"
OLD = ROOT / "results/entity_short_finish_20261015_v1/evaluation"
RUN = ROOT / "results/entity_ctc_live_integration_20261016"


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def index(rows, label):
    out = {}
    for row in rows:
        sid = str(row.get("id", ""))
        if not sid:
            raise ValueError(f"Missing id in {label}")
        if sid in out:
            raise ValueError(f"Duplicate {label} id: {sid}")
        out[sid] = row
    return out


def exact_ids(rows, expected, label):
    actual = set(rows)
    if actual != expected:
        raise ValueError(f"{label} ID mismatch: missing={sorted(expected-actual)[:10]} "
                         f"extra={sorted(actual-expected)[:10]} "
                         f"(expected {len(expected)}, found {len(actual)})")


def aggregate(items):
    hits = sum(x["entity_hits"] for x in items)
    mentions = sum(x["entity_mentions"] for x in items)
    errors = sum(x["cer_errors"] for x in items)
    chars = sum(x["reference_chars"] for x in items)
    return dict(samples=len(items), entity_hits=hits, entity_mentions=mentions,
                entity_recall=hits / mentions if mentions else 0., cer_errors=errors,
                reference_chars=chars, cer=errors / chars if chars else (0. if not errors else 1.))


def score_texts(ids, refs, texts):
    per_id = {}
    for sid in sorted(ids):
        if sid not in texts:
            raise ValueError(f"Missing hypothesis for {sid}")
        ref = refs[sid]
        per_id[sid] = score(str(ref.get("reference", "")), str(texts[sid]),
                             ref.get("gold_entities", []))
    return aggregate(list(per_id.values())), per_id


def text_from_qwen(row):
    return str((row.get("final_payload") or {}).get("text", ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path, default=BASE)
    ap.add_argument("--old-dir", type=Path, default=OLD)
    ap.add_argument("--run-dir", type=Path, default=RUN)
    args = ap.parse_args()
    base, old, run = args.base_dir.resolve(), args.old_dir.resolve(), args.run_dir.resolve()

    inputs_rows = read_jsonl(base / "inputs.jsonl")
    inputs = index(inputs_rows, "frozen input")
    refs = read_json(base / "references.json").get("samples", {})
    if set(inputs) != set(refs):
        raise ValueError("Frozen input/reference IDs differ")
    short_rows = read_jsonl(old / "comparisons.jsonl")
    short = index(short_rows, "short-finish comparison")
    short_test = {sid for sid in short if sid in inputs and inputs[sid].get("split") == "test"}
    control = {sid for sid, row in inputs.items() if row.get("split") == "control"}
    if len(short_test) != 230 or len(control) != 100:
        raise ValueError(f"Expected frozen cohort 230 short_test + 100 control; found {len(short_test)} + {len(control)}")
    if len(short_rows) != 315:
        raise ValueError(f"Expected 315 old short-finish rows, found {len(short_rows)}")
    test_ids = short_test
    all_ids = test_ids | control

    qwen = index(read_jsonl(base / "qwen_primary_streams.jsonl"), "saved Qwen primary")
    selection = read_json(base / "selection.json")
    bypass = index((row for row in read_jsonl(base / "hybrid_results.jsonl")
        if row.get("minimum_gap") == selection["hybrid_minimum_gap"]
        and row.get("minimum_margin") == selection["hybrid_minimum_margin"]), "selected CTC bypass")
    combo = index(read_jsonl(run / "combination_live_results.jsonl"), "live combination")
    exact_ids(combo, all_ids, "live combination")
    for label, rows in (("Qwen primary", qwen), ("CTC bypass", bypass)):
        missing = all_ids - set(rows)
        if missing:
            raise ValueError(f"{label} missing selected IDs: {sorted(missing)[:10]}")

    frozen_shas = {}
    for sid in all_ids:
        inp = inputs[sid]
        if inp.get("split") != ("test" if sid in test_ids else "control"):
            raise ValueError(f"Split mismatch for {sid}")
        expected_hash = str(inp.get("pcm_sha256", ""))
        if not expected_hash:
            raise ValueError(f"Missing frozen PCM hash for {sid}")
        pcm_path = Path(str(inp.get("pcm_path", "")))
        if not pcm_path.is_file() or hashlib.sha256(pcm_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen input PCM hash mismatch: {sid}")
        frozen_shas[sid] = expected_hash
        for label, row in (("CTC bypass", bypass[sid]), ("combination", combo[sid])):
            if row.get("pcm_sha256") != expected_hash:
                raise ValueError(f"{label} PCM hash mismatch: {sid}")
        qtext = text_from_qwen(qwen[sid])
        for label, row in (("CTC bypass", bypass[sid]), ("combination", combo[sid])):
            actual_primary = row.get("primary_text")
            if actual_primary is None or str(actual_primary) != qtext:
                raise ValueError(f"Actual Qwen raw differs from saved primary text ({label}): {sid}")

    texts = {"qwen_raw": {}, "ctc_bypass": {}, "short_finish": {}, "combination": {}}
    for sid in all_ids:
        texts["qwen_raw"][sid] = text_from_qwen(qwen[sid])
        texts["ctc_bypass"][sid] = str(bypass[sid].get("text", ""))
        if sid in test_ids:
            arms = short[sid].get("arms") or {}
            texts["short_finish"][sid] = str((arms.get("entity_audio_raw") or {}).get("text", ""))
        texts["combination"][sid] = str(combo[sid].get("text", ""))

    test_metrics = {}
    test_per_id = {}
    for arm, arm_texts in texts.items():
        test_metrics[arm], test_per_id[arm] = score_texts(test_ids, refs, arm_texts)
    control_metrics = {}
    for arm in ("qwen_raw", "combination"):
        control_metrics[arm], _ = score_texts(control, refs, texts[arm])

    recovered = lost = cer_improved = cer_degraded = 0
    per_id_delta = {}
    for sid in sorted(test_ids):
        old_s = test_per_id["short_finish"][sid]
        new_s = test_per_id["combination"][sid]
        gold = Counter(old_s["gold_counts"])
        oldn, newn = norm(texts["short_finish"][sid]), norm(texts["combination"][sid])
        rec = sum(max(0, min(n, newn.count(c)) - min(n, oldn.count(c))) for c, n in gold.items())
        los = sum(max(0, min(n, oldn.count(c)) - min(n, newn.count(c))) for c, n in gold.items())
        improved = new_s["cer_errors"] < old_s["cer_errors"]
        degraded = new_s["cer_errors"] > old_s["cer_errors"]
        recovered += rec; lost += los; cer_improved += improved; cer_degraded += degraded
        per_id_delta[sid] = dict(entity_recovered=rec, entity_lost=los,
                                 cer_improved=improved, cer_degraded=degraded,
                                 short_finish_cer_errors=old_s["cer_errors"],
                                 combination_cer_errors=new_s["cer_errors"])

    # Library hits absent from the reference are only automatic audit candidates.
    added_candidates = []
    library_cache = {}
    for sid in sorted(all_ids):
        inp = inputs[sid]
        lib = Path(str(inp.get("library_path", "")))
        if not lib.is_file():
            lib = base / "libraries" / lib.name
        names = []
        if lib.is_file():
            resolved = str(lib.resolve())
            if resolved not in library_cache:
                with sqlite3.connect(f"file:{resolved}?mode=ro", uri=True) as con:
                    library_cache[resolved] = [str(x[0]) for x in con.execute(
                        "SELECT DISTINCT canonical_text FROM protected_entities WHERE enabled=1")]
            names = library_cache[resolved]
        refn = norm(str(refs[sid].get("reference", "")))
        oldn, newn = norm(texts["qwen_raw"][sid]), norm(texts["combination"][sid])
        for name in names:
            needle = norm(name)
            if needle and needle not in refn and newn.count(needle) > oldn.count(needle):
                added_candidates.append(dict(id=sid, split=inp['split'], canonical=name,
                    qwen_mentions=oldn.count(needle), combination_mentions=newn.count(needle)))

    ctc_totals = Counter()
    review_checks = 0
    for sid in all_ids:
        repair = combo[sid].get("entity_audio_repair") or {}
        inc = repair.get("incremental") or {}
        ctc = inc.get("ctc") or {}
        review_checks += int(repair.get("checks", 0) or 0)
        for key in ("proposals", "committed", "errors", "finish_timed_out"):
            ctc_totals[key] += int(ctc.get(key, 0) or 0)
        ctc_totals["streaming_review_skips"] += int(ctc.get("streaming_review_skips", 0) or 0)
        ctc_totals["short_finish_checks"] += int(inc.get("short_finish_checks", 0) or 0)
        ctc_totals["short_finish_errors"] += int(inc.get("short_finish_errors", 0) or 0)
        ctc_totals["short_finish_timed_out"] += int(inc.get("short_finish_timed_out", 0) or 0)

    comparison = dict(cohort=dict(short_test=len(test_ids), control=len(control),
                                  ids=sorted(all_ids), pcm_sha256=frozen_shas),
        test={arm: test_metrics[arm] for arm in texts},
        control={arm: control_metrics[arm] for arm in ("qwen_raw", "combination")},
        combination_vs_short_finish=dict(entity_recovered=recovered, entity_lost=lost,
            cer_improved_samples=cer_improved, cer_degraded_samples=cer_degraded,
            paired_samples=len(test_ids), per_id=per_id_delta),
        runtime=dict(audio_review_checks=review_checks, ctc=dict(ctc_totals)),
        added_library_canonical_audit_candidates=added_candidates,
        added_library_canonical_audit_counts=dict(
            short_test=sum(x['split']=='test' for x in added_candidates),
            control=sum(x['split']=='control' for x in added_candidates)),
        audit_note="词库 canonical 新增项是自动词法审计候选，未经人工标注；不得解释为 precision。",
        scoring=dict(implementation="experiments.scripts.report_entity_backend_comparison.score/norm",
            normalization="ZhITN then OpenCC t2s, NFKC, casefold, remove Unicode P/Z/C; CER character edit distance; gold mention hits clipped by normalized canonical count."))
    (run / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# CTC 与原音复核组合评测", "",
        f"冻结配对 cohort：short_test {len(test_ids)} 条，control {len(control)} 条。输入 ID、PCM hash、保存 Qwen 主流文本及实际组合输出均已严格核对。", "",
        "## short_test 四臂", "", "| Arm | 样本数 | Entity hits / mentions | Recall | CER |", "|---|---:|---:|---:|---:|"]
    labels = {"qwen_raw": "Qwen raw", "ctc_bypass": "CTC bypass", "short_finish": "当前短收尾", "combination": "实际组合"}
    for arm, metrics in test_metrics.items():
        lines.append(f"| {labels[arm]} | {metrics['samples']} | {metrics['entity_hits']} / {metrics['entity_mentions']} | {metrics['entity_recall']:.2%} | {metrics['cer']:.2%} |")
    lines += ["", "## Control CER", "", "| Arm | 样本数 | CER |", "|---|---:|---:|"]
    for arm in ("qwen_raw", "combination"):
        m = control_metrics[arm]
        lines.append(f"| {labels[arm]} | {m['samples']} | {m['cer']:.2%} |")
    lines += ["", "## 组合相对当前短收尾", "",
        f"配对 {len(test_ids)} 条；gold entity recovered {recovered}，lost {lost}；CER 改善 {cer_improved} 条、退化 {cer_degraded} 条。", "",
        f"原音复核 checks/session 计数合计 {review_checks}；CTC proposals {ctc_totals['proposals']}、committed {ctc_totals['committed']}、errors {ctc_totals['errors']}、finish timeout {ctc_totals['finish_timed_out']}。短收尾复核 checks {ctc_totals['short_finish_checks']}、errors {ctc_totals['short_finish_errors']}、timeouts {ctc_totals['short_finish_timed_out']}。", "",
        f"发现 {len(added_candidates)} 个新增词库 canonical 自动审计候选。该候选表未经过人工复核，不代表 precision。逐 ID paired delta、hash 与审计候选见 comparison.json。", "",
        "所有指标复用 backend comparison 的 score/norm 实现。控制集比较 Qwen 与组合 CER；实体 recall 的分母为 gold mention 数，CER 分母为规范化 reference 字符数。"]
    (run / "COMBINATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
