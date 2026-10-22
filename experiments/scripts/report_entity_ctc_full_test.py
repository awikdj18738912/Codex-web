#!/usr/bin/env python3
"""Strict full-test report for the current entity audio-review and CTC arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.report_entity_backend_comparison import norm, score

BASE = ROOT / "results/entity_backend_comparison_20261015"
ARCHIVE = ROOT / "results/entity_challenge_scene_eval_20261014_v2"
RUN = ROOT / "results/entity_ctc_live_integration_20261016"
GAP, MARGIN = -2., .1
ARMS = ("qwen_raw", "ctc_bypass", "archived_old_primary", "archived_old_entity_audio_raw",
        "current_review_off", "ctc_review_on")
LABELS = {
    "qwen_raw": "Qwen raw",
    "ctc_bypass": "CTC bypass (gap=-2, margin=0.1)",
    "archived_old_primary": "Archived old paired primary",
    "archived_old_entity_audio_raw": "Archived old entity audio review",
    "current_review_off": "Current short-finish review (CTC off)",
    "ctc_review_on": "Current CTC + audio review (CTC on)",
}


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
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
    result = {}
    for row in rows:
        sid = str(row.get("id", ""))
        if not sid:
            raise ValueError(f"Missing ID in {label}")
        if sid in result:
            raise ValueError(f"Duplicate {label} ID: {sid}")
        result[sid] = row
    return result


def require_exact(rows, expected, label):
    actual = set(rows)
    if actual != expected:
        raise ValueError(f"{label} IDs differ: missing={sorted(expected-actual)[:10]} "
                         f"extra={sorted(actual-expected)[:10]} expected={len(expected)} actual={len(actual)}")


def aggregate(values):
    hits = sum(v["entity_hits"] for v in values)
    mentions = sum(v["entity_mentions"] for v in values)
    errors = sum(v["cer_errors"] for v in values)
    chars = sum(v["reference_chars"] for v in values)
    return dict(samples=len(values), entity_hits=hits, entity_mentions=mentions,
                entity_recall=hits / mentions if mentions else 0., cer_errors=errors,
                reference_chars=chars, cer=errors / chars if chars else (0. if not errors else 1.))


def metric_for_texts(ids, texts, refs):
    per_id = {}
    for sid in sorted(ids):
        if sid not in texts:
            raise ValueError(f"Missing hypothesis: {sid}")
        ref = refs[sid]
        per_id[sid] = score(str(ref.get("reference", "")), str(texts[sid]),
                             ref.get("gold_entities", []))
    return aggregate(list(per_id.values())), per_id


def review_counts(rows, *, live=False):
    out = Counter()
    available = {"short_finish_checks": False, "ctc_committed": False, "ctc_errors": False}
    for row in rows.values():
        repair = row.get("entity_audio_repair") if live else row.get("audio_repair")
        repair = repair if isinstance(repair, dict) else {}
        if live:
            incremental = repair.get("incremental") or {}
            ctc = incremental.get("ctc") or {}
            out["review_checks"] += int(repair.get("checks", 0) or 0)
            available["short_finish_checks"] |= "short_finish_checks" in incremental
            out["short_finish_checks"] += int(incremental.get("short_finish_checks", 0) or 0)
            available["ctc_committed"] |= "committed" in ctc
            available["ctc_errors"] |= "errors" in ctc
            out["ctc_committed"] += int(ctc.get("committed", 0) or 0)
            out["ctc_errors"] += int(ctc.get("errors", 0) or 0)
        else:
            incremental = repair.get("incremental") or {}
            out["review_checks"] += int(repair.get("checks", 0) or 0)
            available["short_finish_checks"] |= "short_finish_checks" in incremental
            out["short_finish_checks"] += int(incremental.get("short_finish_checks", 0) or 0)
    return dict(review_checks=out["review_checks"],
                ctc_committed=out["ctc_committed"] if available["ctc_committed"] else None,
                ctc_errors=out["ctc_errors"] if available["ctc_errors"] else None,
                short_finish_checks=out["short_finish_checks"] if available["short_finish_checks"] else None)


def clipped_entity_delta(old_text, new_text, gold):
    old, new = norm(old_text), norm(new_text)
    recovered = lost = 0
    for canonical, count in gold.items():
        before, after = min(count, old.count(canonical)), min(count, new.count(canonical))
        recovered += max(0, after - before)
        lost += max(0, before - after)
    return recovered, lost


def source_group_bootstrap(ids, inputs, old_scores, new_scores, *, repetitions=1000):
    """Re-sample recordings/speakers, then recompute pooled rate differences."""
    grouped = {}
    for sid in ids:
        group = str(inputs[sid].get("source_group") or sid)
        before, after = old_scores[sid], new_scores[sid]
        totals = grouped.setdefault(group, [0, 0, 0, 0])
        totals[0] += after["entity_hits"]-before["entity_hits"]
        totals[1] += before["entity_mentions"]
        totals[2] += after["cer_errors"]-before["cer_errors"]
        totals[3] += before["reference_chars"]
    groups = sorted(grouped)
    rng = random.Random(20261015)
    entity, cer = [], []
    for _ in range(repetitions):
        totals = [0, 0, 0, 0]
        for _ in groups:
            row = grouped[rng.choice(groups)]
            for i,value in enumerate(row):totals[i] += value
        if totals[1]:entity.append(totals[0]/totals[1])
        if totals[3]:cer.append(totals[2]/totals[3])
    def interval(values):
        values.sort()
        return [values[int((len(values)-1)*.025)],
                values[int((len(values)-1)*.975)]] if values else None
    return dict(source_groups=len(groups), repetitions=repetitions, seed=20261015,
        entity_recall_difference_95ci=interval(entity),
        cer_difference_95ci=interval(cer),
        direction="CTC+review minus current review; negative CER difference is better")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    args = parser.parse_args()
    base, archive, run = args.base_dir.resolve(), args.archive_dir.resolve(), args.run_dir.resolve()

    inputs = index(read_jsonl(base / "inputs.jsonl"), "frozen input")
    refs = read_json(base / "references.json").get("samples", {})
    if set(inputs) != set(refs):
        raise ValueError("Frozen input and reference ID sets differ")
    test_ids = {sid for sid, row in inputs.items() if row.get("split") == "test"}
    controls = {sid for sid, row in inputs.items() if row.get("split") == "control"}
    if len(test_ids) != 1829:
        raise ValueError(f"Frozen test cohort expected 1829, found {len(test_ids)}")
    qwen_all = index(read_jsonl(base / "qwen_primary_streams.jsonl"), "saved Qwen primary")
    qwen = {sid: qwen_all[sid] for sid in test_ids if sid in qwen_all}
    require_exact(qwen, test_ids, "saved Qwen test")
    hybrid_rows = read_jsonl(base / "hybrid_results.jsonl")
    bypass = index((r for r in hybrid_rows if float(r.get("minimum_gap", float("nan"))) == GAP
                    and float(r.get("minimum_margin", float("nan"))) == MARGIN
                    and str(r.get("split")) == "test"), "selected CTC bypass")
    archived_all = index(read_jsonl(archive / "comparisons.jsonl"), "archived comparison")
    archived = {sid: archived_all[sid] for sid in test_ids if sid in archived_all}
    archived_result_all = index(read_jsonl(archive / "results.jsonl"), "archived result")
    archived_review = {sid: {"audio_repair":
        (archived_result_all[sid].get("result") or {}).get("entity_audio_repair")}
        for sid in test_ids if sid in archived_result_all}
    off = index(read_jsonl(run / "full_test_current_review.jsonl"), "current review off")
    on = index(read_jsonl(run / "full_test_combination.jsonl"), "CTC plus review")
    for label, rows in (("CTC bypass", bypass), ("archived comparison test intersection", archived),
                        ("archived review statistics", archived_review),
                        ("current review off", off), ("CTC plus review", on)):
        missing = test_ids - set(rows)
        extra = set(rows) - test_ids
        if missing or extra:
            raise ValueError(f"{label} must cover exactly all 1829 test IDs: missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}")

    texts = {arm: {} for arm in ARMS}
    primary_audit = {"off_vs_saved_exact_mismatches": [],
                     "on_vs_saved_exact_mismatches": [],
                     "off_vs_on_normalized_mismatches": [],
                     "off_vs_on_exact_mismatches": []}
    for sid in sorted(test_ids):
        inp = inputs[sid]
        frozen_sha = str(inp.get("pcm_sha256", ""))
        if not frozen_sha:
            raise ValueError(f"Missing frozen PCM SHA: {sid}")
        pcm_path=Path(str(inp.get("pcm_path", "")))
        if not pcm_path.is_file() or hashlib.sha256(pcm_path.read_bytes()).hexdigest()!=frozen_sha:
            raise ValueError(f"Actual PCM differs from frozen SHA: {sid}")
        if off[sid].get("entity_ctc_mode")!="off" or on[sid].get("entity_ctc_mode")!="on":
            raise ValueError(f"CTC mode mismatch in paired live rows: {sid}")
        for label, row in (("CTC bypass", bypass[sid]), ("current review off", off[sid]),
                           ("CTC plus review", on[sid])):
            if row.get("split") != "test":
                raise ValueError(f"{label} split mismatch: {sid}")
            if row.get("pcm_sha256") != frozen_sha:
                raise ValueError(f"{label} PCM SHA mismatch: {sid}")
            primary = row.get("primary_text")
            if primary is None:
                raise ValueError(f"{label} lacks actual primary text: {sid}")
        saved_text = str((qwen[sid].get("final_payload") or {}).get("text", ""))
        if str(bypass[sid].get("primary_text")) != saved_text:
            raise ValueError(f"CTC bypass primary differs from frozen Qwen: {sid}")
        off_primary, on_primary = str(off[sid]["primary_text"]), str(on[sid]["primary_text"])
        if off_primary != saved_text:
            primary_audit["off_vs_saved_exact_mismatches"].append(sid)
        if on_primary != saved_text:
            primary_audit["on_vs_saved_exact_mismatches"].append(sid)
        if off_primary != on_primary:
            primary_audit["off_vs_on_exact_mismatches"].append(sid)
        if norm(off_primary) != norm(on_primary):
            primary_audit["off_vs_on_normalized_mismatches"].append(dict(
                id=sid, off=off_primary, on=on_primary, saved=saved_text))
        texts["qwen_raw"][sid] = str((qwen[sid].get("final_payload") or {}).get("text", ""))
        texts["ctc_bypass"][sid] = str(bypass[sid].get("text", ""))
        old_arms = archived[sid].get("arms") or {}
        for arm_key in ("paired_primary", "entity_audio_raw"):
            if arm_key not in old_arms:
                raise ValueError(f"Archived row {sid} lacks {arm_key}")
        texts["archived_old_primary"][sid] = str(old_arms["paired_primary"].get("text", ""))
        texts["archived_old_entity_audio_raw"][sid] = str(old_arms["entity_audio_raw"].get("text", ""))
        texts["current_review_off"][sid] = str(off[sid].get("text", ""))
        texts["ctc_review_on"][sid] = str(on[sid].get("text", ""))

    summaries, per_id = {}, {}
    for arm in ARMS:
        summaries[arm], per_id[arm] = metric_for_texts(test_ids, texts[arm], refs)

    paired_ids = test_ids - {r["id"] for r in primary_audit["off_vs_on_normalized_mismatches"]}
    recovered = lost = cer_improved = cer_degraded = 0
    per_id_delta = {}
    for sid in sorted(paired_ids):
        gold = Counter(per_id["ctc_review_on"][sid]["gold_counts"])
        rec, los = clipped_entity_delta(texts["current_review_off"][sid], texts["ctc_review_on"][sid], gold)
        off_errors = per_id["current_review_off"][sid]["cer_errors"]
        on_errors = per_id["ctc_review_on"][sid]["cer_errors"]
        improved, degraded = on_errors < off_errors, on_errors > off_errors
        recovered += rec; lost += los; cer_improved += improved; cer_degraded += degraded
        per_id_delta[sid] = dict(entity_recovered=rec, entity_lost=los,
            cer_improved=improved, cer_degraded=degraded,
            off_cer_errors=off_errors, on_cer_errors=on_errors)

    counts = {
        "archived_old_entity_audio_raw": review_counts(archived_review, live=False),
        "current_review_off": review_counts(off, live=True),
        "ctc_review_on": review_counts(on, live=True),
        "qwen_raw": None,
        "ctc_bypass": None,
        "archived_old_primary": None,
    }
    bootstrap=source_group_bootstrap(paired_ids,inputs,per_id["current_review_off"],
                                      per_id["ctc_review_on"])
    paired_metrics = {arm: aggregate([per_id[arm][sid] for sid in sorted(paired_ids)])
                      for arm in ("current_review_off", "ctc_review_on")}
    lost_cases = []
    for sid, delta in per_id_delta.items():
        if not delta["entity_lost"]:
            continue
        ctc = on[sid]["entity_audio_repair"]["incremental"]["ctc"]
        lost_cases.append(dict(id=sid, entity_mentions_lost=delta["entity_lost"],
            off_text=texts["current_review_off"][sid], on_text=texts["ctc_review_on"][sid],
            ctc_proposals=ctc.get("proposals", 0), ctc_committed=ctc.get("committed", 0)))
    # The separate 330-control result set is cited as a prior report, not
    # merged into the 1829-test metrics or claimed as part of these full tests.
    prior_verification_path = run / "implementation_verification.json"
    prior_comparison_path = run / "comparison.json"
    prior = {}
    if prior_verification_path.is_file() and prior_comparison_path.is_file():
        verification = read_json(prior_verification_path)
        prior_comparison = read_json(prior_comparison_path)
        prior_control = (prior_comparison.get("control") or {}).get("combination") or {}
        prior = dict(source=str(prior_comparison_path.relative_to(ROOT)),
            verification_source=str(prior_verification_path.relative_to(ROOT)),
            samples=int(verification.get("combined_quality_samples", 0)),
            controls=int(prior_control.get("samples", 0)),
            control_entity_hits=prior_control.get("entity_hits"),
            control_entity_mentions=prior_control.get("entity_mentions"),
            control_cer=prior_control.get("cer"),
            separate_from_full_test=True,
            note="Earlier combined-quality cohort of 330 = 230 short-test + 100 controls; control metric is Qwen-plus-combination report output and is not a full-test or new-run control comparison.")

    output = dict(test_cohort=dict(samples=len(test_ids), expected_samples=1829,
                                   split="test", test_tuning="No test parameter selection performed."),
        scoring=dict(source="experiments.scripts.report_entity_backend_comparison.score/norm",
            normalization="ZhITN, OpenCC t2s, NFKC/casefold, remove Unicode P/Z/C; CER character Levenshtein; entity hits clipped per canonical to gold mention count."),
        arms={arm: dict(**summaries[arm], review_counts=counts[arm]) for arm in ARMS},
        combination_on_vs_off=dict(entity_recovered=recovered, entity_lost=lost,
            cer_improved_samples=cer_improved, cer_degraded_samples=cer_degraded,
            paired_samples=len(paired_ids), paired_metrics=paired_metrics,
            source_group_bootstrap=bootstrap, entity_loss_cases=lost_cases,
            per_id=per_id_delta),
        primary_audit=dict(**primary_audit,
            exact_off_saved_matches=len(test_ids)-len(primary_audit["off_vs_saved_exact_mismatches"]),
            exact_on_saved_matches=len(test_ids)-len(primary_audit["on_vs_saved_exact_mismatches"]),
            exact_off_on_matches=len(test_ids)-len(primary_audit["off_vs_on_exact_mismatches"]),
            normalized_off_on_matches=len(paired_ids)),
        prior_330_control_results=prior,
        validation=dict(unique_exact_test_ids=True, split_checked=True,
            pcm_sha_checked=True, actual_primary_drift_reported_separately=True,
            ctc_bypass_parameters=dict(minimum_gap=GAP, minimum_margin=MARGIN)))
    (run / "full_test_comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Live CTC + 当前原音复核：冻结 test 全量比较", "",
        "冻结 test 的 1829 个唯一 ID 均已覆盖；逐臂核验 split 与 PCM SHA。实际 Qwen 首轮可能有轻微波动，off/on 相互规范化一致的样本才纳入严格配对收益；波动逐条记于 JSON。CTC bypass 固定使用 gap=-2、margin=0.1；不基于 test 结果调参。", "",
        "## 六臂质量指标", "", "| Arm | Samples | Entity hits / mentions | Recall | CER | Review checks | Short-finish checks | CTC committed | CTC errors |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        m = summaries[arm]; c = counts[arm] or {}
        lines.append(f"| {LABELS[arm]} | {m['samples']} | {m['entity_hits']} / {m['entity_mentions']} | {m['entity_recall']:.2%} | {m['cer']:.2%} | {c.get('review_checks', '—')} | {c.get('short_finish_checks', '—')} | {c.get('ctc_committed', '—')} | {c.get('ctc_errors', '—')} |")
    lines += ["", "## CTC on 相对当前 off", "",
        f"按 canonical 和 gold 提及次数上限裁剪：新增命中 {recovered} 个实体提及、丢失 {lost} 个提及，净增 {recovered-lost}；CER 改善 {cer_improved} 条音频、退化 {cer_degraded} 条。首轮规范化一致的配对样本 {len(paired_ids)} / {len(test_ids)}。提及变化不是修改精确率或误改次数。逐 ID delta 见 `full_test_comparison.json`。", "",
        f"Qwen 首轮与保存基线逐字不一致：off {len(primary_audit['off_vs_saved_exact_mismatches'])} 条、on {len(primary_audit['on_vs_saved_exact_mismatches'])} 条；off/on 首轮规范化后仍不一致 {len(primary_audit['off_vs_on_normalized_mismatches'])} 条，已从严格配对增益中排除。", "",
        f"严格配对分母：{paired_metrics['current_review_off']['entity_mentions']} 个实体提及、{paired_metrics['current_review_off']['reference_chars']} 个参考字符；off/on 实体命中分别为 {paired_metrics['current_review_off']['entity_hits']} / {paired_metrics['ctc_review_on']['entity_hits']}，字符错误分别为 {paired_metrics['current_review_off']['cer_errors']} / {paired_metrics['ctc_review_on']['cer_errors']}。", "",
        f"按 {bootstrap['source_groups']} 个来源组有放回重采样 1000 次，命中率差值的 95% 百分位区间为 +{100*bootstrap['entity_recall_difference_95ci'][0]:.2f} 至 +{100*bootstrap['entity_recall_difference_95ci'][1]:.2f} 个百分点；CER 差值为 {100*bootstrap['cer_difference_95ci'][0]:.3f} 至 {100*bootstrap['cer_difference_95ci'][1]:.3f} 个百分点。该区间不包含单次服务运行的随机性。", "",
        f"丢失的 {lost} 个提及来自 {len(lost_cases)} 条音频；其中 {sum(not c['ctc_proposals'] for c in lost_cases)} 条的 CTC 提案数为零，不能直接归因为错误的 CTC 替换。复核输出与窗口调度的差异仍可能影响结果，需另行固定复核响应或重复运行来隔离。", "",
        "## 另行引用的控制结果", ""]
    if prior:
        lines.append(f"先前 `{prior['source']}` 报告含 {prior['samples']} 条合并质量样本，其中 controls {prior['controls']} 条；此前组合 control CER={prior.get('control_cer')}。这组控制来自先前组合评测，数据、cohort 与口径均单独保留，不并入本次 1829 条 full-test，也不是本次 full-test off/on 的 control 对照。")
    else:
        lines.append("未找到先前 330 样本控制报告文件；此处不报告控制指标。")
    lines += ["", "实体 recall 使用规范化 canonical 子串计数，无实体边界或上下文判断，是词面命中指标；分母是 gold mentions 总数。CER 分母是规范化 reference 字符数。Review checks 是日志中的复核检查计数，单次检查可能涉及多个流式请求；CTC errors=0 仅指错误计数器为零。", "",
        "词库场景划分曾借助参考文本，属于已知场景条件，不能代表未知领域或自动领域路由。两臂各运行一次、加速提交音频，耗时不能视作交互延迟；严格配对仅排除了首轮文本漂移，没有固定后续复核响应与调度。归档旧模块的两个输出臂分别列示。", ""]
    (run / "FULL_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
