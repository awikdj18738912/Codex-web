"""Summarize the frozen scale run and validate independent review evidence."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from src.generators.asr_simulator import valid_partial_pair
from src.utils.numeric_policy import prepare_training_pair


def read_jsonl(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    final = read_jsonl("final/train.jsonl")
    attempts = read_jsonl("attempts.jsonl")
    accepted = read_jsonl("accepted_candidates.jsonl")
    structural = []
    for row in final:
        source, target, meta = row["input"], row["output"]["refined_text"], row["meta"]
        prepared = prepare_training_pair(source, target)
        if prepared is None or prepared[:2] != (source, target):
            structural.append({"id": row["id"], "problem": "numeric/entity validation not stable"})
        if meta.get("is_fragment"):
            if not valid_partial_pair(row["oral_text"], meta["fragment_reference_clean"], source, target):
                structural.append({"id": row["id"], "problem": "invalid fragment"})
        elif target != meta["generation_plan"]["expected_clean"]:
            structural.append({"id": row["id"], "problem": "target differs from plan"})
    by_input = defaultdict(set)
    for row in accepted:
        by_input[row["input"]].add(row["output"]["refined_text"])
    conflicts = [{"input": source, "targets": sorted(targets)} for source, targets in by_input.items() if len(targets) > 1]
    reviews = []
    evidence_errors = []
    for reviewer in (1, 2):
        blind = {row["id"]: row for row in json.loads((OUT / f"blind_review_{reviewer}.json").read_text())}
        report = json.loads((OUT / f"review_{reviewer}.json").read_text())
        rows = report["reviews"]
        ids = [row["id"] for row in rows]
        if set(ids) != set(blind) or len(ids) != len(set(ids)):
            evidence_errors.append({"reviewer": reviewer, "problem": "review ID coverage mismatch"})
        for row in rows:
            item = blind.get(row["id"])
            if item is None:
                continue
            if row.get("decision") not in {"pass", "fail", "uncertain"}:
                evidence_errors.append({"id": row["id"], "problem": "invalid decision"})
            for key, actual in (("source_quote", item["input"]), ("target_quote", item["target"])):
                quote = row.get(key, "")
                if not isinstance(quote, str) or not quote or quote not in actual:
                    evidence_errors.append({"id": row["id"], "problem": "quote is not an actual substring", "field": key})
            reviews.append({**row, "reviewer": reviewer})
    initial_decisions = Counter(row["decision"] for row in reviews)
    adjudications = []
    if (OUT / "adjudications.json").exists():
        adjudications = json.loads((OUT / "adjudications.json").read_text())["adjudications"]
        for adjudication in adjudications:
            matches = [row for row in reviews if row["id"] == adjudication["id"]]
            if len(matches) != 1:
                raise ValueError("Adjudication does not refer to exactly one reviewed case")
            matches[0]["original_decision"] = matches[0]["decision"]
            matches[0]["decision"] = adjudication["decision"]
            matches[0]["adjudication_reason"] = adjudication["reason"]
    source_issues = [row for row in reviews if row.get("source_issue") not in (False, None, "", [], "none", "无")]
    decisions = Counter(row["decision"] for row in reviews)
    flagged = [row for row in reviews if row["decision"] != "pass"]
    audit_reasons = Counter(str(row.get("audit_reason", "")).split(":", 1)[0]
                            for row in attempts if row["status"] == "audit_rejected")
    per_scene = []
    for scene, total in summary["scene_quotas"].items():
        scene_attempts = [row for row in attempts if row["scene"] == scene]
        scene_reviews = [row for row in reviews if any(
            item["id"] == row["id"] and item["meta"]["scene"] == scene for item in final)]
        per_scene.append(dict(scene=scene, attempted=total,
                              status_counts=dict(Counter(row["status"] for row in scene_attempts)),
                              accepted=summary["accepted_by_scene"].get(scene, 0),
                              unique=summary["unique_by_scene"].get(scene, 0),
                              review_counts=dict(Counter(row["decision"] for row in scene_reviews))))
    analysis = dict(summary=summary, structural_errors=structural,
                    exact_input_target_conflicts=conflicts, review_evidence_errors=evidence_errors,
                    initial_review_counts=dict(initial_decisions), adjudications=adjudications,
                    review_counts=dict(decisions), reviewed=len(reviews), flagged_reviews=flagged,
                    source_quality_notes=source_issues, audit_rejection_categories=dict(audit_reasons),
                    per_scene=per_scene,
                    scope="The 200-case review is deliberately scene/branch balanced, not a uniform estimate of population error rate.")
    (OUT / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 1000条候选扩大测试", "", "## 自动流程", "",
             f"- 候选：{summary['candidates_completed']}",
             f"- 去重前接收：{summary['accepted_before_dedup']}",
             f"- 最终保留：{summary['unique_retained']}",
             f"- 近似重复移除：{summary['near_duplicates_removed']}",
             f"- 实际截断样本：{summary['real_fragments']}",
             f"- 总耗时：{summary['elapsed_seconds'] / 60:.1f}分钟", "", "## 独立抽样复核", "",
             f"- 复核：{len(reviews)}条；判定：{dict(decisions)}",
             f"- 初审：{dict(initial_decisions)}；争议复核：{len(adjudications)}条，原报告与原标签保留不改",
             f"- 引用／ID证据问题：{len(evidence_errors)}",
             f"- 全量结构复验问题：{len(structural)}",
             f"- 同一输入对应不同目标：{len(conflicts)}", "",
             "抽样按场景和稀有分支覆盖分层，不能直接将比例当作全部数据的误差率。源自然度问题与标签忠实性问题分别记录。", "",
             "| 场景 | 候选 | 去重前接收 | 最终保留 | 抽样复核 |", "| --- | ---: | ---: | ---: | --- |"]
    for row in per_scene:
        lines.append(f"| {row['scene']} | {row['attempted']} | {row['accepted']} | {row['unique']} | {row['review_counts']} |")
    if (OUT / "distribution_analysis.json").exists():
        distribution = json.loads((OUT / "distribution_analysis.json").read_text())
        lines += ["", "## 主要剩余问题", "",
                  "抽查未发现明确标签错误，但生成效率和保留后的类型分布仍有问题：", "",
                  "| 请求的改口结构 | 候选 | 去重前接收 | 接收率 |",
                  "| --- | ---: | ---: | ---: |"]
        for structure, values in distribution["control_attempt_outcomes"].items():
            lines.append(f"| {structure} | {values['attempted']} | {values.get('accepted',0)} | {100*values.get('accepted',0)/values['attempted']:.1f}% |")
        lines += ["", "回退和多次改口接收率约10%，最终数据因此偏向简单清理和原样保留。全量生成应对困难类型设置保留后配额，并优化其计划生成成功率。", "",
                  f"实际截断样本 {summary['real_fragments']}/{len(final)}（{100*summary['real_fragments']/len(final):.1f}%）。20%是分支抽样概率，不能当作最终配额；本轮{distribution['asr_selected_without_natural_boundary']}条在选中ASR分支时没有合适自然切点而被拒收。", "",
                  f"最终{len(final)}条共有{distribution['unique_targets']}种不同目标，同一目标的不同干扰版本应在后续训练／验证切分时按最终意图分组，避免跨集合近似泄漏。当前尚未执行该训练划分。", "",
                  "一条初审争议涉及是否保留历史信息。原审阅者重审和另一独立审阅者都依据“不对”“最后改成”改判为明确即时改口；原始初审、复核证据和原标签全部保留。两条源自然度备注属于刻意注入的误断句／口吃，目标已正确清理。"]
    lines += ["", "## 检查边界", "", "本次只测试合成文本生成、筛选、去重与标签质量；没有训练或评测学生模型，也没有跑音频识别。教师同时生成标签和自动审查，独立代理仅审阅预设抽样。", "",
              "样本口语干扰仍较模板化。样本标签抽查通过不能证明真实ASR泛化或0.5B学生训练效果；200条为代理独立分层盲审，并非真人专家全量标注。", "",
              "本轮使用两份相同教师实例、两张RTX3090并行；18.2分钟包含生成及自动筛选，不是单卡计时，也不包括后续独立审阅。生产源码哈希没有变化。临时GPU1实例在测试完成后关闭，原GPU0教师服务保留。", "",
              "详细证据见 analysis.json、review_1.json、review_2.json、adjudications.json；原始请求见 api_calls.jsonl，拒收见 attempts.jsonl。"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"unique":len(final), "reviewed":len(reviews), "review_counts":dict(decisions),
                     "structural_errors":len(structural), "review_evidence_errors":len(evidence_errors),
                     "conflicting_inputs":len(conflicts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
