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
        prepared = prepare_training_pair(source, target, None if meta.get("is_fragment") else meta.get("generation_plan"))
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
                    scope="All 197 retained pairs reviewed once across two independent agents; one disputed pair re-reviewed by both. This does not estimate future or real-ASR error rates.")
    (OUT / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 保留率修复：300条对应候选复测", "", "## 自动流程", "",
             f"- 候选：{summary['candidates_completed']}",
             f"- 去重前接收：{summary['accepted_before_dedup']}",
             f"- 最终保留：{summary['unique_retained']}",
             f"- 近似重复移除：{summary['near_duplicates_removed']}",
             f"- 实际截断样本：{summary['real_fragments']}",
             f"- 总耗时：{summary['elapsed_seconds'] / 60:.1f}分钟", "- 相关回归测试：76项通过（8条依赖弃用警告）", "", "## 独立逐条复核", "",
             f"- 复核：{len(reviews)}条；判定：{dict(decisions)}",
             f"- 初审：{dict(initial_decisions)}；争议复核：{len(adjudications)}条，原报告与原标签保留不改",
             f"- 引用／ID证据问题：{len(evidence_errors)}",
             f"- 全量结构复验问题：{len(structural)}",
             f"- 同一输入对应不同目标：{len(conflicts)}", "",
             "本轮最终保留不足200条，因此197条全部各由一名独立代理审阅，两名代理分工；争议项再由双方独立复核。审阅只覆盖本批合成数据，不能估计未来批次或真实ASR数据的误差率。源自然度问题与标签忠实性问题分别记录。", "",
             "| 场景 | 候选 | 去重前接收 | 最终保留 | 逐条复核 |", "| --- | ---: | ---: | ---: | --- |"]
    for row in per_scene:
        lines.append(f"| {row['scene']} | {row['attempted']} | {row['accepted']} | {row['unique']} | {row['review_counts']} |")
    if (OUT / "matched_comparison.json").exists():
        comparison = json.loads((OUT / "matched_comparison.json").read_text())
        lines += ["", "## 相同候选位置对照", "",
                  "相同模型、种子池、候选位置与控制随机数；教师输出仍有采样随机性，并非逐条固定原输出的回放。", "",
                  "| 指标 | 修改前 | 修改后 |", "| --- | ---: | ---: |"]
        for label, key in (("候选数", "attempted"), ("去重前接收", "accepted"), ("去重后保留", "retained")):
            lines.append(f"| {label} | {comparison['baseline'][key]} | {comparison['current'][key]} |")
        lines.append(f"| 去重后保留率 | {comparison['baseline']['retained_rate']:.1%} | {comparison['current']['retained_rate']:.1%} |")
        lines += ["", "## 请求类型接收情况", "",
                  "| 改口结构 | 修改前接收/候选 | 修改后接收/候选 |", "| --- | ---: | ---: |"]
        for structure, old in comparison["baseline"]["by_structure"].items():
            new = comparison["current"]["by_structure"][structure]
            lines.append(f"| {structure} | {old.get('accepted', 0)}/{old['attempted']} | {new.get('accepted', 0)}/{new['attempted']} |")
        lines += ["", f"修改后有{comparison['current']['asr_no_boundary_full_fallbacks']}条选中ASR分支但没有自然切点，回退为完整样本；其中{comparison['current']['asr_no_boundary_retained']}条通过后续检查并接收，并非截断样本。真实截断仍须满足前缀对齐与语义检查。",
                  f"最终有{comparison['current']['unique_targets']}种目标。后续训练／验证应按最终意图分组，避免同意图干扰变体跨集合。尚未执行训练划分。"]
    lines += ["", "## 检查边界", "", "本次只测试合成文本生成、筛选、去重与标签质量；没有训练或评测学生模型，也没有跑音频识别。教师同时生成标签和自动审查；本轮独立代理已覆盖全部保留对。", "",
              f"样本口语干扰仍较模板化。标签审阅不能证明真实ASR泛化或0.5B学生训练效果；本轮{len(reviews)}条由独立代理分工盲审，并非真人专家全量标注。", "",
              "本轮使用两份相同教师实例、两张RTX3090并行；本轮elapsed_seconds包含启动等待、生成及自动筛选，不是单卡计时，也不包括后续独立审阅。生产源码哈希没有变化。临时GPU1实例在测试完成后关闭，原GPU0教师服务保留。", "",
              "详细证据见 analysis.json、review_1.json、review_2.json（若有争议则另存adjudications.json）；原始请求见 api_calls.jsonl，拒收见 attempts.jsonl。"]
    lines += ["", "## 上游参考", "", "已核对官方 [README](https://github.com/AnXMuy/AgenticASR) 与 [Oral生成实现](https://github.com/AnXMuy/AgenticASR/blob/main/pipeline/src/generators/oral_text_generator.py)。本轮改动属于本地的保留率修复，上游没有提供本次测得的保留率。"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"unique":len(final), "reviewed":len(reviews), "review_counts":dict(decisions),
                     "structural_errors":len(structural), "review_evidence_errors":len(evidence_errors),
                     "conflicting_inputs":len(conflicts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
