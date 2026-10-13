"""Validate and export separately recorded text reviews; never create human gold.

Run after collect_gate_pairs.py report to refresh copies of human annotations.
The decision manifest contains individual assistant judgments, not labeling rules.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


BASIS_FIELDS = (
    "case_id", "task_kind", "data_kind", "recording_group", "audio_path",
    "raw_window", "raw_full", "reference_full", "baseline", "always_output",
    "candidate_source", "candidate_target", "candidate_context",
)
ASSISTANT_FIELDS = (
    "assistant_label", "assistant_confidence", "assistant_category",
    "assistant_reason", "assistant_harm_severity", "assistant_needs_audio",
    "assistant_audio_checked", "assistant_status", "assistant_train_eligible",
    "review_basis_sha256", "baseline_sha256", "always_output_sha256",
)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def basis_hash(row: dict) -> str:
    # Human fields are deliberately excluded: confirming a label does not stale
    # the original auxiliary review, whereas changes to text/reference do.
    payload = {key: row[key] for key in BASIS_FIELDS}
    return text_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def unique_index(rows: list[dict], name: str) -> dict:
    result = {row["case_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate case_id in {name}")
    return result


def attach_reviews(rows: list[dict], reviews: list[dict]) -> list[dict]:
    annotations = unique_index(rows, "annotations")
    index = unique_index(reviews, "reviews")
    changed = {key for key, row in annotations.items()
               if row["baseline"] != row["always_output"]}
    if set(index) != changed:
        raise ValueError("Reviews must cover exactly every changed case; "
                         f"missing={sorted(changed - set(index))}, "
                         f"extra={sorted(set(index) - changed)}")
    output = []
    for row in rows:
        # This is a copy; neither human_label nor any other human field is edited.
        merged = {**row, **{key: "" for key in ASSISTANT_FIELDS}}
        review = index.get(row["case_id"])
        if review is not None:
            if (review["review_basis_sha256"] != basis_hash(row)
                    or review["baseline_sha256"] != text_hash(row["baseline"])
                    or review["always_output_sha256"] != text_hash(row["always_output"])):
                raise ValueError(f"Stale review for {row['case_id']}; re-review changed evidence")
            label = review["assistant_label"]
            if label not in {"benefit", "neutral", "harm", "uncertain"}:
                raise ValueError("Invalid assistant label")
            if review["assistant_confidence"] not in {"high", "medium", "low"}:
                raise ValueError("Confidence must be qualitative, not a probability")
            severity = review["assistant_harm_severity"]
            if severity not in {"none", "minor", "major"} or ((label == "harm") != (severity != "none")):
                raise ValueError("Harm severity must agree with the label")
            if not review["assistant_reason"].strip() or not review["assistant_category"].strip():
                raise ValueError("Each review needs an individual reason and category")
            if type(review["assistant_needs_audio"]) is not bool:
                raise ValueError("needs_audio must be a boolean")
            if label == "uncertain" and not review["assistant_needs_audio"]:
                raise ValueError("Uncertain text review needs source-audio adjudication")
            if review["assistant_audio_checked"] is not False:
                raise ValueError("This text-only manifest cannot claim an audio check")
            if review["assistant_status"] != "auxiliary_unconfirmed" or review["assistant_train_eligible"] is not False:
                raise ValueError("Auxiliary reviews cannot be promoted to training gold")
            merged.update({key: review[key] for key in ASSISTANT_FIELDS})
        output.append(merged)
    return output


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_annotations(out: Path) -> list[dict]:
    with (out / "annotations.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = unique_index(rows, "annotations")
    pairs = unique_index(read_jsonl(out / "pairs.jsonl"), "pairs")
    cases = unique_index(read_jsonl(out / "cases.jsonl"), "cases")
    if set(index) != set(pairs) or not set(index) <= set(cases):
        raise ValueError("Annotation/pair coverage differs; rerun collector report")
    for key, row in index.items():
        pair, case = pairs[key], cases[key]
        arms = pair["arms"]
        if any(arm.get("session_stats", {}).get("failed_call_count", 0)
               or any(call.get("error_type") for call in arm.get("logical_calls", []) + arm.get("plausibility_calls", []))
               for arm in arms.values()):
            raise ValueError(f"Failed operation cannot be reviewed: {key}")
        expected = dict(baseline=arms["rules"]["comparison_text"],
                        always_output=arms["always"]["comparison_text"],
                        current_output=arms["current"]["comparison_text"],
                        raw_window=case.get("raw_text", ""), raw_full=case["raw_text_full"],
                        reference_full=case["reference_full"])
        expected.update({field: case[field] for field in
                         ("task_kind", "data_kind", "recording_group", "audio_path")})
        expected.update({f"candidate_{field}": case.get("candidate", {}).get(field, "")
                         for field in ("source", "target", "context")})
        if any(row[field] != value for field, value in expected.items()):
            raise ValueError(f"Stale annotation evidence for {key}; rerun collector report")
        identical = row["baseline"] == row["always_output"]
        if row["automatic_pair_label"] != ("neutral" if identical else ""):
            raise ValueError(f"Invalid automatic neutral label: {key}")
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def priority(row: dict) -> int:
    if row["assistant_label"] == "uncertain" or row["assistant_harm_severity"] == "major":
        return 1
    if row["assistant_label"] == "harm" or row["assistant_needs_audio"]:
        return 2
    return 3 if row["assistant_label"] == "benefit" else 4


def export(out: Path) -> dict:
    rows = load_annotations(out)
    manifest = json.loads((out / "assistant_review_decisions.json").read_text(encoding="utf-8"))
    if manifest["review_method"] != "individual_text_comparison_no_audio":
        raise ValueError("Unexpected review method")
    merged = attach_reviews(rows, manifest["reviews"])
    changed = [row for row in merged if row["assistant_label"]]
    fields = list(rows[0]) + list(ASSISTANT_FIELDS)
    summary = dict(
        total=len(rows), changed_reviewed=len(changed),
        identical_output_neutral=sum(row["baseline"] == row["always_output"] for row in rows),
        auxiliary_label_counts=dict(Counter(row["assistant_label"] for row in changed)),
        auxiliary_harm_severity=dict(Counter(row["assistant_harm_severity"] for row in changed if row["assistant_label"] == "harm")),
        needs_audio_to_judge=sum(row["assistant_needs_audio"] for row in changed),
        audio_checked=0, auxiliary_train_eligible=0,
        human_labels_present=sum(bool(row["human_label"].strip()) for row in rows),
        human_labels_on_changed=sum(bool(row["human_label"].strip()) for row in changed),
        unique_recordings_changed=len({row["recording_group"] for row in changed}),
        by_task={kind: dict(Counter(row["assistant_label"] for row in changed if row["task_kind"] == kind))
                 for kind in sorted({row["task_kind"] for row in changed})},
        by_data_kind={kind: dict(Counter(row["assistant_label"] for row in changed if row["data_kind"] == kind))
                      for kind in sorted({row["data_kind"] for row in changed})},
        manifest_sha256=text_hash((out / "assistant_review_decisions.json").read_text(encoding="utf-8")),
        review_method=manifest["review_method"],
    )
    # All validation precedes writes. Existing human-source files remain intact.
    write_csv(out / "assistant_annotations.csv", merged, fields)
    write_csv(out / "assistant_changed_for_review.csv", changed, fields)
    records = [{"case_id": row["case_id"], "task_kind": row["task_kind"],
                "recording_group": row["recording_group"],
                **{key: row[key] for key in ASSISTANT_FIELDS}} for row in changed]
    (out / "assistant_reviews.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    queue = [{"review_priority": priority(row), "audio_scope": "full_recording",
              "human_confirmation_required": True, **row} for row in changed]
    queue.sort(key=lambda row: (row["review_priority"], row["recording_group"], row["case_id"]))
    write_csv(out / "audio_review_queue.csv", queue,
              ["review_priority", "audio_scope", "human_confirmation_required"] + fields)
    (out / "assistant_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = summary["auxiliary_label_counts"]
    lines = ["# 门控配对样本：文本辅助审核", "",
             f"共 {summary['total']} 条配对结果；{summary['identical_output_neutral']} 条最终输出逐字相同，"
             f"其余 {len(changed)} 条已逐例完成文本辅助审核，来自 {summary['unique_recordings_changed']} 个原始录音来源。", "",
             "标签比较的是强制执行该神经操作的 always 分支与只运行规则的 baseline 分支。"
             "current_output 供核查现有门控是否执行该操作；这里的分类数量不是当前线上错误率。", "",
             "| 辅助标签 | 数量 | 含义 |", "| --- | ---: | --- |"]
    meanings = {"benefit": "候选收益：清理明显重复、赘词或修复断开的语法结构",
                "neutral": "主要是等价标点、数字或字形变化，未见明确净收益",
                "harm": "候选退化：切断句法结构或删除有意义的表达",
                "uncertain": "强调、说话人边界、事实改写或参考失配，需核对原音"}
    lines += [f"| {label} | {counts.get(label, 0)} | {meanings[label]} |" for label in meanings]
    lines += ["", f"候选 harm 程度：{json.dumps(summary['auxiliary_harm_severity'], ensure_ascii=False)}。"
              "minor 指标点／可读性退化；major 指可能改变信息的修改，两者不可混报为事实错误率。", "",
              f"{summary['needs_audio_to_judge']} 条存在局部或整句音频疑点，需听原音进一步判定；所有辅助审核的 audio_checked=false。"
              f"当前人工标签 {summary['human_labels_present']} 条。辅助标签可作为审核建议，不能直接作为训练真值或论文实验结论。", "",
              "收益判断采用忠实保留信息、允许清理口吃与非必要语气词的精修目标。逐字转录目标下，"
              "部分清理会增加 CER；已在相应理由中说明。不审核医学观点正确性，不把参考距离下降或模型接受当作收益证明。", "",
              "high / medium / low 是辅助审核的定性把握，不是校准概率，也不是 ASR 置信度。"
              "assistant_needs_audio=false 仅表示相对文本变化可以初判；转为人工真值仍须按原标注协议确认。", "",
              "## 如何复核", "",
              "1. 按 audio_review_queue.csv 的 review_priority 审核：1 为存疑或内容风险，2 为格式退化或需听音，3 为收益候选，4 为等价改动。",
              "2. 队列覆盖全部发生变化的样本，audio_path 指向完整录音，没有伪造局部音频时间戳。同一录音、相邻窗口及噪声版本应结合查看。",
              "3. 仅在 annotations.csv 填写 human_label、rationale、reviewer、reference_status，存疑可填 uncertain；辅助合并表只是可刷新副本。",
              "4. 重跑 collector 的 report 后，再运行下方导出命令；原人工字段会原样复制，配对输出、参考或审核上下文改变会拒绝复用旧判断。", "",
              "```bash", "python experiments/scripts/export_gate_reviews.py --output results/gate_pilot_20261009", "```", "",
              "## 文件", "",
              "- assistant_review_decisions.json：157 条逐例判断原件；含证据哈希，不按正则自动赋标签。",
              "- assistant_reviews.jsonl：校验后的辅助标签；全部 auxiliary_unconfirmed、assistant_train_eligible=false。",
              "- assistant_annotations.csv：500 条人工表与辅助字段的合并副本；人工原表不会被改写。",
              "- assistant_changed_for_review.csv：157 条变化样本的合并审核视图。",
              "- audio_review_queue.csv：全部 157 条变化样本的原音复核队列。",
              "- assistant_review_summary.json：辅助审核统计，按任务和来源分别列出。", "",
              "## 对门控设计的影响", "",
              "大量中性改动提示门控需要预测净收益，而不是检测句子有无可改之处。条件句、主谓关系、"
              "并列结构及有意义的重叠词应进入内容保护和 harm 学习目标。纯格式 harm 与实体、数字、否定或事实 harm 需要分层报告。", "",
              "典型样本（均为尚未确认的文本辅助结论）：", "",
              "| 原文本 → 操作后 | 辅助判断 |", "| --- | --- |",
              "| `均面临。不均衡的复苏前景` → `均面临不均衡的复苏前景` | benefit：修复动宾连接 |",
              "| `如果脚气一直不好，该怎么解决` → `如果脚气一直不好。该怎么解决` | minor harm：条件句误断 |",
              "| `一级一级地向高处提水` → `一级地向高处提水` | major harm 候选：可能删除逐级含义，需听音 |",
              "| `学了很多很多` → `学了很多` | uncertain：可能是有意强调 |",
              "| `五连冠` → `5连冠` | neutral：数字写法变化 |", "",
              "这批资料属于已用于开发的探索性样本。后续先确认收益／风险标签与精修目标，"
              "再按原始录音来源分组划分；独立测试录音需另采集。尚未训练或启用学习门控，不能据此声称已验证泛化、校准或端到端加速。", ""]
    (out / "ASSISTANT_REVIEW_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
