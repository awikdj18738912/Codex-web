"""Replay stored window responses through a strict schema; no model rerun."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.synthetic_gate_data import file_hash, read_rows, write_json
from system.refinement_protocol import apply_structured_patch, STRICT_LOCAL_PROTOCOL_VERSION


def audit(source: Path, output: Path) -> dict:
    pairs = read_rows(source / "pairs.jsonl")
    records = []
    for pair in pairs:
        if pair["task_kind"] != "window_refine":
            continue
        for name, arm in pair["arms"].items():
            if arm.get("physical_execution") != "executed":
                continue
            for index, call in enumerate(arm["logical_calls"]):
                if not call["method"].startswith("refine") or call.get("error_type"):
                    continue
                raw_input, raw_output = call["input"][0], call["output"][0]
                _text, payload, issue = apply_structured_patch(raw_input, raw_output, policy="strict_local")
                records.append(dict(case_id=pair["case_id"], arm=name, call_index=index,
                    strict_schema_accepted=issue is None, issue=issue,
                    action=payload.get("action") if isinstance(payload, dict) else None,
                    source_input=raw_input, stored_response=raw_output))
    if not records:
        raise ValueError("No stored real window model responses")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh audit output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "response_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = dict(responses=len(records), accepted=sum(row["strict_schema_accepted"] for row in records),
        rejected=sum(not row["strict_schema_accepted"] for row in records),
        issues=dict(Counter(row["issue"] for row in records if row["issue"])),
        protocol_version=STRICT_LOCAL_PROTOCOL_VERSION, source_pairs_sha256=file_hash(source / "pairs.jsonl"),
        source_protocol_sha256=file_hash(source / "protocol.json"),
        model_rerun=False, semantic_quality_measured=False,
        scope="stored_responses_from_old_prompt_schema_compatibility_only")
    write_json(output / "summary.json", summary)
    lines = ["# 现有 Refiner 输出的严格协议兼容性", "",
        f"读取已有真实窗口模型响应 {summary['responses']} 条，按物理执行去除 current/always 共享响应。",
        f"严格协议接受 {summary['accepted']} 条，拒绝 {summary['rejected']} 条。", "",
        "| 拒绝原因 | 数量 |", "| --- | ---: |"]
    lines += [f"| {key} | {value} |" for key, value in summary["issues"].items()]
    lines += ["", "这些响应来自旧提示词，未用严格提示词重新推理；结果只反映旧响应的格式兼容性，"
        "不能据此估算严格模式的纠错率、误改率或速度。", "",
        "若大量响应是整段文本，直接启用严格模式会拒绝既有纠错。应先进行保留／局部补丁格式适配，"
        "然后在冻结 ASR 和当前验证链上重新运行成对实验，分别报告拦截风险与丢失的有效修改。", "",
        "严格协议只约束响应结构、唯一源位置、长度、修改区域和窗口末尾边界。"
        "一个格式合格的补丁仍可能删掉强调或改错词义；现有实体、数字和语义校验继续执行，真人质量仍需验证。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/gate_pilot_20261009")
    parser.add_argument("--output", type=Path, default=ROOT / "results/refiner_local_edit_20261009/protocol_audit")
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
