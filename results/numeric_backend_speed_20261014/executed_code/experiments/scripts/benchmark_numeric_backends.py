"""Compare warmed numeric backends on matched text windows, without ASR."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def cases():
    groups = {
        "numeric_only": [
            "会议定于二零二六年十月十四日上午九点三十分，参加人数是三十人。",
            "本次采购预算是一百二十三万七千八百五十一元，首付款为三十万元。",
            "产品合格率达到百分之九十八点五，本季度累计交付九万五千一百四十二辆。",
            "测量距离为一到两米，建筑限高三十六米，面积约二百一十六点零三平方米。",
            "测试持续十到十五分钟，每隔二十分钟检查一次，每二十四小时更新记录。",
            "联系电话是一三八零零一三八零零零，备用电话是幺三九九幺二六三七九七。",
            "第五十届运动会已有六十四个代表团报名，预计接待观众超过三十五万人次。",
            "测试版本是四点零点八点二三点一，缓存目录还有五百一十二兆字节。",
            "计算结果是负三点二，传感器允许误差为正负零点零三。",
            "今天气温大概有三十七度，室温为二十五度，水温为三十点五度。",
            "仓库有十五瓶水、十五套设备和二百零五件零件，需要三到四个人搬运。",
            "大概有三十七度。",
        ],
        "numeric_cleanup": [
            "嗯，会议会议定于二零二六年十月十四日上午九点三十分，参加人数是三十人。",
            "本次采购预算预算是一百二十三万七千八百五十一元，呃，首付款为三十万元。",
            "产品合格率达到百分之九十八点五，嗯，本季度本季度交付九万五千一百四十二辆。",
            "测量距离为一到两米，我我觉得建筑限高三十六米，面积约二百一十六点零三平方米。",
            "测试持续十到十五分钟，呃，每隔二十分钟检查检查一次。",
            "联系电话是一三八零零一三八零零零，嗯，我再说一次，联系电话是一三八零零一三八零零零。",
            "第五十届运动会，嗯，已有六十四个代表团报名报名。",
            "测试版本版本是四点零点八点二三点一，缓存目录还有五百一十二兆字节。",
            "计算结果是负三点二，呃，传感器允许误差为正负零点零三。",
            "大概有三十七度。然后哈哈哈，我一无一实的告诉你，",
            "需要十二件，不对，十三件，嗯，请按最后确认的数量安排配送。",
            "会议在一点，哦不，一点十分开始，嗯，请提前到会议室。",
        ],
        "nonnumeric_control": [
            "会议定在明天上午，参加人员包括研发部门和市场部门，请提前安排好会议室。",
            "本次采购已经完成审核，首付款将在合同签署以后按照约定支付。",
            "产品质量符合要求，本季度的交付任务已经完成，请及时整理相关材料。",
            "测量结果已经保存，建筑资料和场地资料也已经交给负责人员。",
            "测试期间请持续观察运行情况，发现异常以后及时联系维护人员。",
            "联系电话已更新到通讯录，备用联系人也已经收到通知。",
            "嗯，会议会议定在明天上午，请研发部门和市场部门提前做好准备。",
            "本次采购预算预算已经审核，呃，请财务部门按约定支付首付款。",
            "我我觉得测量结果应该保存，相关资料也应该交给负责人员。",
            "测试期间请检查检查设备，嗯，发现异常以后及时联系维护人员。",
            "然后哈哈哈，我想告诉你，你觉得这个安排怎么样？",
            "我想去会议室，哦不，应该先去办公室，嗯，请帮我确认地点。",
        ],
    }
    return [dict(id=f"{group}_{i:02d}", group=group, source=source)
            for group, texts in groups.items() for i, source in enumerate(texts)]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["policy"])].append(row)
        grouped[("all", row["policy"])].append(row)
    summary = {}
    for (group, policy), selected in grouped.items():
        elapsed = sorted(row["wall_ms"] for row in selected)
        summary.setdefault(group, {})[policy] = dict(
            observations=len(selected), total_ms=sum(elapsed),
            mean_ms=statistics.mean(elapsed), median_ms=statistics.median(elapsed),
            p95_ms=elapsed[min(len(elapsed)-1, int(.95*len(elapsed)))],
            model_calls=sum(row["model_calls"] for row in selected),
            retries=sum(row.get("retries", 0) for row in selected),
            rejected=sum(row.get("rejected", False) for row in selected),
            rule_ms=sum(row.get("rule_ms", 0) for row in selected),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/home/aim0/data/models/ASR/AgenticASR-Refiner"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from threadpoolctl import threadpool_limits
    from system import web_app
    from system.live_qwen_refiner import TransformersRefiner
    from system.protection import EntityProtector, strip_refiner_key_suffix
    from system.repetition_plausibility import RepetitionPlausibility
    from system.zh_itn_bridge import DEFAULT_LIBRARY, ZhITNNormalizer

    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    samples = cases()
    write(output / "cases.json", samples)
    library = DEFAULT_LIBRARY.resolve()
    normalizer = ZhITNNormalizer(library)
    metadata = dict(
        scope="warmed_sequential_matched_text_windows_no_audio_no_ASR",
        device=args.device, repetitions=args.repetitions, cases=len(samples),
        model=str(args.model), max_new_tokens=256, do_sample=False,
        rule_pack=normalizer.rule_pack, library=str(library), library_sha256=sha(library),
        cpu_threads=2, native_gate="tri_state", learned_gate="off", entity_db=None,
        window="each sample is one completed editable window; identical single_window=True",
        shared_model=True, caches="no output reuse across observations",
        baseline_definitions={
            "refiner_only": "direct refine on raw text, no application guards or numeric fallback",
            "model_web_chain": "Web core with zh-itn disabled; includes existing Python numeric fallback",
            "zh_itn_web_chain": "same Web core with zh-itn enabled, numeric routing and surface guard",
            "zh_itn_only": "CPU rule normalization only; no language cleanup",
        },
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/"system/web_app.py",
            ROOT/"system/refinement_gate.py", ROOT/"system/refinement_guard.py", ROOT/"system/live_qwen_refiner.py"]},
        gpu_before=subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
    )
    write(output / "configuration.json", metadata)

    refiner = TransformersRefiner(args.model, args.device, 256)
    plausibility = RepetitionPlausibility(args.device)
    with patch.object(web_app, "TransformersRefiner", return_value=refiner), \
         patch.object(web_app, "RepetitionPlausibility", return_value=plausibility):
        pipelines = {}
        for policy, lib in [("model_web_chain", None), ("zh_itn_web_chain", library)]:
            app = web_app.create_app(args.model, args.device, "http://unused-asr", "Chinese", 256, None,
                refinement_gate_mode="tri_state", final_refinement_mode="off", zh_itn_library=lib)
            route = next(route for route in app.routes if getattr(route, "path", None) == "/ws/stream")
            closure = dict(zip(route.endpoint.__code__.co_freevars, route.endpoint.__closure__))
            pipelines[policy] = closure["refine_update"].cell_contents

    def run(case, policy):
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        details = {}
        if policy == "refiner_only":
            candidate, model_ms = refiner.refine(case["source"])
            text = strip_refiner_key_suffix(candidate).strip()
            details.update(model_calls=1, model_ms=model_ms, raw_model_output=candidate)
        elif policy == "zh_itn_only":
            result = normalizer.normalize(case["source"])
            text = result.text
            details.update(model_calls=0, changes=[c.public_dict() for c in result.changes],
                           decision_reason=result.decision_reason)
        else:
            stats = web_app._RefinerSessionStats()
            trace = []
            result = pipelines[policy](case["source"], "Chinese", True,
                EntityProtector((), enable_rule_protection=True, protect_numeric_spans=False),
                None, None, single_window=True, confidence_metadata={}, call_stats=stats,
                trace_segments=trace)
            snapshot = stats.snapshot()
            if snapshot["failed_call_count"] or snapshot["inflight_call_count"]:
                raise RuntimeError(str(snapshot))
            text = result["clean_text"]
            details.update(model_calls=snapshot["call_count"], retries=snapshot["retry_call_count"],
                model_ms=snapshot["completed_wall_latency_ms"],
                rule_ms=result.get("numeric_rule_latency_ms", 0),
                rejected=not result["refiner_accepted"],
                reject_reasons=result["refiner_reject_reasons"],
                gate=result["refinement_gate_decisions"], trace=trace,
                numeric_repairs=result.get("numeric_rule_repairs", []))
        torch.cuda.synchronize(args.device)
        return dict(case, policy=policy, output=text, wall_ms=(time.perf_counter()-started)*1000,
                    chars=len(case["source"]), **details)

    policies = ["refiner_only", "model_web_chain", "zh_itn_web_chain", "zh_itn_only"]
    print("Warming shared model and both Web paths", flush=True)
    for policy in policies:
        for case in [samples[0], samples[12], samples[24]]:
            run(case, policy)
    # The local repetition classifier loads lazily; warm it before timing.
    if plausibility._ensure_model():
        plausibility.decide("请确认确认会议地点。", 1, 5, "确认")
    rows = []
    for repetition in range(args.repetitions):
        indices = list(range(len(samples)))
        random.Random(3100 + repetition).shuffle(indices)
        for position, index in enumerate(indices):
            order = policies[index % len(policies):] + policies[:index % len(policies)]
            if repetition % 2:
                order.reverse()
            for policy in order:
                row = run(samples[index], policy)
                row["repetition"] = repetition
                rows.append(row)
                with (output / "observations.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"round {repetition+1}/{args.repetitions}: {position+1}/{len(samples)} {samples[index]['group']}", flush=True)
        write(output / "summary.json", summarize(rows))
    summary = summarize(rows)
    write(output / "summary.json", summary)
    metadata["gpu_after"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    write(output / "configuration.json", metadata)
    lines = ["# 数字后端耗时对比", "", "同一模型预热后交错运行，单窗口文本回放；不包含 ASR、模型加载、音频上传或 WebSocket 调度。", "",
        "每类 12 句，每句 3 轮；所有模型条件复用同一个实例。独立学习门控和实体数据库关闭，隔离数字后端影响。", "",
        "| 类别 | 模式 | 平均 ms | 中位 ms | P95 ms | 模型调用 | 重试 | 拒绝 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group, modes in summary.items():
        for policy, s in modes.items():
            lines.append(f"| {group} | {policy} | {s['mean_ms']:.3f} | {s['median_ms']:.3f} | {s['p95_ms']:.3f} | {s['model_calls']} | {s['retries']} | {s['rejected']} |")
    lines += ["", "纯规则组不承担口语清理，与混合句/非数字精修组不构成同任务质量比较。model_web_chain 含 Python 数字兜底，refiner_only 才是直接纯模型。", "",
        "输出与轨迹见 observations.jsonl；速度结果没有独立人工质量评分。GPU 上还有现有服务，交错多轮可减小顺序影响，不能代表所有线上负载。"]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
