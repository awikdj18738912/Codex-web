"""Experiment: retain numeric guards but skip numeric-failure model retries."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.benchmark_numeric_backends import cases, sha, summarize, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from threadpoolctl import threadpool_limits
    from system import web_app
    from system.entity_pipeline import has_placeholder_failure
    from system.live_qwen_refiner import TransformersRefiner
    from system.protection import EntityProtector, strip_refiner_key_suffix
    from system.repetition_plausibility import RepetitionPlausibility
    from system.zh_itn_bridge import DEFAULT_LIBRARY, ZhITNNormalizer

    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    samples = [case for case in cases() if case["group"] == "numeric_cleanup"]
    write(output / "cases.json", samples)
    model = Path("/home/aim0/data/models/ASR/AgenticASR-Refiner")
    refiner = TransformersRefiner(model, args.device, 256)
    plausibility = RepetitionPlausibility(args.device)
    with patch.object(web_app, "TransformersRefiner", return_value=refiner), \
         patch.object(web_app, "RepetitionPlausibility", return_value=plausibility):
        app = web_app.create_app(model, args.device, "http://unused-asr", "Chinese", 256, None,
            refinement_gate_mode="tri_state", final_refinement_mode="off", zh_itn_library=DEFAULT_LIBRARY)
        route = next(route for route in app.routes if getattr(route, "path", None) == "/ws/stream")
        closure = dict(zip(route.endpoint.__code__.co_freevars, route.endpoint.__closure__))
        pipeline = closure["refine_update"].cell_contents
    native_retry = web_app.has_retryable_integrity_failure
    numeric_failures = {"numeric_value_mismatch", "numeric_rule_surface_mismatch",
                        "partial_numeric_range_conversion"}

    def run(case, policy):
        skipped = []

        def fast_retry(reasons, *, numeric_rule_mode=False):
            if not has_placeholder_failure(reasons) and numeric_failures.intersection(reasons):
                skipped.append(list(reasons))
                return False
            return native_retry(reasons, numeric_rule_mode=numeric_rule_mode)

        context = (patch.object(web_app, "has_retryable_integrity_failure", side_effect=fast_retry)
                   if policy == "zh_itn_fast_fallback" else nullcontext())
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        with context:
            if policy == "refiner_only":
                text, model_ms = refiner.refine(case["source"])
                text = strip_refiner_key_suffix(text).strip()
                details = dict(model_calls=1, model_ms=model_ms)
            else:
                stats = web_app._RefinerSessionStats()
                trace = []
                result = pipeline(case["source"], "Chinese", True,
                    EntityProtector((), enable_rule_protection=True, protect_numeric_spans=False),
                    None, None, single_window=True, confidence_metadata={}, call_stats=stats,
                    trace_segments=trace)
                snap = stats.snapshot()
                if snap["failed_call_count"] or snap["inflight_call_count"]:
                    raise RuntimeError(str(snap))
                text = result["clean_text"]
                details = dict(model_calls=snap["call_count"], model_ms=snap["completed_wall_latency_ms"],
                    retries=snap["retry_call_count"], rule_ms=result.get("numeric_rule_latency_ms", 0),
                    rejected=not result["refiner_accepted"], reject_reasons=result["refiner_reject_reasons"],
                    skipped_numeric_retries=skipped, trace=trace)
        torch.cuda.synchronize(args.device)
        return dict(case, policy=policy, output=text, wall_ms=(time.perf_counter()-started)*1000, **details)

    policies = ["refiner_only", "zh_itn_web_chain", "zh_itn_fast_fallback"]
    metadata = dict(scope="experimental_numeric_retry_policy_matched_warm_text_windows",
        production_change=False, numeric_checks="unchanged; only numeric-failure retry admission differs",
        fallback="existing validated source fallback, including existing safe salvage",
        cases=len(samples), repetitions=args.repetitions, device=args.device,
        rule_pack=ZhITNNormalizer().rule_pack, learned_gate="off", entity_db=None,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/"system/web_app.py",
            ROOT/"system/refinement_guard.py", ROOT/"system/entity_pipeline.py"]})
    write(output / "configuration.json", metadata)
    print("Warming both retry policies", flush=True)
    for policy in policies:
        for case in [samples[0], samples[5], samples[9]]:
            run(case, policy)
    if plausibility._ensure_model():
        plausibility.decide("请确认确认会议地点。", 1, 5, "确认")
    rows = []
    for repetition in range(args.repetitions):
        indices = list(range(len(samples)))
        random.Random(4100+repetition).shuffle(indices)
        for position, index in enumerate(indices):
            order = policies[index%3:]+policies[:index%3]
            if repetition%2:
                order.reverse()
            for policy in order:
                row = run(samples[index], policy)
                row["repetition"] = repetition
                rows.append(row)
                with (output/"observations.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            print(f"round {repetition+1}/{args.repetitions}: {position+1}/{len(samples)}", flush=True)
    summary = summarize(rows)
    pairs = {}
    for row in rows:
        pairs.setdefault((row['id'], row['repetition']), {})[row['policy']] = row
    comparison = dict(pairs=len(pairs), identical_outputs=sum(
        p['zh_itn_web_chain']['output']==p['zh_itn_fast_fallback']['output'] for p in pairs.values()),
        skipped_numeric_retry_opportunities=sum(len(r.get('skipped_numeric_retries', [])) for r in rows),
        output_differences=[dict(id=k[0], repetition=k[1],
            before=p['zh_itn_web_chain']['output'], after=p['zh_itn_fast_fallback']['output'])
            for k,p in pairs.items() if p['zh_itn_web_chain']['output']!=p['zh_itn_fast_fallback']['output']])
    write(output/"summary.json", dict(timing=summary, output_comparison=comparison))
    lines=['# 数字重试策略实验','', '数字校验和数值保护保留；遇到数字校验失败且无占位符故障时，使用既有安全回退，省去模型重试。只在独立实验进程临时替换重试入口，未修改或部署生产代码。','',
        '| 策略 | 平均 ms | 中位 ms | 模型调用 | 重试 | 拒绝 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for policy in policies:
        s=summary['numeric_cleanup'][policy]
        lines.append(f"| {policy} | {s['mean_ms']:.2f} | {s['median_ms']:.2f} | {s['model_calls']} | {s['retries']} | {s['rejected']} |")
    lines+=['', f"36 组配对中最终输出相同：{comparison['identical_outputs']}/{comparison['pairs']}。", '',
        '这是此前混合句集合的针对性回放，无独立泛化或质量提升声明。安全回退保留数字，也可能保留未清理完的口语；未来某些重试本来能成功，这种策略会放弃该机会。', '',
        '不含 ASR、模型加载或线上排队。每条窗口 3 轮，学习门控和实体库关闭。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(timing=summary, output_comparison=comparison),ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
