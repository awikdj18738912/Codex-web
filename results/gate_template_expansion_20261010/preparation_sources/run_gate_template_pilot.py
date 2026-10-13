"""Run the new train-only audio/paired pilot, with resumable stages and audit logs.

Owns only the TTS helper it starts; never stops ASR or existing Web services.
Evaluates the previously frozen student without fitting or changing thresholds.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.scripts import synthetic_gate_data as data
from experiments.scripts import collect_gate_pairs as collector
from experiments.scripts.train_weak_refinement_gate import policy_metrics, write_json, write_rows


def summarize(source, checkpoint):
    import torch
    from system.zh_itn_bridge import ZhITNNormalizer
    from system.weak_refinement_gate import WeakRefinementGate
    torch.set_num_threads(2)
    labels, outcomes, protocols = [], {}, {}
    for name, directory in (("window_full", "pairs_full"), ("review_full", "pairs_review_full")):
        path = source / directory
        if not path.exists():
            continue
        protocol = collector.verify_frozen(path)
        protocols[name] = dict(protocol_sha256=data.file_hash(path / "protocol.json"),
                               run_configuration=json.loads((path / "run_configuration.json").read_text()))
        cases = {r["case_id"]: r for r in data.read_rows(path / "cases.jsonl")}
        pairs = {r["case_id"]: r for r in data.read_rows(path / "pairs.jsonl")}
        norm = ZhITNNormalizer(path / "native/libzh_itn_bridge.so")
        normalized_cache = {}
        def normalize(text):
            if text not in normalized_cache:
                normalized_cache[text] = data.lexical(norm.normalize(text).text)
            return normalized_cache[text]
        for row in data.read_rows(path / "weak_gate_labels.jsonl"):
            row["pair_set"] = name
            assert row["split"] == "train" and not row["human_gold"]
            labels.append(row)
            case, pair = cases[row["case_id"]], pairs[row["case_id"]]
            intended = case["synthetic_metadata"]["intended_text"]
            spoken = case["synthetic_metadata"]["spoken_text"]
            target, spoken_target = normalize(intended), normalize(spoken)
            base = pair["arms"]["rules"]["comparison_text"]
            arms = {}
            for arm, value in pair["arms"].items():
                text = value["comparison_text"]
                effect, _ = data.weak_label(base, text, intended, lambda x: norm.normalize(x).text)
                calls = value["logical_calls"] + value["plausibility_calls"]
                arms[arm] = dict(intended_distance=collector.edit_distance(target, normalize(text)),
                                 spoken_distance=collector.edit_distance(spoken_target, normalize(text)),
                                 exact_intended=target == normalize(text), effect=effect,
                                 logical_refiner_calls=len(value["logical_calls"]),
                                 logical_plausibility_calls=len(value["plausibility_calls"]),
                                 operation_cost_ms=sum(c["wall_ms"] for c in calls), executed=bool(calls),
                                 changed=text != base)
            outcomes[row["case_id"]] = dict(intended_chars=len(target), spoken_chars=len(spoken_target), arms=arms)
    if not labels or len({r["case_id"] for r in labels}) != len(labels):
        raise ValueError("Missing or duplicate paired labels")
    write_rows(source / "weak_gate_labels_train_all.jsonl", labels)
    gate = WeakRefinementGate(checkpoint)
    actions, predictions, elapsed = {}, [], 0.0
    # The model and thresholds are fixed from the previous experiment.
    for row in labels:
        started = time.perf_counter()
        decision = gate.decide(row["input"])
        elapsed += (time.perf_counter() - started) * 1000
        actions[row["case_id"]] = decision["action"]
        predictions.append(dict(case_id=row["case_id"], weak_label=row["label"], decision=decision))
    write_rows(source / "frozen_student_predictions.jsonl", predictions)
    coverage = {}
    for task in ("window_refine", "local_edit_review"):
        rows = [r for r in labels if r["input"]["task_kind"] == task]
        coverage[task] = dict(labels=dict(Counter(r["label"] for r in rows)),
                             harm_template_groups=sorted({r["split_group"] for r in rows if r["label"] == "harm"}),
                             benefit_template_groups=sorted({r["split_group"] for r in rows if r["label"] == "benefit"}),
                             eligible=sum(r["eligible_for_weak_pretraining"] for r in rows))
    policies = {name: policy_metrics(labels, outcomes, lambda r, name=name: name) for name in ("rules", "current", "always")}
    policies["frozen_student"] = policy_metrics(labels, outcomes,
        lambda r: "always" if actions[r["case_id"]] == "refine" else "rules", elapsed)
    report = dict(total=len(labels), labels=dict(Counter(r["label"] for r in labels)), by_task=coverage,
                  eligible=sum(r["eligible_for_weak_pretraining"] for r in labels),
                  unique_recordings=len({r["recording_group"] for r in labels}),
                  unique_template_groups=len({r["split_group"] for r in labels}), policies=policies,
                  frozen_student=dict(path=str(checkpoint.resolve()), weights_sha256=data.file_hash(checkpoint / "gate.pt"),
                                      metadata_sha256=data.file_hash(checkpoint / "metadata.json"), refit=False, threshold_retuned=False),
                  protocols=protocols, human_gold=0, audio_script_verified=0, source_split_used="train_only",
                  source_dev_test_synthesized=False, student_comparison_scope="new_training_candidates_never_used_to_fit_this_student",
                  limits=["Synthetic weak labels; no verified audio or real risk calibration",
                          "Only new source-train audio generated; new dev/test scripts reserved",
                          "Student scores are diagnostic, not a retuned or retrained model",
                          "Frozen decision replay, not a full sequential session or streaming latency"])
    write_json(source / "expansion_report.json", report)
    lines = ["# 新句式训练扩展试点", "", "只合成和解码新数据的 train；原有数据、原型阈值及两批 dev/test 均保留。情景类别不是 benefit/harm 标签，下面标签由实际 rules/always 配对产生。", "",
             f"完成 {len(data.read_rows(source/'tts.jsonl'))} 条 GPU TTS 与 clean ASR，得到 {len(labels)} 个配对案例、{report['unique_template_groups']} 个模板组。", "",
             "| 路径 | benefit | neutral | harm | uncertain | harm 模板组数 |", "|---|---:|---:|---:|---:|---:|"]
    for task, values in coverage.items():
        c = values["labels"]
        lines.append(f"| {task} | {c.get('benefit',0)} | {c.get('neutral',0)} | {c.get('harm',0)} | {c.get('uncertain',0)} | {len(values['harm_template_groups'])} |")
    lines += ["", "## 原模型在新训练候选上的诊断", "", "| 政策 | 精修调用 | 捕获 benefit | weak harm | 目标精确率 |", "|---|---:|---:|---:|---:|"]
    for name, m in policies.items():
        lines.append(f"| {name} | {m['logical_refiner_calls']} | {m['captured_benefit']} | {m['weak_harm']} | {m['exact_intended_fraction']:.3f} |")
    lines += ["", "这是未用于拟合原模型的新训练候选诊断，不是新的最终验收集。原模型保持完全冻结，没有利用新结果调阈值。下一轮必须先核对实际 harm 的任务/模板覆盖和 teacher 版本，再固定训练/开发/独立验收集合。不能通过把正常重复情景直接标作 harm 来凑数。", "",
              "job_state.json 与 logs/ 保存可续跑阶段与日志。TTS 服务仅在合成期间运行；原 ASR/Web 服务没有被本脚本停止。人工金标及已核验音频均为零。", ""]
    (source / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/gate_template_expansion_20261010")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/learned_gate_weak_prototype_20261010/encoder_weighted")
    args = parser.parse_args()
    source = args.source.resolve()
    seeds = data.seeds(source)
    train_count = sum(r["synthetic_split"] == "train" for r in seeds)
    logs = source / "logs"; logs.mkdir(exist_ok=True)
    state_path = source / "job_state.json"
    state = dict(status="running", stage="preflight", train_texts=train_count,
                 source_split_used="train", device=args.device, started_unix=time.time(), refit_student=False,
                 driver_pid=os.getpid(), driver_sha256=data.file_hash(Path(__file__)),
                 frozen_student_weights_sha256=data.file_hash(args.checkpoint / "gate.pt"),
                 frozen_student_metadata_sha256=data.file_hash(args.checkpoint / "metadata.json"))
    (source / "preparation_sources" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    def save():
        state["updated_unix"] = time.time()
        temporary = state_path.with_suffix(".tmp")
        write_json(temporary, state); temporary.replace(state_path)
    def run(stage, command):
        state.update(stage=stage, status="running"); save()
        print("STAGE " + stage, flush=True)
        with (logs / f"{stage}.log").open("a") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        state.setdefault("completed_stages", []).append(stage); save()
    python = sys.executable
    service = None
    service_log = None
    try:
        import httpx
        with httpx.Client(timeout=5) as client:
            client.get("http://127.0.0.1:8766/health").raise_for_status()
        tts_existing = data.read_rows(source / "tts.jsonl")
        if len(tts_existing) < train_count:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", 8011)) == 0:
                    raise RuntimeError("Port 8011 already owned by another process; do not stop or replace it")
            service_log = (logs / "tts_service.log").open("a")
            service = subprocess.Popen([str(ROOT / ".runtime/qwen3-tts-env/bin/python"), "-m", "pipeline.src.speech.qwen_tts_service",
                                        "--device", args.device, "--port", "8011"], cwd=ROOT,
                                       stdout=service_log, stderr=subprocess.STDOUT)
            state["owned_tts_pid"] = service.pid; save()
            deadline = time.monotonic() + 45
            with httpx.Client(timeout=2) as client:
                while True:
                    if service.poll() is not None:
                        raise RuntimeError("Owned TTS helper failed startup; see logs/tts_service.log")
                    try:
                        client.get("http://127.0.0.1:8011/health").raise_for_status(); break
                    except httpx.HTTPError:
                        if time.monotonic() > deadline:
                            raise RuntimeError("TTS helper startup timed out")
                        time.sleep(0.5)
            run("synthesize", [python, "experiments/scripts/synthetic_gate_data.py", "synthesize", "--output", str(source), "--split", "train"])
            service.send_signal(signal.SIGINT); service.wait(timeout=20); service = None
            service_log.close(); service_log = None
            state["owned_tts_stopped"] = True; save()
        tts = data.read_rows(source / "tts.jsonl")
        assert len(tts) == train_count and all(r["synthetic_split"] == "train" for r in tts)
        run("decode", [python, "experiments/scripts/synthetic_gate_data.py", "decode", "--output", str(source), "--split", "train", "--variants", "clean"])
        asr = data.read_rows(source / "asr.jsonl")
        assert len(asr) == train_count and len({r["seed_id"] for r in asr}) == train_count
        assert all(r["synthetic_split"] == "train" and r["variant"] == "clean" for r in asr)
        cases = source / "pairs_full"
        if not (cases / "protocol.json").exists():
            run("prepare_windows", [python, "experiments/scripts/collect_gate_pairs.py", "prepare", "--output", str(cases),
                "--sources", str(source / "asr.jsonl"), "--no-demo-sources", "--window-count", str(train_count), "--review-count", "0", "--seed", "20261014"])
        run("run_windows", [python, "experiments/scripts/collect_gate_pairs.py", "run", "--output", str(cases), "--device", args.device])
        run("label_windows", [python, "experiments/scripts/synthetic_gate_data.py", "label", "--output", str(source), "--pairs-dir", str(cases)])
        reviews = json.loads((cases / "protocol.json").read_text())["pool"]["local_edit_review"]
        state["local_review_candidates"] = reviews; save()
        if reviews:
            review_path = source / "pairs_review_full"
            if not (review_path / "protocol.json").exists():
                run("prepare_reviews", [python, "experiments/scripts/collect_gate_pairs.py", "prepare", "--output", str(review_path),
                    "--sources", str(source / "asr.jsonl"), "--no-demo-sources", "--window-count", "0", "--review-count", str(reviews), "--seed", "20261015"])
            run("run_reviews", [python, "experiments/scripts/collect_gate_pairs.py", "run", "--output", str(review_path), "--device", args.device])
            run("label_reviews", [python, "experiments/scripts/synthetic_gate_data.py", "label", "--output", str(source), "--pairs-dir", str(review_path)])
        state.update(stage="summarize", status="running"); save()
        assert data.file_hash(args.checkpoint / "gate.pt") == state["frozen_student_weights_sha256"]
        assert data.file_hash(args.checkpoint / "metadata.json") == state["frozen_student_metadata_sha256"]
        report = summarize(source, args.checkpoint.resolve())
        state.update(status="complete", stage="complete", labels=report["labels"], total_pairs=report["total"]); save()
        print("COMPLETE " + json.dumps(dict(labels=report["labels"], by_task=report["by_task"]), ensure_ascii=False), flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error)); save()
        raise
    finally:
        if service is not None and service.poll() is None:
            service.send_signal(signal.SIGINT)
            try: service.wait(timeout=20)
            except subprocess.TimeoutExpired:
                service.terminate(); service.wait(timeout=10)
        if service_log is not None: service_log.close()


if __name__ == "__main__":
    main()
