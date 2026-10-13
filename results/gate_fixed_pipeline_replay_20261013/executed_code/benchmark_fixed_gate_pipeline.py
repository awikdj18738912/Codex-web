"""Replay consumed ASR text through the real Refiner with identical windows.

This measures the refinement chain only. It does not measure audio/ASR,
browser upload latency, or independent transcription accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
POLICIES = ("all_refine", "native", "learned")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class ReplayASR:
    def __init__(self, text):
        self.text = text

    def __call__(self, url, endpoint, session_id=None, data=b"", params=None):
        if endpoint == "/stream/start":
            return {"session_id": "fixed-source-replay"}
        if endpoint == "/stream/cancel":
            return {"cancelled": True}
        result = {"text": self.text, "language": "Chinese"}
        if endpoint == "/stream/chunk":
            result["completed_segments"] = [{"segment_id": 1, "text": self.text, "vad_boundary": True}]
        return result


def receive(ws, target):
    for _ in range(200):
        payload = ws.receive_json()
        if payload["event"] == "error":
            raise RuntimeError(str(payload))
        if payload["event"] == target:
            return payload
    raise RuntimeError("Missing " + target)


def run(args):
    import torch
    from fastapi.testclient import TestClient
    from threadpoolctl import threadpool_limits
    from system import web_app
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    from system.learned_window_gate import LearnedWindowGate

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    sources = [json.loads(line) for line in args.sources.open()]
    cases = [sources[index] for index in args.indices]
    write(output / "sources.json", cases)
    executed = output / "executed_code"
    executed.mkdir()
    files = [Path(__file__), ROOT / "system/web_app.py", ROOT / "system/window_refinement.py",
             ROOT / "system/refinement_gate.py", ROOT / "system/learned_window_gate.py"]
    for file in files:
        (executed / file.name).write_bytes(file.read_bytes())
    gate = LearnedWindowGate("on")
    if not gate.status()["ready"]:
        raise RuntimeError("Frozen learning gate not ready: " + str(gate.status()))
    configuration = dict(scope="consumed_ASR_text_replay_refinement_chain_only", device=args.device,
                         repetitions=args.repetitions, policies=POLICIES, sources=str(args.sources),
                         sources_sha256=sha(args.sources), indices=args.indices,
                         source_code_sha256={str(file.relative_to(ROOT)): sha(file) for file in files},
                         refiner_weights_sha256={file.name: sha(file) for file in args.model.glob("*.safetensors")},
                         learned_gate=gate.status(), numeric_backend="zh-itn", entity_db=None,
                         final_refinement_mode="off", cache="fresh_per_session",
                         warmup="all models prewarmed before measured sessions", test_tuning=False)
    write(output / "configuration.json", configuration)
    refiner = TransformersRefiner(args.model, args.device, 256)
    plausibility = RepetitionPlausibility(args.device)
    if not plausibility._ensure_model():
        raise RuntimeError("Local repetition model unavailable")
    refiner.refine_with_numeric_rules("请确认会议地点。")
    plausibility.decide("请确认确认会议地点。", 1, 5, "确认")
    gate.decide(dict(task_kind="window_refine", raw_text="请确认会议地点。",
                     baseline_text="请确认会议地点。", available_context="", domain="general"), "refine")
    results = []
    with patch.object(web_app, "TransformersRefiner", return_value=refiner), \
         patch.object(web_app, "RepetitionPlausibility", return_value=plausibility), \
         patch.object(web_app, "STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS", 0):
        for repetition in range(args.repetitions):
            for index, case in enumerate(cases):
                order = list(POLICIES[index % 3:] + POLICIES[:index % 3])
                if repetition % 2:
                    order.reverse()
                for policy in order:
                    trace = output / f"{index:02d}.{repetition}.{policy}.jsonl"
                    mode = "off" if policy == "all_refine" else "tri_state"
                    actual_gate = gate if policy == "learned" else LearnedWindowGate("off")
                    before = gate.status()
                    with patch.object(web_app, "LearnedWindowGate", return_value=actual_gate), \
                         patch.object(web_app, "_stream_request", ReplayASR(case["raw_text"])):
                        app = web_app.create_app(args.model, args.device, "http://replayed-asr", "Chinese", 256,
                                                 trace, refinement_gate_mode=mode,
                                                 learned_window_gate_mode="on" if policy == "learned" else "off",
                                                 final_refinement_mode="off", zh_itn_library=args.zh_itn)
                        with TestClient(app) as client:
                            started = time.perf_counter()
                            with client.websocket_connect("/ws/stream?mode=streaming") as ws:
                                ready = receive(ws, "ready")
                                ws.send_bytes(b"replay-only")
                                receive(ws, "update")
                                ws.send_json({"event": "finish"})
                                final = receive(ws, "final")
                            elapsed = time.perf_counter() - started
                    after = gate.status()
                    rows = [json.loads(line) for line in trace.with_suffix(".stream_trace.jsonl").open()]
                    windows = [(segment["source"], segment["window_kind"], segment["window_chunk_count"])
                               for row in rows if row["event"] == "window_result"
                               for segment in row["segments"] if segment["window_kind"] != "pending_tail"]
                    result = dict(index=index, seed_id=case["seed_id"], repetition=repetition, policy=policy,
                                  raw_text=case["raw_text"], final_text=final["clean_text"], windows=windows,
                                  elapsed_seconds=elapsed, window_pipeline=ready["window_pipeline"],
                                  stats=final["refiner_session_stats"], final=final,
                                  learner_scored=after["scored_windows"] - before["scored_windows"],
                                  learner_vetoed=after["vetoed_windows"] - before["vetoed_windows"])
                    if not result["stats"]["stats_complete"] or result["stats"]["failed_call_count"]:
                        raise RuntimeError("Incomplete/failed calls: " + str(result["stats"]))
                    with (output / "sessions.jsonl").open("a") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results.append(result)
                    print(json.dumps(dict(index=index, repetition=repetition, policy=policy,
                                          calls=result["stats"]["call_count"], seconds=round(elapsed, 3),
                                          vetoes=result["learner_vetoed"])), flush=True)
    groups = defaultdict(list)
    for result in results:
        groups[(result["index"], result["repetition"])].append(result)
    matched = all(len(group) == 3 and all(row["windows"] == group[0]["windows"]
                  and row["window_pipeline"] == group[0]["window_pipeline"] for row in group)
                  for group in groups.values())
    if not matched:
        raise RuntimeError("Window pipeline differs across policies")
    summary = dict(scope=configuration["scope"], matched_windows=matched, cases=len(cases),
                   sessions=len(results), repetitions=args.repetitions,
                   limitations=["Consumed regression data; no independent accuracy claim",
                                "No audio decoding, ASR timing, upload, or user-facing total latency",
                                "Small warm replay; timing is descriptive only"], policies={})
    for policy in POLICIES:
        selected = [row for row in results if row["policy"] == policy]
        summary["policies"][policy] = dict(calls=sum(row["stats"]["call_count"] for row in selected),
            refiner_wall_ms=sum(row["stats"]["completed_wall_latency_ms"] for row in selected),
            chain_total_seconds=sum(row["elapsed_seconds"] for row in selected),
            chain_median_seconds=statistics.median(row["elapsed_seconds"] for row in selected),
            learner_scored=sum(row["learner_scored"] for row in selected),
            learner_vetoed=sum(row["learner_vetoed"] for row in selected))
    summary["native_vs_learned_identical_final_pairs"] = sum(
        next(row for row in group if row["policy"] == "native")["final_text"] ==
        next(row for row in group if row["policy"] == "learned")["final_text"] for group in groups.values())
    write(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=Path("/home/aim0/data/models/ASR/AgenticASR-Refiner"))
    parser.add_argument("--sources", type=Path, default=ROOT / "results/gate_clause_repetition_guard_20261012/new_regression/sources.jsonl")
    parser.add_argument("--zh-itn", type=Path, default=ROOT / "zh-itn/zh-itn/build/libzh_itn_bridge.so")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 2, 3, 21, 22, 23, 29, 40])
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    run(args)
