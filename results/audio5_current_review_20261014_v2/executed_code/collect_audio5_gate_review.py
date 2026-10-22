"""Fresh complete audio ASR followed by matched-window real Refiner replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.benchmark_fixed_gate_pipeline import receive, sha, write


def request(url, endpoint, session_id=None, data=b"", params=None):
    from urllib.parse import urlencode
    query = dict(params or {})
    if session_id:
        query["session_id"] = session_id
    address = url + endpoint + ("?" + urlencode(query) if query else "")
    result = urllib.request.urlopen(urllib.request.Request(address, data=data,
        headers={"Content-Type": "application/octet-stream"}), timeout=180)
    return json.load(result)


def collect(args):
    import numpy as np
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "asr_final.json").exists():
        raise FileExistsError("Never overwrite a fresh ASR collection")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(args.audio), "-ac", "1",
        "-ar", "16000", "-f", "f32le", "pipe:1"], check=True, stdout=subprocess.PIPE).stdout
    pcm = np.frombuffer(decoded, dtype="float32")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(args.audio), "-ac", "1", "-ar",
        "16000", "-c:a", "pcm_s16le", str(out / "audio_16k.wav")], check=True)
    health = json.load(urllib.request.urlopen(args.asr_url + "/health"))
    if not health.get("confidence_window_scores"):
        raise ValueError("Source-aligned confidence service required")
    write(out / "audio_provenance.json", dict(source_audio=str(args.audio.resolve()),
        source_audio_sha256=sha(args.audio), decoded_samples=len(pcm), sample_rate=16000,
        decoded_float32_sha256=hashlib.sha256(decoded).hexdigest(), seconds=len(pcm)/16000,
        full_audio=True, ASR_health=health, context="", original_ASR_without_refiner=True,
        segment_times="request-end audio bounds, not word-level forced alignment"))
    started = time.perf_counter()
    session = request(args.asr_url, "/stream/start", params={"language": "Chinese"})["session_id"]
    events, previous_end = [], 0.
    try:
        for index, offset in enumerate(range(0, len(decoded), 64000)):
            payload = request(args.asr_url, "/stream/chunk", session, decoded[offset:offset+64000])
            end = min(offset+64000, len(decoded))/64000
            for event in payload.get("completed_segments", []):
                events.append(dict(**event, audio_start_bound=previous_end, audio_end_bound=end))
                previous_end = end
            if (index+1) % 30 == 0:
                write(out / "asr_progress.json", dict(audio_seconds=end, elapsed_seconds=time.perf_counter()-started,
                                                     completed_segments=len(events), text_chars=len(payload["text"])))
                print("ASR_PROGRESS", round(end,1), len(payload["text"]), flush=True)
        payload = request(args.asr_url, "/stream/finish", session)
        for event in payload.get("completed_segments", []):
            events.append(dict(**event, audio_start_bound=previous_end, audio_end_bound=len(pcm)/16000))
            previous_end = len(pcm)/16000
    except BaseException:
        request(args.asr_url, "/stream/cancel", session)
        raise
    payload["all_completed_segments"] = events
    payload["elapsed_seconds"] = time.perf_counter()-started
    write(out / "asr_final.json", payload)
    (out / "raw_asr.txt").write_text(payload["text"] + "\n")
    print("ASR_DONE", len(payload["text"]), len(events), round(payload["elapsed_seconds"],1), flush=True)


class FrozenASR:
    def __init__(self, final):
        self.final = final

    def __call__(self, url, endpoint, session_id=None, data=b"", params=None):
        if endpoint == "/stream/start":
            return dict(session_id="fresh-audio5-frozen", context_applied=False)
        if endpoint == "/stream/cancel":
            return dict(cancelled=True)
        return {**self.final, "completed_segments": self.final["all_completed_segments"]}


def replay(args):
    import torch
    from fastapi.testclient import TestClient
    from threadpoolctl import threadpool_limits
    from system import web_app
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    out = args.output.resolve()
    final = json.loads((out / "asr_final.json").read_text())
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    configuration = dict(audio=args.audio.name, source_audio_sha256=sha(args.audio),
        frozen_ASR_sha256=sha(out / "asr_final.json"), policies=list(args.policies),
        low_threshold=.8, threshold_frozen_before_refiner=True, missing_evidence="review",
        high_score_veto=False, calibrated=False, learned_veto_disabled_in_supplement=True,
        numeric_backend="zh-itn", entity_db=str(args.entity_db) if args.entity_db else None,
        refiner_model=str(args.model), device=args.device,
        final_refinement_mode="off", window_pipeline="reported_by_current_app_health",
        source_chunks="fresh ASR VAD-completed segments", independent_acceptance=False,
        scope="fresh_complete_ASR_then_frozen_current_refinement_chain",
        upstream_entity_audio_repair=False, primary_policy="deployed_learned",
        human_labels=0, references_passed_to_model=False, production_services_restarted=False)
    write(out / "replay_configuration.json", configuration)
    snapshot = out / "executed_code"
    snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__), ROOT/"system/confidence_supplement.py", ROOT/"system/web_app.py",
                 ROOT/"system/asr_window_confidence.py", ROOT/"system/qwen_asr_stream_server.py",
                 ROOT/"system/refinement_gate.py", ROOT/"system/window_refinement.py"]:
        (snapshot/path.name).write_bytes(path.read_bytes())
    print("LOADING_REFINER", flush=True)
    refiner = TransformersRefiner(args.model, args.device, 256)
    configuration["actual_refiner_dtype"] = str(next(refiner._model.parameters()).dtype)
    write(out / "replay_configuration.json", configuration)
    plausibility = RepetitionPlausibility(args.device)
    if not plausibility._ensure_model():
        raise RuntimeError("Existing local repetition model unavailable")
    refiner.refine_with_numeric_rules("请确认会议地点。")
    results = []
    with patch.object(web_app,"TransformersRefiner",return_value=refiner), \
         patch.object(web_app,"RepetitionPlausibility",return_value=plausibility), \
         patch.object(web_app,"STREAMING_REFINEMENT_MIN_INTERVAL_SECONDS",0):
        for policy in args.policies:
            if (out / f"{policy}.json").exists():
                raise FileExistsError("Never overwrite a replay arm")
            native = "off" if policy == "all_refine" else "tri_state"
            trace = out / f"{policy}.jsonl"
            with patch.object(web_app,"_stream_request",FrozenASR(final)):
                app = web_app.create_app(args.model,args.device,"http://frozen-asr","Chinese",256,trace,
                    entity_db=args.entity_db, entity_fuzzy_mode="auto", refinement_gate_mode=native,
                    learned_window_gate_mode="on" if policy == "deployed_learned" else "off",
                    confidence_supplement_mode="on" if policy == "supplement" else "off",
                    final_refinement_mode="off",zh_itn_library=args.zh_itn)
                with TestClient(app) as client:
                    write(out / f"{policy}.health.json",client.get("/health").json())
                    started = time.perf_counter()
                    with client.websocket_connect("/ws/stream?mode=streaming&domain=general") as ws:
                        ready = receive(ws,"ready")
                        ws.send_bytes(b"frozen-source-only")
                        receive(ws,"update")
                        ws.send_json({"event":"finish"})
                        result = receive(ws,"final")
                    elapsed = time.perf_counter()-started
            write(out / f"{policy}.json",result)
            rows = [json.loads(line) for line in trace.with_suffix(".stream_trace.jsonl").open()]
            segments = [segment for row in rows if row["event"] == "window_result"
                        for segment in row["segments"] if segment["window_kind"] != "pending_tail"]
            write(out / f"{policy}.windows.json",segments)
            stats = result["refiner_session_stats"]
            if not stats["stats_complete"] or stats["failed_call_count"]:
                raise RuntimeError("Incomplete/failed model calls")
            item = dict(policy=policy, calls=stats["call_count"], seconds=elapsed,
                        windows=len(segments), stats=stats, window_pipeline=ready["window_pipeline"])
            results.append(item)
            write(out/"replay_progress.json",results)
            print("REPLAY_DONE",json.dumps(item,ensure_ascii=False),flush=True)
    write(out/"replay_summary.json",results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("asr", "replay"))
    parser.add_argument("--audio",type=Path,default=ROOT/"测试音频5.mp3")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--asr-url",default="http://127.0.0.1:8876")
    parser.add_argument("--model",type=Path,default=Path("/home/aim0/data/models/ASR/AgenticASR-Refiner"))
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--entity-db",type=Path,default=Path("/home/aim0/桌面/tan/ASR/AgenticASR_entity_e5bedb3/data/entities.db"))
    parser.add_argument("--zh-itn",type=Path,default=ROOT/"zh-itn/zh-itn/build/libzh_itn_bridge.so")
    parser.add_argument("--policies",nargs="+",choices=("native","all_refine","supplement","deployed_learned"),
                        default=["all_refine","native","supplement","deployed_learned"])
    args = parser.parse_args()
    (collect if args.stage == "asr" else replay)(args)
