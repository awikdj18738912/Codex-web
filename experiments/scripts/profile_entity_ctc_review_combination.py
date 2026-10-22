#!/usr/bin/env python3
"""Paired wallclock-paced Qwen profile for CTC plus audio entity review."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.evaluate_entity_ctc_fusion import definitions
from experiments.scripts.evaluate_entity_ctc_review_combination import ASR_URL, LiveAsr
from system.entity_audio_repair import EntityAudioRepair
from system.entity_ctc_sidecar import EntityCtcRuntime

BASE = ROOT / "results/entity_backend_comparison_20261015"
OLD = ROOT / "results/entity_short_finish_20261015_v1/evaluation"
DEFAULT_OUTPUT = ROOT / "results/entity_ctc_live_integration_20261016/paired_paced_results.jsonl"
PACKET_BYTES = 10240  # 160 ms of 16-kHz mono float32 audio.
PACKET_SECONDS = .160
ARMS = ("off", "on")


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


def short_test_rows(base_dir: Path, old_dir: Path):
    # The old score CSV contains IDs and aggregate columns only. We read only
    # the ID column; references and gold annotations are never loaded here.
    with (old_dir / "sample_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        old_ids = {str(row["id"]) for row in csv.DictReader(f) if row.get("id")}
    inputs = {}
    for row in read_jsonl(base_dir / "inputs.jsonl"):
        sid = str(row.get("id", ""))
        if not sid or sid in inputs:
            raise ValueError(f"Missing or duplicate frozen input id: {sid!r}")
        inputs[sid] = row
    ids = [sid for sid in old_ids if sid in inputs and inputs[sid].get("split") == "test"]
    if len(ids) != 230:
        raise ValueError(f"Expected 230 old short-finish/test IDs, found {len(ids)}")
    ids.sort(key=lambda sid: (hashlib.sha256(("entity-combo-paced-v1:" + sid).encode()).hexdigest(), sid))
    chosen = ids[:8]
    if len(chosen) != 8:
        raise ValueError("Could not select eight paced test IDs")
    return inputs, chosen


class TimedLiveAsr(LiveAsr):
    def __init__(self, client):
        super().__init__(client)
        self.arm_started = None
        self.partial_arrivals = []

    async def __call__(self, endpoint, session_id=None, data=b"", params=None):
        result = await super().__call__(endpoint, session_id, data, params)
        if endpoint == "/stream/chunk" and self.arm_started is not None:
            self.partial_arrivals.append(dict(
                wall_seconds=time.perf_counter() - self.arm_started,
                text=str(result.get("text", "")),
                completed_segments=result.get("completed_segments", [])))
        return result


async def run_arm(client, input_row, entity_definitions, ctc_session, arm):
    pcm_path = Path(str(input_row["pcm_path"]))
    pcm = pcm_path.read_bytes()
    digest = hashlib.sha256(pcm).hexdigest()
    if digest != input_row.get("pcm_sha256"):
        raise ValueError(f"PCM hash mismatch: {input_row['id']}")
    if len(pcm) % 4:
        raise ValueError(f"PCM byte count is not float32 aligned: {input_row['id']}")

    backend = TimedLiveAsr(client)
    guard = EntityAudioRepair(backend, entity_definitions,
        ctc_session=ctc_session, ctc_mode=arm,
        short_finish_max_seconds=16., short_finish_wait=4.)
    began = time.perf_counter()
    backend.arm_started = began
    last_audio_submit = None
    submissions = []
    final = None
    try:
        started = await guard.request("/stream/start", params=dict(language="Chinese", chunk_seconds=2.))
        sid = started["session_id"]
        audio_clock = time.perf_counter()
        chunks = [pcm[offset:offset + PACKET_BYTES] for offset in range(0, len(pcm), PACKET_BYTES)]
        for index, packet in enumerate(chunks):
            available_audio = min((index + 1) * PACKET_SECONDS, len(pcm) / 64000)
            target = audio_clock + available_audio
            wait = target - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            last_audio_submit = time.perf_counter() - began
            submissions.append(dict(audio_seconds=available_audio,
                wall_seconds=last_audio_submit))
            # The WebSocket consumer awaits each ASR packet in this order.
            # The CTC CPU worker still runs independently while Qwen responds.
            await guard.request("/stream/chunk", sid, packet)
        final = await guard.request("/stream/finish", sid)
        finished = time.perf_counter()
        snapshot = guard.snapshot()
        partials = backend.partial_arrivals
        nonempty = [x["wall_seconds"] for x in partials if x["text"].strip()]
        repair = snapshot.get("incremental", {})
        return dict(id=input_row["id"], split=input_row["split"], domain=input_row.get("domain"),
            pcm_sha256=digest, audio_seconds=len(pcm) / 64000, arm=arm,
            arm_order_index=None, first_nonempty_text_wall_seconds=min(nonempty) if nonempty else None,
            last_audio_submit_seconds=last_audio_submit,
            audio_clock_wall_seconds=audio_clock-began,
            chunk_submissions=submissions,
            finish_tail_seconds=(finished - began - last_audio_submit) if last_audio_submit is not None else None,
            total_wall_seconds=finished - began,
            raw_primary_text=backend.primary_final,
            final_text=str(final.get("text", "")), partial_arrivals=partials,
            audio_review_checks=int(snapshot.get("checks", 0) or 0),
            short_finish_checks=int(repair.get("short_finish_checks", 0) or 0),
            short_finish_errors=int(repair.get("short_finish_errors", 0) or 0),
            short_finish_timeouts=int(repair.get("short_finish_timed_out", 0) or 0),
            ctc=dict(repair.get("ctc", {})),
            entity_audio_repair=snapshot,
            pacing=dict(packet_bytes=PACKET_BYTES, packet_seconds=PACKET_SECONDS,
                        schedule="new PCM arrives at its recorded audio position; WebSocket ASR requests remain ordered"),
            timing_scope="Wallclock-paced actual Qwen HTTP plus CPU CTC/review; includes stream start and finish")
    finally:
        guard.close()
        await backend.close()


async def profile(base_dir: Path, old_dir: Path, output: Path, limit: int | None):
    inputs, chosen = short_test_rows(base_dir, old_dir)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        chosen = chosen[:limit]

    existing = read_jsonl(output) if output.exists() else []
    done = set()
    for row in existing:
        key = (str(row.get("id", "")), str(row.get("arm", "")))
        if key in done:
            raise ValueError(f"Duplicate existing profile result: {key}")
        done.add(key)

    selected = [(sid, arm) for i, sid in enumerate(chosen)
                for arm in (("off", "on") if i % 2 == 0 else ("on", "off"))
                if (sid, arm) not in done]
    if not selected:
        print("PAIRED_PACED_COMPLETE", len(chosen), flush=True)
        return

    # Resolve glossaries before timing. The model and acoustic context are
    # shared, and an individual CTC session is prepared before its arm starts.
    loaded_definitions = {}
    for sid in chosen:
        row = inputs[sid]
        path = str(Path(row["library_path"]).resolve())
        if path not in loaded_definitions:
            loaded_definitions[path] = definitions(path)
    model_dir = base_dir / "models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30"
    runtime = EntityCtcRuntime(model_dir) if any(arm == "on" for _, arm in selected) else None

    output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        health = (await client.get(ASR_URL + "/health")).raise_for_status().json()
        if health.get("ok") is not True:
            raise RuntimeError("Qwen ASR is unhealthy")
        # Preserve the per-ID alternating order even when resuming a partial run.
        for sid in chosen:
            i = chosen.index(sid)
            order = ("off", "on") if i % 2 == 0 else ("on", "off")
            for arm in order:
                if (sid, arm) in done:
                    continue
                row = inputs[sid]
                ctc = None
                if arm == "on":
                    lib = str(Path(row["library_path"]).resolve())
                    ctc = await asyncio.to_thread(runtime.create_session, loaded_definitions[lib])
                    if ctc is None:
                        raise RuntimeError(f"No CTC session created for {sid}")
                lib = str(Path(row["library_path"]).resolve())
                record = await run_arm(client, row, loaded_definitions[lib], ctc, arm)
                record["arm_order_index"] = order.index(arm)
                with output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                done.add((sid, arm))
                print("PAIRED_PACED", len(done), "/", len(chosen) * 2, sid, arm,
                      round(record["total_wall_seconds"], 3), flush=True)
    print("PAIRED_PACED_COMPLETE", len(chosen), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument("--old-dir", type=Path, default=OLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Run only the first n of the fixed eight for a smoke profile")
    args = parser.parse_args()
    asyncio.run(profile(args.base_dir.resolve(), args.old_dir.resolve(), args.output.resolve(), args.limit))


if __name__ == "__main__":
    main()
