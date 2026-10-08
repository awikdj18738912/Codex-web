"""Compare per-session ASR decode intervals with one-second file transport."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.parse import urlencode

import websockets

from test_offline_streaming import CHUNK_BYTES, decode_audio


async def benchmark(audio: bytes, interval: float, trace_path: Path) -> dict:
    url = "ws://127.0.0.1:8081/ws/stream?" + urlencode({
        "language": "Chinese", "mode": "streaming", "domain": "general",
        "asr_chunk_seconds": interval,
    })
    offset = trace_path.stat().st_size if trace_path.exists() else 0
    records: list[dict] = []
    trace_id = None

    def read_trace() -> None:
        nonlocal offset
        if not trace_path.exists():
            return
        with trace_path.open("rb") as handle:
            handle.seek(offset)
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    offset = position
                    break
                offset = handle.tell()
                record = json.loads(line)
                if record.get("event") not in {
                    "asr_chunk_timing", "asr_hypothesis", "asr_session_started"
                }:
                    continue
                if record.get("trace_id") != trace_id:
                    raise RuntimeError(
                        "Concurrent transcription detected; stopping benchmark client"
                    )
                records.append(record)

    started = time.perf_counter()
    first_text = None
    first_text_chunk = None
    total = (len(audio) + CHUNK_BYTES - 1) // CHUNK_BYTES
    async with websockets.connect(url, max_size=None, ping_timeout=None) as socket:
        ready = json.loads(await asyncio.wait_for(socket.recv(), timeout=120))
        if ready.get("event") != "ready":
            raise RuntimeError(ready)
        trace_id = ready.get("trace_id")
        if ready.get("asr_chunk_seconds") != interval:
            raise RuntimeError(f"Decode interval not applied: {ready}")
        for index in range(total):
            await socket.send(audio[index * CHUNK_BYTES:(index + 1) * CHUNK_BYTES])
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), timeout=330))
                if event.get("event") == "error":
                    raise RuntimeError(event.get("detail"))
                if event.get("event") == "transcript" and event.get("raw_text") and first_text is None:
                    first_text = time.perf_counter() - started
                    first_text_chunk = index + 1
                if event.get("event") == "chunk_ack":
                    break
            if (index + 1) % 15 == 0:
                read_trace()
            if (index + 1) % 60 == 0 or index + 1 == total:
                print(f"decode={interval:g}s {index + 1}/{total}, elapsed={time.perf_counter() - started:.1f}s", flush=True)
        feed_seconds = time.perf_counter() - started
        await socket.send(json.dumps({"event": "finish"}))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=1200))
            if event.get("event") == "error":
                raise RuntimeError(event.get("detail"))
            if event.get("event") == "final":
                result = event
                break
    elapsed = time.perf_counter() - started
    read_trace()
    timings = {r["chunk_index"]: r for r in records if r["event"] == "asr_chunk_timing"}
    hypotheses = [r for r in records if r["event"] == "asr_hypothesis"]
    post_seconds = sum(
        (datetime.fromisoformat(r["captured_at"]) - datetime.fromisoformat(timings[r["chunk_index"]]["captured_at"])).total_seconds()
        for r in hypotheses if r["chunk_index"] in timings
    )
    metrics = {
        "trace_id": trace_id, "decode_interval_s": interval,
        "transport_interval_s": 1, "audio_duration_s": len(audio) / CHUNK_BYTES,
        "elapsed_s": round(elapsed, 3), "feed_s": round(feed_seconds, 3),
        "finish_s": round(elapsed - feed_seconds, 3),
        "first_text_s": round(first_text, 3) if first_text is not None else None,
        "first_text_chunk": first_text_chunk,
        "chunks": len(timings), "hypothesis_updates": len(hypotheses),
        "post_asr_to_hypothesis_sum_s": round(post_seconds, 3),
        "raw_chars": len(result.get("raw_text", "")),
        "clean_chars": len(result.get("clean_text", "")),
    }
    for key in ("request_ms", "model_ms", "lock_wait_ms", "vad_ms"):
        metrics[key.replace("_ms", "_sum_s")] = round(sum(
            r.get("server_timing", {}).get(key, 0) for r in timings.values()
        ) / 1000, 3)
    result["_benchmark"] = metrics
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--decode-interval", type=float, required=True)
    parser.add_argument("--trace", type=Path, default=Path("results/web/session.stream_trace.jsonl"))
    args = parser.parse_args()
    audio = decode_audio(args.audio)
    result = asyncio.run(benchmark(audio, args.decode_interval, args.trace))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
