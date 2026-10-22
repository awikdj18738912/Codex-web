#!/usr/bin/env python3
"""Paired long-form raw Qwen vs entity-repaired wallclock-paced profile.

Input manifest rows contain only inference inputs: pcm_path, pcm_sha256,
library_path, duration_seconds, id, and target_terms. This script records
latency and runtime behavior; it does not score accuracy.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.evaluate_entity_ctc_fusion import definitions
from experiments.scripts.evaluate_entity_ctc_review_combination import ASR_URL, LiveAsr
from system.entity_audio_repair import EntityAudioRepair
from system.entity_ctc_sidecar import EntityCtcRuntime

PACKET_BYTES = 10240  # 160 ms, 16-kHz mono float32.
PACKET_SECONDS = .160
ARMS = ("raw", "entity")
PRODUCTION_SOURCES = ("system/entity_audio_repair.py", "system/entity_streaming_live.py",
                      "system/entity_ctc_sidecar.py",
                      "experiments/scripts/evaluate_entity_ctc_review_combination.py",
                      "experiments/scripts/entity_ctc_frontend.py",
                      "experiments/scripts/entity_ctc_word_spotter.py",
                      "experiments/scripts/evaluate_entity_ctc_fusion.py")


def read_jsonl(path: Path):
    if not path.exists():
        return []
    result = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        result.append(value)
    return result


def append_jsonl(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    files = (Path(__file__), *(ROOT / name for name in PRODUCTION_SOURCES))
    return {str(path.relative_to(ROOT)): digest_file(path) for path in files}


def load_manifest(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a non-empty JSON object or array")
    required = {"pcm_path", "pcm_sha256", "library_path", "duration_seconds", "id", "target_terms"}
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Manifest rows require {sorted(required)}")
        sid = str(row["id"])
        if not sid or sid in seen:
            raise ValueError(f"Missing or duplicate manifest id: {sid!r}")
        seen.add(sid)
        if not isinstance(row["target_terms"], list):
            raise ValueError(f"target_terms must be a list: {sid}")
        library_path = Path(str(row["library_path"])).expanduser().resolve()
        if not library_path.is_file():
            raise ValueError(f"Entity library does not exist or is not a file: {library_path}")
        if row.get("library_sha256") and digest_file(library_path).lower() != str(row["library_sha256"]).lower():
            raise ValueError(f"Entity library SHA256 mismatch: {sid}")
        row["id"] = sid
    return rows


class TimedLiveAsr(LiveAsr):
    def __init__(self, client, event_sink, arm, round_index, sample_id, began):
        super().__init__(client)
        self.event_sink = event_sink
        self.arm, self.round_index, self.sample_id = arm, round_index, sample_id
        self.began = began
        self.primary_first_nonempty_wall = None
        self.primary_text = ""
        self.primary_final_wall = None
        self.external_first_nonempty_wall = None

    def event(self, kind, **fields):
        self.event_sink(dict(event=kind, id=self.sample_id, round=self.round_index,
                             arm=self.arm, wall_seconds=time.perf_counter() - self.began,
                             **fields))

    async def __call__(self, endpoint, session_id=None, data=b"", params=None):
        request_started = time.perf_counter()
        previous_primary_id = self.primary_id
        result = await super().__call__(endpoint, session_id, data, params)
        if endpoint == "/stream/start" and previous_primary_id is not None:
            self.primary_id = previous_primary_id
        response_wall = time.perf_counter() - self.began
        if endpoint == "/stream/chunk" and session_id == self.primary_id:
            text = str(result.get("text", ""))
            if text.strip():
                self.primary_text = text
                if self.primary_first_nonempty_wall is None:
                    self.primary_first_nonempty_wall = response_wall
                    self.event("primary_first_nonempty", text=text)
            self.event("primary_response", text=text,
                       completed_segments=result.get("completed_segments", []),
                       request_seconds=time.perf_counter() - request_started)
        elif endpoint == "/stream/finish" and session_id == self.primary_id:
            self.primary_final_wall = response_wall
            self.primary_text = str(result.get("text", ""))
            self.event("primary_finish_response", text=self.primary_text,
                       request_seconds=time.perf_counter() - request_started)
            if self.primary_text.strip() and self.primary_first_nonempty_wall is None:
                self.primary_first_nonempty_wall = response_wall
                self.event("primary_first_nonempty", text=self.primary_text, source="finish")
        return result

    def observe_external(self, response):
        text = str(response.get("text", ""))
        self.event("external_response", text=text,
                   revision=response.get("entity_audio_repair_revision", 0))
        if text.strip() and self.external_first_nonempty_wall is None:
            self.external_first_nonempty_wall = time.perf_counter() - self.began
            self.event("external_first_nonempty", text=text)


async def run_arm(client, row, defs, ctc_session, round_index, arm, event_sink):
    pcm_path = Path(str(row["pcm_path"])).expanduser().resolve()
    pcm = pcm_path.read_bytes()
    actual_hash = hashlib.sha256(pcm).hexdigest()
    if actual_hash.lower() != str(row["pcm_sha256"]).lower():
        raise ValueError(f"PCM SHA256 mismatch for {row['id']}")
    if len(pcm) % 4:
        raise ValueError(f"PCM is not float32 aligned: {row['id']}")
    audio_seconds = len(pcm) / 64000
    declared = float(row["duration_seconds"])
    if declared <= 0 or abs(declared - audio_seconds) > max(.02, 1 / 16000):
        raise ValueError(f"duration_seconds disagrees with PCM length for {row['id']}: {declared} vs {audio_seconds}")

    began = time.perf_counter()
    backend = TimedLiveAsr(client, event_sink, arm, round_index, row["id"], began)
    snapshots = []

    def record_snapshot(snapshot, origin):
        snapshots.append((time.perf_counter() - began, snapshot))
        backend.event("repair_status", origin=origin, revision=snapshot.get("revision"),
                      snapshot=snapshot)

    async def on_status(snapshot):
        record_snapshot(snapshot, "status_callback")

    if arm == "entity":
        guard = EntityAudioRepair(backend, defs, ctc_session=ctc_session, ctc_mode="on",
                                  on_status=on_status)
        request = guard.request
    else:
        guard = None
        request = backend

    final_result = None
    last_submit = None
    submissions = []
    chunk_request_seconds = 0.
    audio_clock = None
    schedule_lateness_max = 0.
    target_seen_raw_wall = None
    target_seen_published_wall = None
    observed_revision = 0
    targets = [str(term) for term in row["target_terms"] if str(term)]

    def detect_targets(text, wall, published):
        nonlocal target_seen_raw_wall, target_seen_published_wall
        if not text:
            return
        found = [term for term in targets if term in text]
        if found and not published and target_seen_raw_wall is None:
            target_seen_raw_wall = wall
            backend.event("target_first_raw", terms=found, text=text)
        if found and published and target_seen_published_wall is None:
            target_seen_published_wall = wall
            backend.event("target_first_published", terms=found, text=text)

    try:
        started = await request("/stream/start", params=dict(language="Chinese", chunk_seconds=2.))
        sid = started["session_id"]
        audio_clock = time.perf_counter()
        packets = [pcm[pos:pos + PACKET_BYTES] for pos in range(0, len(pcm), PACKET_BYTES)]
        for index, packet in enumerate(packets):
            available_audio = min((index + 1) * PACKET_SECONDS, audio_seconds)
            scheduled = audio_clock + available_audio
            wait = scheduled - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            submit_clock = time.perf_counter()
            last_submit = submit_clock - began
            lateness = max(0., submit_clock - scheduled)
            schedule_lateness_max = max(schedule_lateness_max, lateness)
            packet_meta = dict(index=index, audio_seconds=available_audio,
                               submitted_wall_seconds=last_submit,
                               scheduled_wall_seconds=scheduled - began,
                               cumulative_schedule_lateness_seconds=lateness)
            submissions.append(packet_meta)
            backend.event("packet_submit", **packet_meta)
            request_start = time.perf_counter()
            response = await request("/stream/chunk", sid, packet)
            request_seconds = time.perf_counter() - request_start
            chunk_request_seconds += request_seconds
            packet_meta.update(response_wall_seconds=time.perf_counter() - began,
                               chunk_request_seconds=request_seconds)
            backend.event("packet_response", **packet_meta)
            backend.observe_external(response)
            detect_targets(backend.primary_text, time.perf_counter() - began, False)
            if guard is not None:
                snap = guard.snapshot()
                if guard.revision != observed_revision:
                    observed_revision = guard.revision
                    record_snapshot(snap, "packet_revision_snapshot")
                    backend.event("local_patch_snapshot", patches=list(guard._live_patches),
                                  revision=guard.revision)
                detect_targets(str(response.get("text", "")), time.perf_counter() - began, True)

        finish_called = time.perf_counter()
        backend.event("finish_call", last_packet_submit_wall_seconds=last_submit)
        final_result = await request("/stream/finish", sid)
        finish_returned = time.perf_counter()
        backend.observe_external(final_result)
        if guard is not None:
            snap = guard.snapshot()
            record_snapshot(snap, "final_snapshot")
        final_text = str(final_result.get("text", ""))
        detect_targets(backend.primary_text, finish_returned - began, False)
        detect_targets(final_text, finish_returned - began, arm == "entity")
        if arm == "raw":
            target_seen_published_wall = target_seen_raw_wall
        repair_snapshot = guard.snapshot() if guard is not None else {}
        incremental = repair_snapshot.get("incremental", {})
        changes = repair_snapshot.get("changes", [])
        first_patch = next((wall for wall, snap in snapshots
                            if snap.get("revision", 0) or snap.get("changes")), None)
        output = dict(
            id=row["id"], round=round_index, arm=arm, pcm_sha256=actual_hash,
            duration_seconds=audio_seconds, target_terms=targets,
            first_primary_nonempty_wall_seconds=backend.primary_first_nonempty_wall,
            first_external_text_wall_seconds=backend.external_first_nonempty_wall,
            raw_primary_final_text=backend.primary_final if backend.primary_final is not None else backend.primary_text,
            final_text=final_text, target_first_raw_wall_seconds=target_seen_raw_wall,
            target_first_published_wall_seconds=target_seen_published_wall,
            last_packet_submit_wall_seconds=last_submit,
            finish_call_wall_seconds=finish_called - began,
            finish_return_wall_seconds=finish_returned - began,
            finish_tail_from_last_packet_submit_seconds=(finish_returned - (began + last_submit) if last_submit is not None else None),
            finish_tail_from_finish_call_seconds=finish_returned - finish_called,
            total_wall_seconds=finish_returned - began,
            audio_clock_start_wall_seconds=audio_clock - began,
            chunk_request_total_seconds=chunk_request_seconds,
            max_cumulative_schedule_lateness_seconds=schedule_lateness_max,
            chunk_submissions=submissions, local_patch_first_wall_seconds=first_patch,
            local_patch_snapshots=[{"wall_seconds": wall, "snapshot": snap}
                                   for wall, snap in snapshots if snap.get("revision", 0) or snap.get("changes")],
            ctc=incremental.get("ctc", {}), ctc_stats=incremental,
            audio_review_checks=int(repair_snapshot.get("checks", 0) or 0),
            entity_audio_repair=repair_snapshot,
            pacing={"packet_bytes": PACKET_BYTES, "packet_seconds": PACKET_SECONDS,
                    "first_packet_wait_seconds": PACKET_SECONDS,
                    "schedule": "each packet submitted at cumulative PCM audio time after stream/start"},
        )
        return output
    finally:
        if guard is not None:
            guard.close()
            cleanup_tasks = tuple(getattr(guard, "_cleanup_tasks", ()))
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        await backend.close()


def summarize(results, manifest_rows, rounds):
    keyed = {(str(r.get("id")), int(r.get("round", -1)), str(r.get("arm"))): r for r in results}
    paired = []
    for round_index in range(rounds):
        for row in manifest_rows:
            raw = keyed.get((row["id"], round_index, "raw"))
            entity = keyed.get((row["id"], round_index, "entity"))
            if raw and entity:
                paired.append({"id": row["id"], "round": round_index,
                    "entity_minus_raw_total_wall_seconds": entity.get("total_wall_seconds", 0) - raw.get("total_wall_seconds", 0),
                    "entity_minus_raw_first_text_wall_seconds": (entity.get("first_external_text_wall_seconds") or 0) - (raw.get("first_external_text_wall_seconds") or 0),
                    "entity_minus_raw_finish_tail_seconds": (entity.get("finish_tail_from_finish_call_seconds") or 0) - (raw.get("finish_tail_from_finish_call_seconds") or 0),
                    "entity_final_text": entity.get("final_text", ""), "raw_final_text": raw.get("final_text", "")})
    return {"rounds": rounds, "manifest_count": len(manifest_rows),
            "completed_arms": len(results), "paired_count": len(paired), "paired_deltas": paired,
            "limitations": ["Timing describes this host, this Qwen service, and these PCM inputs.",
                            "The run does not estimate transcription accuracy or generalize beyond this manifest.",
                            "Target-term timestamps are exact substring observations in returned text, not phonetic onset times.",
                            "Each arm is sequential and can be affected by service load and order effects."]}


def write_report(path: Path, summary, health, hashes):
    lines = ["# Long paced entity profile", "",
             "This report compares wall-clock behavior for the frozen manifest. It makes no accuracy claim.", "",
             f"- Manifest rows: {summary['manifest_count']}",
             f"- Requested rounds: {summary['rounds']}",
             f"- Completed arms: {summary['completed_arms']}",
             f"- Complete paired sample-rounds: {summary['paired_count']}",
             f"- Qwen health captured before run: `{json.dumps(health, ensure_ascii=False)}`", "",
             "## Paired deltas", "",
             "Positive values mean entity took longer than raw. These are per-item observations, not accuracy results.", "",
             "| ID | Round | Total wall delta (s) | First text delta (s) | Finish tail delta (s) |",
             "|---|---:|---:|---:|---:|"]
    for p in summary["paired_deltas"]:
        lines.append(f"| {p['id']} | {p['round']} | {p['entity_minus_raw_total_wall_seconds']:.3f} | {p['entity_minus_raw_first_text_wall_seconds']:.3f} | {p['entity_minus_raw_finish_tail_seconds']:.3f} |")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in summary["limitations"]]
    lines += ["", "## Source hashes", "", "```json", json.dumps(hashes, ensure_ascii=False, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(manifest: Path, output_dir: Path, rounds: int):
    if rounds < 1:
        raise ValueError("--rounds must be positive")
    rows = load_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path, events_path = output_dir / "results.jsonl", output_dir / "events.jsonl"
    progress_path = output_dir / "progress.json"
    existing = read_jsonl(results_path)
    done = set()
    for result in existing:
        key = (str(result.get("id")), int(result.get("round", -1)), str(result.get("arm")))
        if key in done:
            raise ValueError(f"Duplicate result row: {key}")
        done.add(key)
    hashes = source_hashes()
    previous = read_jsonl(output_dir / "run_metadata.jsonl")
    manifest_hash = digest_file(manifest)
    if previous and previous[-1].get("source_hashes") != hashes:
        raise ValueError("Source files changed since this output directory was started")
    if previous and previous[-1].get("manifest_sha256") != manifest_hash:
        raise ValueError("Manifest changed since this output directory was started")
    append_jsonl(output_dir / "run_metadata.jsonl", {"source_hashes": hashes,
        "manifest_sha256": manifest_hash, "started_utc": datetime.now(timezone.utc).isoformat()})

    defs_cache = {}
    for row in rows:
        lib = str(Path(row["library_path"]).expanduser().resolve())
        if lib not in defs_cache:
            defs_cache[lib] = definitions(lib)
    model_dir = ROOT / "results/entity_backend_comparison_20261015/models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30"
    runtime = EntityCtcRuntime(model_dir)
    total = len(rows) * rounds * len(ARMS)
    done_count = len(done)
    last_progress = time.monotonic()
    current = {"id": None, "round": None, "arm": None}
    event_file = events_path.open("a", encoding="utf-8")

    def event_sink(event):
        event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        event_file.flush()

    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        health = (await client.get(ASR_URL + "/health")).raise_for_status().json()
        if health.get("ok") is not True:
            raise RuntimeError(f"Qwen ASR is unhealthy: {health}")
        write_json(output_dir / "qwen_health.json", {"observed_utc": datetime.now(timezone.utc).isoformat(), "url": ASR_URL, "health": health})
        async def progress_ticker():
            nonlocal last_progress
            while True:
                await asyncio.sleep(30)
                write_json(progress_path, {"updated_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_arms": done_count, "total_arms": total, "current_id": current["id"],
                    "round": current["round"], "arm": current["arm"], "complete": False})
                print(f"LONG_PACED_PROGRESS {done_count}/{total} round={current['round']} arm={current['arm']} id={current['id']}", flush=True)
                last_progress = time.monotonic()
        ticker = asyncio.create_task(progress_ticker())
        try:
            for round_index in range(rounds):
                order = ARMS if round_index % 2 == 0 else tuple(reversed(ARMS))
                for row in rows:
                    for arm in order:
                        key = (row["id"], round_index, arm)
                        if key in done:
                            continue
                        current.update(id=row["id"], round=round_index, arm=arm)
                        ctc = None
                        defs = defs_cache[str(Path(row["library_path"]).expanduser().resolve())]
                        if arm == "entity":
                            # Model and session setup are explicitly outside run_arm's timer.
                            ctc = await asyncio.to_thread(runtime.create_session, defs)
                            if ctc is None:
                                raise RuntimeError(f"Could not create CTC session for {row['id']}")
                        result = await run_arm(client, row, defs, ctc, round_index, arm, event_sink)
                        result["arm_order_index"] = order.index(arm)
                        append_jsonl(results_path, result)
                        existing.append(result)
                        done.add(key)
                        done_count += 1
                        print(f"LONG_PACED {done_count}/{total} round={round_index} arm={arm} id={row['id']} wall={result['total_wall_seconds']:.2f}s", flush=True)
            write_json(progress_path, {"updated_utc": datetime.now(timezone.utc).isoformat(),
                "completed_arms": done_count, "total_arms": total, "complete": True})
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
            event_file.close()
    summary = summarize(existing, rows, rounds)
    summary["source_hashes"] = hashes
    summary["qwen_health"] = health
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "REPORT.md", summary, health, hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON array of frozen PCM and entity-library rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(run(args.manifest.resolve(), args.output_dir.resolve(), args.rounds))


if __name__ == "__main__":
    main()
