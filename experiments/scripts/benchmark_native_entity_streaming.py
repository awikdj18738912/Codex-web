#!/usr/bin/env python3
"""Compare native sherpa-onnx streaming entity biasing on fixed PCM inputs.

This is an isolated experiment runner. It reads entity databases in SQLite
read-only mode and never initializes or mutates the application EntityStore.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODEL_DIR = Path("/home/aim0/data/models/ASR/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30")
SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.160)
TAIL_SAMPLES = int(SAMPLE_RATE * 0.660)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_canonicals(db_path: Path, domain: str) -> list[str]:
    """Read enabled canonical names without EntityStore (which may initialize DB)."""
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "protected_entities" not in tables:
            raise ValueError(f"No protected_entities table in {db_path}")
        # Benchmark libraries are per-domain files, but EntityStore writes
        # their records under the default `general` domain. Accept either the
        # row's semantic domain or that file-local default.
        rows = con.execute(
            "SELECT canonical_text FROM protected_entities "
            "WHERE enabled=1 AND domain IN (?, 'general') "
            "ORDER BY priority DESC, id",
            (domain,),
        ).fetchall()
    return list(dict.fromkeys(str(r[0]).strip() for r in rows if r[0] and str(r[0]).strip()))


def token_inventory() -> set[str]:
    result = set()
    with (MODEL_DIR / "tokens.txt").open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rsplit(" ", 1)
            if len(parts) == 2:
                result.add(parts[0])
    return result


def hotword_payload(canonicals: list[str], tokens: set[str]) -> tuple[str, list[str]]:
    """Encode exactly as native cjkchar: bare UTF-8 codepoints, space-delimited.

    sherpa's C++ SplitUtf8 path does not apply SentencePiece's `▁` prefix or
    byte fallback in cjkchar mode. A missing bare character is therefore
    reported unsupported instead of silently substituted with another token.
    """
    encoded, unsupported = [], []
    for term in canonicals:
        chars = [c for c in term if not c.isspace()]
        if not chars or any(c not in tokens for c in chars):
            unsupported.append(term)
            continue
        encoded.append(" ".join(chars))
    return "/".join(encoded), unsupported


def _result_snapshot(recognizer: Any, stream: Any) -> dict[str, Any]:
    result = recognizer.get_result(stream)
    # sherpa-onnx exposes text/tokens/timestamps on current bindings; preserve
    # any additional JSON-safe fields where the binding provides them.
    out: dict[str, Any] = {}
    for name in ("text", "tokens", "timestamps", "json"):
        try:
            value = getattr(result, name)
            if callable(value):
                value = value()
            out[name] = value
        except (AttributeError, TypeError, RuntimeError):
            pass
    if not out:
        out["text"] = str(result)
    return out


def decode(recognizer: Any, pcm_path: str | Path, hotwords: str = "") -> dict[str, Any]:
    """Decode one little-endian float32 PCM file and return streaming trace.

    Input audio is submitted in 160 ms chunks. A 660 ms zero tail is submitted
    only to flush the algorithm and is included in finish timing, never in the
    reported source audio duration. `compute_elapsed` excludes deliberate waits.
    """
    import numpy as np

    path = Path(pcm_path)
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"PCM byte length is not divisible by float32: {path}")
    audio = np.frombuffer(raw, dtype="<f4").copy()
    stream = recognizer.create_stream(hotwords=hotwords) if hotwords else recognizer.create_stream()
    started = time.perf_counter()
    partials: list[dict[str, Any]] = []
    submitted = 0

    def feed(chunk: Any, is_padding: bool = False) -> None:
        nonlocal submitted
        if not len(chunk):
            return
        stream.accept_waveform(SAMPLE_RATE, chunk)
        submitted += len(chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
            snap = _result_snapshot(recognizer, stream)
            partials.append({
                "input_audio_seconds": submitted / SAMPLE_RATE,
                "compute_elapsed": time.perf_counter() - started,
                "is_flush_padding": is_padding,
                **snap,
            })

    for offset in range(0, len(audio), CHUNK_SAMPLES):
        feed(audio[offset:offset + CHUNK_SAMPLES])
    compute_seconds = time.perf_counter() - started
    tail_start = time.perf_counter()
    feed(np.zeros(TAIL_SAMPLES, dtype=np.float32), is_padding=True)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
        partials.append({
            "input_audio_seconds": submitted / SAMPLE_RATE,
            "compute_elapsed": time.perf_counter() - started,
            "is_flush_padding": True,
            **_result_snapshot(recognizer, stream),
        })
    final = _result_snapshot(recognizer, stream)
    finish_seconds = time.perf_counter() - tail_start
    total_compute_seconds = time.perf_counter() - started
    return {
        "final": final,
        "partials": partials,
        "audio_seconds": len(audio) / SAMPLE_RATE,
        "submitted_audio_seconds_including_flush": submitted / SAMPLE_RATE,
        "flush_padding_seconds": TAIL_SAMPLES / SAMPLE_RATE,
        "compute_seconds_before_flush": compute_seconds,
        "finish_seconds_including_flush": finish_seconds,
        "compute_seconds_total": total_compute_seconds,
        "timing_scope": "accelerated; no intentional sleeps; flush includes 0.66s zero padding",
    }


def make_recognizer(condition: str, score: float):
    import sherpa_onnx

    common = dict(
        tokens=str(MODEL_DIR / "tokens.txt"),
        encoder=str(MODEL_DIR / "encoder.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder.onnx"),
        joiner=str(MODEL_DIR / "joiner.int8.onnx"),
        num_threads=2,
        enable_endpoint_detection=False,
        modeling_unit="cjkchar",
        provider="cpu",
    )
    if condition == "greedy":
        common["decoding_method"] = "greedy_search"
    else:
        common.update(decoding_method="modified_beam_search", max_active_paths=4)
    common["hotwords_score"] = score
    return sherpa_onnx.OnlineRecognizer.from_transducer(**common)


def choose_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return rows
    return [r for r in rows if r.get("split") == split]


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    inputs_path = run_dir / "inputs.jsonl"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Missing prepared inputs: {inputs_path}")
    rows = choose_rows(read_jsonl(inputs_path), args.split)
    rows = rows[args.shard_index::args.num_shards]
    if args.limit is not None:
        rows = rows[:args.limit]
    destination = run_dir / (f"native_results_shard{args.shard_index}.jsonl" if args.num_shards>1 else "native_results.jsonl")
    previous = read_jsonl(run_dir / "native_results.jsonl")
    if destination.name != 'native_results.jsonl':previous += read_jsonl(destination)
    done = {(r["id"], r["condition"], r.get("score")) for r in previous}
    tokens = token_inventory()
    recognizers: dict[tuple[str, float], tuple[Any, float]] = {}
    added = 0
    for index, row in enumerate(rows, 1):
        pcm_path = Path(row["pcm_path"])
        if not pcm_path.is_absolute():
            pcm_path = (ROOT / pcm_path).resolve()
        if not pcm_path.exists():
            raise FileNotFoundError(pcm_path)
        actual_hash = sha256(pcm_path)
        expected_hash = row.get("pcm_sha256", row.get("hash"))
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"PCM hash mismatch for {row['id']}: {actual_hash} != {expected_hash}")
        canonicals: list[str] = []
        unsupported: list[str] = []
        payload = ""
        if "hotword" in args.conditions:
            library = Path(row.get("library_path") or row.get("library") or "")
            if not library.is_absolute():
                library = (ROOT / library).resolve()
            canonicals = load_canonicals(library, row["domain"])
            if not canonicals:
                raise ValueError(f"No enabled canonical entities in {library} for domain={row['domain']!r} or 'general'")
            payload, unsupported = hotword_payload(canonicals, tokens)
            if not payload:
                raise ValueError(f"No canonical names are representable as cjkchar in {library}")
        for condition in args.conditions:
            # Score sweeps apply only to native hotword biasing. Keep one
            # no-hotword control run for each decoder regardless of sweep size.
            condition_scores: list[float | None] = args.scores if condition == "hotword" else [None]
            for output_score in condition_scores:
                score = float(output_score if output_score is not None else 1.5)
                key = (row["id"], condition, output_score)
                if key in done:
                    continue
                rk = (condition, score)
                if rk not in recognizers:
                    load_started = time.perf_counter()
                    recognizers[rk] = (make_recognizer(condition, score), time.perf_counter() - load_started)
                recognizer, load_seconds = recognizers[rk]
                trace = decode(recognizer, pcm_path, payload if condition == "hotword" else "")
                record = {
                    "id": row["id"], "condition": condition, "score": output_score,
                    "split": row.get("split"), "domain": row.get("domain"),
                    "pcm_sha256": actual_hash, "audio_seconds": trace.pop("audio_seconds"),
                    "model_load_seconds_first_instance": load_seconds,
                    "hotword_canonical_count": len(canonicals) if condition == "hotword" else 0,
                    "hotword_encoded_count": len(canonicals) - len(unsupported) if condition == "hotword" else 0,
                    "hotword_canonical_coverage": ((len(canonicals) - len(unsupported)) / len(canonicals)) if condition == "hotword" and canonicals else None,
                    "unsupported_canonical": unsupported if condition == "hotword" else [],
                    **trace,
                }
                with destination.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done.add(key)
                added += 1
        if index % 50 == 0:
            print(f"PROGRESS {index}/{len(rows)} samples; appended={added}", flush=True)
    print(f"COMPLETE split={args.split} samples={len(rows)} appended={added} output={destination}", flush=True)


def smoke() -> None:
    """Exercise installed binding/API against model-provided WAV fixtures."""
    import numpy as np
    import soundfile as sf

    wav = MODEL_DIR / "test_wavs" / "0.wav"
    data, rate = sf.read(wav, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"Expected 16 kHz test fixture, got {rate} Hz")
    temp = MODEL_DIR / "test_wavs" / ".benchmark_native_smoke.f32le"
    try:
        np.asarray(data, dtype="<f4").tofile(temp)
        tokens = token_inventory()
        payload, unsupported = hotword_payload(["中国", "北京大学"], tokens)
        for condition in ("greedy", "beam", "hotword"):
            rec = make_recognizer(condition, 1.5)
            result = decode(rec, temp, payload if condition == "hotword" else "")
            print(json.dumps({"condition": condition, "text": result["final"].get("text"),
                              "partials": len(result["partials"]), "hotwords": payload,
                              "unsupported": unsupported}, ensure_ascii=False), flush=True)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/entity_backend_comparison_20261015")
    parser.add_argument("--split", choices=("dev", "test", "control", "all"), default="dev")
    parser.add_argument("--scores", nargs="+", type=float, default=[1.5])
    parser.add_argument("--conditions", nargs="+", choices=("greedy", "beam", "hotword"), default=["greedy", "beam", "hotword"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Run the bundled test_wavs API smoke check")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:parser.error('invalid shard')
    if args.smoke:
        smoke()
    else:
        run(args)


if __name__ == "__main__":
    main()
