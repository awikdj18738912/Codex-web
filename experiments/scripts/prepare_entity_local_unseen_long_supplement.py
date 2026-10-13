#!/usr/bin/env python3
"""Prepare three supplemental long SpeechIO scenes from unseen references."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/entity_local_replay_20261012/v2/unseen_long_supplement"
SOURCE = ROOT / "results/entity_reliable_data_20261012/source/speechio"
PILOT = ROOT / "results/entity_reliable_data_20261012/pilot/manifest.json"
BASE = ROOT / "results/entity_local_replay_20261012/v2/unseen_long/manifest.json"
SEED = "entity-local-unseen-long-supplement-20261012-v1"
SILENCE = np.zeros(12800, dtype="<f4").tobytes()
SCENES = [
    ("football_argentina", "SPEECHIO_ASR_ZH00003", "阿根廷", "LOC"),
    ("auto_bmw", "SPEECHIO_ASR_ZH00021", "宝马", "ORG"),
    ("auto_four_wheel_drive", "SPEECHIO_ASR_ZH00021", "四驱系统", "TERM"),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rank(key: str, sid: str) -> str:
    return sha(f"{SEED}\0{key}\0{sid}".encode())


def decode(row: dict) -> dict:
    raw = row["audio"]["bytes"]
    with wave.open(io.BytesIO(raw), "rb") as wav:
        channels, rate, width, frames = wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getnframes()
        if width != 2 or rate != 16000:
            raise ValueError(f"Unexpected WAV format for {row['segment_id']}: {rate}Hz, {width} bytes")
        samples = np.frombuffer(wav.readframes(frames), dtype="<i2").astype("<f4") / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
    return {"source_id": row["segment_id"], "reference": row["text"],
            "duration_seconds": frames / rate, "raw_audio_sha256": sha(raw),
            "sample_rate": rate, "channels": channels, "frames": frames,
            "pcm": np.asarray(samples, dtype="<f4").tobytes(), "speaker": row["speaker"],
            "original_full_path": row["original_full_path"]}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    prior = json.loads(BASE.read_text(encoding="utf-8"))
    excluded = {r["source_id"] for r in pilot["samples"]}
    excluded |= {sid for case in prior["cases"] for sid in case["source_ids"]}
    from system.entity_store import EntityStore
    from system.refinement_guard import join_refined_segments

    cases, references, missing = [], {}, []
    for case_id, source, target, entity_type in SCENES:
        source_rows = pq.read_table(SOURCE / f"{source}.parquet").to_pylist()
        all_targets = [r for r in source_rows if target in r["text"]]
        rows = [decode(r) for r in source_rows if r["segment_id"] not in excluded]
        targets = [r for r in rows if target in r["reference"]]
        if not targets:
            missing.append({"id": case_id, "registered_entity": target,
                            "source_all_matches": len(all_targets),
                            "excluded_used": sum(r["segment_id"] in excluded for r in all_targets),
                            "available_unseen": sum(r["segment_id"] not in excluded for r in all_targets)})
            continue
        focal = min(targets, key=lambda r: rank("target-" + target, r["source_id"]))
        fillers = sorted((r for r in rows if target not in r["reference"] and r is not focal),
                         key=lambda r: rank("filler-" + case_id, r["source_id"]))
        pieces, duration = [], 0.0
        for row in fillers:
            if duration >= 91:
                break
            pieces.append(row)
            duration += row["duration_seconds"] + .8
        slot = (0, len(pieces) // 2, len(pieces))[(len(cases)) % 3]
        pieces.insert(slot, focal)
        duration = sum(r["duration_seconds"] + .8 for r in pieces)
        if not 90 <= duration <= 120:
            raise RuntimeError(f"Duration outside 90-120 seconds for {case_id}: {duration}")
        ids = [r["source_id"] for r in pieces]
        if len(ids) != len(set(ids)) or any(sid in excluded for sid in ids):
            raise RuntimeError(f"Duplicate or excluded source ID in {case_id}")
        pcm = b"".join(r["pcm"] + SILENCE for r in pieces)
        audio_path = OUT / f"{case_id}.f32le"
        audio_path.write_bytes(pcm)
        store = EntityStore(OUT / "libraries" / case_id / "entities.db")
        store.upsert_entity(target, entity_type=entity_type, normalization_policy="normalize")
        cases.append({
            "id": case_id, "kind": "positive", "pcm_path": str(audio_path),
            "pcm_sha256": sha(pcm), "format": "f32le mono 16000 Hz",
            "duration_seconds": len(pcm) / 64000, "library": str(store.path),
            "library_sha256": sha(store.path.read_bytes()), "registered_entity": target,
            "entity_type": entity_type, "target_insertion_index": pieces.index(focal),
            "source_ids": ids,
            "components": [{k: r[k] for k in ("source_id", "raw_audio_sha256", "sample_rate", "channels", "frames", "duration_seconds", "speaker", "original_full_path")} for r in pieces],
        })
        joined = join_refined_segments(r["reference"] for r in pieces)
        references[case_id] = {"reference": joined, "registered_entity": target,
                               "gold_occurrences": joined.count(target),
                               "caveat": "Real human speech clips concatenated with 800 ms silence; not naturally continuous original audio. No ASR or listening validation was performed."}
    manifest = {"dataset": "entity_local_replay_v2_unseen_long_supplement", "seed": SEED,
                "selection": "fixed SHA-256 rank over source IDs, references, and duration only; excludes 96 pilot IDs and all 83 components of the original five scenes",
                "case_count": len(cases), "unique_component_source_id_count": len({sid for c in cases for sid in c["source_ids"]}),
                "missing_scenes": missing, "cases": cases}
    mp = OUT / "manifest.json"
    rp = OUT / "references.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rp.write_text(json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {mp.name: sha(mp.read_bytes()), rp.name: sha(rp.read_bytes())}
    hashes.update({c["id"] + ".f32le": c["pcm_sha256"] for c in cases})
    (OUT / "frozen_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (OUT / "READY").write_text("Prepared from source references and audio metadata only. No ASR run; no human listening validation.\n")
    print(json.dumps({"cases": len(cases), "unique_source_ids": manifest["unique_component_source_id_count"],
                      "durations": {c["id"]: c["duration_seconds"] for c in cases},
                      "manifest_sha256": hashes["manifest.json"]}, indent=2))


if __name__ == "__main__":
    main()
