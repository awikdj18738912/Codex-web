#!/usr/bin/env python3
"""Prepare a frozen, source-auditable long SpeechIO set without running ASR."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/entity_local_replay_20261012/v2/unseen_long"
PILOT = ROOT / "results/entity_reliable_data_20261012/pilot/manifest.json"
SRC = ROOT / "results/entity_reliable_data_20261012/source/speechio"
SEED = "entity-local-unseen-long-20261012-v2"
SILENCE = np.zeros(12800, dtype="<f4").tobytes()  # 800 ms at 16 kHz
SCENES = [
    ("football_baggio", "SPEECHIO_ASR_ZH00003", "巴乔", "PERSON"),
    ("football_inter_milan", "SPEECHIO_ASR_ZH00003", "国际米兰", "ORG"),
    ("football_san_siro", "SPEECHIO_ASR_ZH00003", "圣西罗", "LOC"),
    ("football_uefa_cup", "SPEECHIO_ASR_ZH00003", "欧洲联盟杯", "TERM"),
    ("auto_lexus", "SPEECHIO_ASR_ZH00021", "雷克萨斯", "ORG"),
    ("auto_turbo", "SPEECHIO_ASR_ZH00021", "涡轮增压", "TERM"),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rank(key: str, sid: str) -> str:
    return sha(f"{SEED}\0{key}\0{sid}".encode())


def decode(row: dict) -> dict:
    raw = row["audio"]["bytes"]
    with wave.open(__import__("io").BytesIO(raw), "rb") as wf:
        channels, rate, width, frames = wf.getnchannels(), wf.getframerate(), wf.getsampwidth(), wf.getnframes()
        if width != 2 or rate != 16000:
            raise ValueError(f"Unexpected WAV format {row['segment_id']}: {rate}Hz/{width}B")
        samples = np.frombuffer(wf.readframes(frames), dtype="<i2").astype("<f4") / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        pcm = np.asarray(samples, dtype="<f4").tobytes()
    return {"source_id": row["segment_id"], "reference": row["text"],
            "duration_seconds": frames / rate, "raw_audio_sha256": sha(raw),
            "sample_rate": rate, "channels": channels, "frames": frames, "pcm": pcm,
            "speaker": row["speaker"], "original_full_path": row["original_full_path"]}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads(PILOT.read_text(encoding="utf-8"))
    excluded = {x["source_id"] for x in old["samples"]}
    rows: dict[str, list[dict]] = {}
    for source in ("SPEECHIO_ASR_ZH00003", "SPEECHIO_ASR_ZH00021"):
        table = pq.read_table(SRC / f"{source}.parquet").to_pylist()
        rows[source] = [decode(r) for r in table if r["segment_id"] not in excluded]
    all_used: set[str] = set()
    # Candidate-like negatives are selected using reference-only retrieval.
    from system.entity_audio_repair import GenericEntitySelector, _short_expansion_supported
    from system.entity_store import EntityStore
    from system.refinement_guard import join_refined_segments

    cases, references, missing = [], {}, []
    specs = [(x[0], x[1], x[2], x[3], True) for x in SCENES]
    specs += [("negative_haoyue", "SPEECHIO_ASR_ZH00003", "浩悦", "TERM", False),
              ("negative_figo", "SPEECHIO_ASR_ZH00003", "菲戈", "TERM", False)]
    for scene_index, (case_id, source, target, entity_type, positive) in enumerate(specs):
        pool = [r for r in rows[source] if r["source_id"] not in all_used]
        focal = None
        if positive:
            candidates = [r for r in pool if target in r["reference"]]
            if not candidates:
                total = [r for r in rows[source] if target in r["reference"]]
                missing.append({"id": case_id, "registered_entity": target,
                                "reason": "no eligible reference after excluding pilot source IDs" if total else "no exact target in supplied source references",
                                "source_total_exact_matches": len(total),
                                "excluded_exact_match_count": sum(r["source_id"] in excluded for r in total)})
                continue
            focal = min(candidates, key=lambda r: rank("target-" + target, r["source_id"]))
        # Keep negative components free of the registered canonical.
        fillers = [r for r in pool if target not in r["reference"] and r is not focal]
        if not fillers:
            raise RuntimeError(f"No reference-negative fillers for {case_id}")
        filler_order = sorted(fillers, key=lambda r: rank("filler-" + case_id, r["source_id"]))
        pieces: list[dict] = []
        duration = 0.0
        for row in filler_order:
            if duration >= 91.0:
                break
            pieces.append(row)
            duration += row["duration_seconds"] + 0.8
        if focal:
            # Rotate the target to the beginning, middle, and tail.
            slot = (0, len(pieces) // 2, len(pieces))[sum(1 for c in cases if c.get("positive")) % 3]
            pieces.insert(slot, focal)
        if not 90 <= sum(p["duration_seconds"] + 0.8 for p in pieces) <= 120:
            raise RuntimeError(f"Duration outside 90-120 seconds for {case_id}")
        if len({p["source_id"] for p in pieces}) != len(pieces):
            raise RuntimeError(f"Duplicate components in {case_id}")
        if any(target in p["reference"] for p in pieces if p is not focal):
            raise RuntimeError(f"Unexpected target in filler for {case_id}")
        all_used.update(p["source_id"] for p in pieces)
        pcm = b"".join(p["pcm"] + SILENCE for p in pieces)
        audio_path = OUT / f"{case_id}.f32le"
        audio_path.write_bytes(pcm)
        store = EntityStore(OUT / "libraries" / case_id / "entities.db")
        store.upsert_entity(target, entity_type=entity_type, normalization_policy="normalize")
        if not positive:
            selector = GenericEntitySelector(store.list_entities())
            options = []
            for part_index, part in enumerate(pieces):
                for candidate in selector.select(part["reference"])[0]:
                    if (candidate["retrieval_score"] >= 0.63 and
                            _short_expansion_supported(candidate)):
                        options.append((rank("near-" + case_id, part["source_id"]), part_index, candidate))
            focus = min(options, default=None, key=lambda x: x[0])
        else:
            focus = None
        case = {
            "id": case_id, "kind": "positive" if positive else "negative",
            "positive": positive, "pcm_path": str(audio_path), "pcm_sha256": sha(pcm),
            "format": "f32le mono 16000 Hz", "duration_seconds": len(pcm) / 64000,
            "library": str(store.path), "library_sha256": sha(store.path.read_bytes()),
            "registered_entity": target, "entity_type": entity_type,
            "target_insertion_index": pieces.index(focal) if focal else None,
            "focus_reference_candidate": ({"segment_index": focus[1], **focus[2]} if focus else None),
            "components": [{k: p[k] for k in ("source_id", "raw_audio_sha256", "sample_rate", "channels", "frames", "duration_seconds", "speaker", "original_full_path")} for p in pieces],
            "source_ids": [p["source_id"] for p in pieces],
        }
        cases.append(case)
        refs = [p["reference"] for p in pieces]
        joined = join_refined_segments(refs)
        references[case_id] = {
            "reference": joined, "registered_entity": target,
            "gold_occurrences": joined.count(target),
            "caveat": "Real human speech clips concatenated with 800 ms silence; this is a replay stress stream, not naturally continuous original audio. No ASR or listening validation was performed.",
        }
    manifest = {"dataset": "entity_local_replay_v2_unseen_long", "seed": SEED,
                "selection": "fixed SHA-256 rank over source IDs, references, and duration only; excludes source IDs in the 96-row pilot manifest",
                "excluded_pilot_source_id_count": len(excluded), "case_count": len(cases),
                "unique_component_source_id_count": len(all_used), "missing_required_scenes": missing, "cases": cases}
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "references.json").write_text(json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {p.name: sha(p.read_bytes()) for p in [manifest_path, OUT / "references.json"]}
    hashes.update({case["id"] + ".f32le": case["pcm_sha256"] for case in cases})
    (OUT / "frozen_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (OUT / "READY").write_text("Prepared from source references and audio metadata only. No ASR run; no human listening validation.\n")
    print(json.dumps({"cases": len(cases), "positive": sum(c["positive"] for c in cases),
                      "negative": sum(not c["positive"] for c in cases),
                      "unique_source_ids": len(all_used), "frozen_sha256": sha(manifest_path.read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
