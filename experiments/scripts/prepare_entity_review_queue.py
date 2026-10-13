#!/usr/bin/env python3
"""Prepare a deterministic, offline entity-review listening queue.

Reads only the frozen benchmark samples/audio log and the frozen 1,346-entry
entity database. It does not train, infer, update data, or modify production.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/large_entity_benchmark_v1_20261010"
SAMPLES_PATH = DATA / "dataset/samples.jsonl"
AUDIO_PATH = DATA / "full_evaluation/audio.jsonl"
LEXICON_PATH = DATA / "dataset/library/entities.db"
OUT_DIR = ROOT / "results/entity_conflict_veto_20261012"
QUEUE_PATH = OUT_DIR / "review_queue.jsonl"
MANIFEST_PATH = OUT_DIR / "review_queue_manifest.json"
NEW_CANDIDATE_LIMIT = 196
HASH_SCAN_TARGET = 392
HASH_SALT = "entity-review-queue-v1-20261012"
EXPECTED_ABSENT_RECORDINGS = {
    "BAC009S0770W0368",
    "BAC009S0907W0443",
    "BAC009S0912W0420",
    "BAC009S0907W0447",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def stable_hash(*parts: object) -> str:
    value = "\x1f".join([HASH_SALT, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_definitions():
    # Import the current selector and definition type; the database connection
    # below is explicitly read-only to preserve the frozen lexicon.
    sys.path.insert(0, str(ROOT))
    from system.entity_audio_repair import GenericEntitySelector
    from system.entity_store import EntityDefinition

    uri = f"file:{LEXICON_PATH.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """SELECT e.id, e.canonical_text, e.entity_type, e.domain,
                      e.normalization_policy, e.priority, e.enabled, e.source,
                      e.created_at, e.updated_at, a.alias
               FROM protected_entities e
               LEFT JOIN entity_aliases a ON a.entity_id=e.id
               ORDER BY e.priority DESC, e.id ASC, a.id ASC"""
        ).fetchall()
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        entity_id = int(row[0])
        item = grouped.setdefault(entity_id, {
            "canonical_text": row[1], "entity_type": row[2], "domain": row[3],
            "normalization_policy": row[4], "priority": int(row[5]),
            "enabled": bool(row[6]), "source": row[7], "created_at": row[8],
            "updated_at": row[9], "aliases": [],
        })
        if row[10] is not None:
            item["aliases"].append(str(row[10]))
    definitions = tuple(EntityDefinition(
        entity_id=entity_id,
        canonical_text=str(item["canonical_text"]),
        entity_type=str(item["entity_type"]),
        domain=str(item["domain"]),
        normalization_policy=str(item["normalization_policy"]),
        priority=int(item["priority"]), aliases=tuple(item["aliases"]),
        enabled=bool(item["enabled"]), source=str(item["source"]),
        created_at=str(item["created_at"]), updated_at=str(item["updated_at"]),
    ) for entity_id, item in grouped.items())
    return definitions, GenericEntitySelector(definitions), len(grouped)


def read_inputs():
    samples = {row["id"]: row for row in jsonl(SAMPLES_PATH)}
    audios = {row["id"]: row for row in jsonl(AUDIO_PATH)}
    if len(samples) != len(audios):
        raise ValueError(f"Input row count mismatch: samples={len(samples)} audio={len(audios)}")
    if samples.keys() != audios.keys():
        raise ValueError("Frozen sample/audio IDs do not match exactly")
    return samples, audios


def initial_text(audio: dict[str, Any]) -> str:
    return str((audio.get("initial") or {}).get("text") or "")


def build_actual_conflicts(samples, audios):
    conflicts = []
    for sample_id, audio in audios.items():
        repair = ((audio.get("production_final") or {}).get("entity_audio_repair") or {})
        if not repair.get("applied"):
            continue
        sample = samples[sample_id]
        reference = str(sample.get("reference") or "")
        for change in repair.get("changes") or []:
            canonical = str(change.get("canonical") or "")
            if not canonical or canonical in reference:
                continue
            conflicts.append(make_row(
                sample, audio, change, priority="actual_accepted_conflict",
                accepted=True,
            ))
    conflicts.sort(key=lambda row: (row["recording_id"], row["sample_id"], row["candidate"]["start"]))
    recordings = {row["recording_id"] for row in conflicts}
    if len(conflicts) != 4 or recordings != EXPECTED_ABSENT_RECORDINGS:
        raise ValueError(f"Expected the four inventoried accepted conflicts; got {len(conflicts)} / {sorted(recordings)}")
    return conflicts


def make_row(sample, audio, candidate, *, priority: str, accepted: bool):
    canonical = str(candidate["canonical"])
    reference = str(sample.get("reference") or "")
    sample_id = str(sample["id"])
    recording_id = str(sample.get("recording_id") or sample_id)
    start, end = int(candidate["start"]), int(candidate["end"])
    return {
        "sample_id": sample_id,
        "recording_id": recording_id,
        "audio_path": sample.get("audio_path"),
        "pcm_path": sample.get("pcm_path"),
        "audio_sha256": sample.get("audio_sha256"),
        "pcm_sha256": sample.get("pcm_sha256"),
        "initial": initial_text(audio),
        "reference": reference,
        "candidate": {
            "start": start,
            "end": end,
            "observed": str(candidate.get("observed") or ""),
            "canonical": canonical,
            "entity_type": str(candidate.get("entity_type") or "UNKNOWN"),
            "retrieval_score": float(candidate.get("retrieval_score") or 0.0),
            "phonetic_score": candidate.get("phonetic_score"),
            "character_score": candidate.get("character_score"),
            "length_score": candidate.get("length_score"),
        },
        "priority": priority,
        "actually_accepted": accepted,
        "weak_label": "reference_absent_candidate",
        "human_verdict": None,
        "selection_hash": stable_hash(sample_id, recording_id, start, end, canonical),
    }


def collect_new_candidates(samples, audios, selector, definitions):
    excluded = EXPECTED_ABSENT_RECORDINGS
    definitions_by_id = {definition.entity_id: definition for definition in definitions}
    best_by_recording = {}
    scanned = 0
    sample_order = sorted(samples, key=lambda sample_id: (
        stable_hash(sample_id, samples[sample_id].get("recording_id") or sample_id), sample_id
    ))
    for sample_id in sample_order:
        sample, audio = samples[sample_id], audios[sample_id]
        recording_id = str(sample.get("recording_id") or sample_id)
        text = initial_text(audio)
        if not text.strip() or recording_id in excluded:
            continue
        reference = str(sample.get("reference") or "")
        recorded_terms = tuple(dict.fromkeys(
            str(term) for term in (((audio.get("production_final") or {}).get("entity_audio_repair") or {}).get("prompt_terms") or [])
            if term and str(term) not in reference
        ))
        # The saved candidate-term list bounds what is reviewed; the current
        # selector still supplies the actual span and retrieval score.
        if not recorded_terms:
            continue
        scanned += 1
        candidates, _terms = selector.select(text)
        for candidate in candidates:
            canonical = str(candidate.get("canonical") or "")
            if not canonical or canonical not in recorded_terms or canonical in reference:
                continue
            owner = definitions_by_id.get(int(candidate["entity_id"]))
            enriched = dict(candidate)
            enriched["entity_type"] = owner.entity_type if owner else "UNKNOWN"
            if owner is None or owner.canonical_text != canonical:
                raise ValueError(f"Selector entity_id/canonical mismatch: {candidate}")
            key = (recording_id, int(candidate["start"]), int(candidate["end"]), canonical)
            row = make_row(sample, audio, enriched, priority="hypothesis", accepted=False)
            row["candidate_key"] = list(key)
            previous = best_by_recording.get(recording_id)
            ranking = (-row["candidate"]["retrieval_score"], stable_hash(*key), sample_id)
            if previous is None or ranking < previous[0]:
                best_by_recording[recording_id] = (ranking, row)
        if scanned % 100 == 0:
            print(f"selector scanned {scanned} recordings; eligible={len(best_by_recording)}", flush=True)
        if len(best_by_recording) >= HASH_SCAN_TARGET:
            break
    candidates = [item[1] for item in best_by_recording.values()]

    # Stable per-type hash ordering, then round-robin types to get early
    # coverage without replacing candidates with fabricated examples.
    by_type = defaultdict(list)
    for row in candidates:
        kind = row["candidate"]["entity_type"]
        row["selection_hash"] = stable_hash(row["sample_id"], row["candidate"]["start"],
                                            row["candidate"]["end"], row["candidate"]["canonical"])
        by_type[kind].append(row)
    for kind in by_type:
        by_type[kind].sort(key=lambda row: (row["selection_hash"], row["sample_id"]))
    ordered = []
    types = sorted(by_type)
    while any(by_type.values()):
        for kind in types:
            if by_type[kind]:
                ordered.append(by_type[kind].pop(0))
    selected = ordered[:NEW_CANDIDATE_LIMIT]
    for row in selected:
        row.pop("candidate_key", None)
        row["selection_method"] = "one_best_scoring_reference_absent_candidate_per_recording; entity_type_round_robin; fixed_sha256_order"
    return selected, scanned, len(candidates)


def main():
    samples, audios = read_inputs()
    definitions, selector, dictionary_count = load_definitions()
    if dictionary_count != 1346:
        raise ValueError(f"Expected frozen 1,346-entry dictionary; found {dictionary_count}")
    actual = build_actual_conflicts(samples, audios)
    new, scanned, unique_candidate_recordings = collect_new_candidates(samples, audios, selector, definitions)
    type_by_canonical = defaultdict(set)
    for definition in definitions:
        type_by_canonical[definition.canonical_text].add(definition.entity_type)
    for row in actual + new:
        canonical = row["candidate"]["canonical"]
        if row["candidate"]["entity_type"] not in type_by_canonical[canonical]:
            raise ValueError(f"Candidate type is not in the frozen dictionary for {canonical!r}: {row['candidate']['entity_type']}")
    for row in actual + new:
        start, end = row["candidate"]["start"], row["candidate"]["end"]
        if row["initial"][start:end] != row["candidate"]["observed"]:
            raise ValueError(f"Candidate span does not match preserved initial text: {row['sample_id']}")
    queue = actual + new

    # Essential integrity checks: uniqueness, exact weak-label semantics,
    # and no overlap between historical conflicts and newly proposed items.
    keys = [(r["recording_id"], r["candidate"]["start"], r["candidate"]["end"], r["candidate"]["canonical"]) for r in queue]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate queue candidate keys")
    if len({r["recording_id"] for r in new}) != len(new):
        raise ValueError("New candidate selection contains more than one item per recording")
    if any(r["recording_id"] in EXPECTED_ABSENT_RECORDINGS for r in new):
        raise ValueError("A known accepted-conflict recording entered the new-candidate pool")
    if any(r["human_verdict"] is not None for r in queue):
        raise ValueError("Human verdict must remain unassigned")
    if any(r["weak_label"] != "reference_absent_candidate" for r in queue):
        raise ValueError("Unexpected weak-label value")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with QUEUE_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        for row in queue:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    all_code = [
        ROOT / "system/entity_audio_repair.py",
        ROOT / "system/entity_retrieval.py",
        ROOT / "system/entity_matcher.py",
        ROOT / "system/entity_boundaries.py",
        ROOT / "system/entity_store.py",
    ]
    by_type = defaultdict(lambda: {"count": 0, "actual_accepted": 0, "hypothesis": 0})
    by_priority = CounterLike()
    for row in queue:
        kind = row["candidate"]["entity_type"]
        by_type[kind]["count"] += 1
        by_type[kind]["actual_accepted"] += int(row["actually_accepted"])
        by_type[kind]["hypothesis"] += int(not row["actually_accepted"])
        by_priority[row["priority"]] += 1
    manifest = {
        "schema_version": 1,
        "created_from": "frozen inputs only; no training, inference, or integration",
        "queue_path": str(QUEUE_PATH.relative_to(ROOT)),
        "selection": {
            "actual_accepted_conflicts": "Include the four inventoried applied entity_audio_repair changes whose canonical string is absent from the exact reference string.",
            "hypotheses": "Preorder recordings by fixed SHA-256 hash. Skip blank initial text, the four conflict recordings, and recordings with no saved prompt_terms absent from reference. Run current GenericEntitySelector on each eligible initial text and keep matching recorded prompt_terms whose canonical is absent from reference; retain the highest retrieval_score candidate per recording. Stop after 392 recordings produce candidates, then order each entity type by fixed SHA-256 hash and round-robin types; cap at 196.",
            "fixed_hash_salt": HASH_SALT,
            "hash_scan_target_recordings": HASH_SCAN_TARGET,
            "max_new_hypotheses": NEW_CANDIDATE_LIMIT,
            "hypotheses_are_not_actual_errors": True,
            "weak_label_is_not_training_truth": True,
        },
        "frozen_inputs": {
            "samples_jsonl": {"path": str(SAMPLES_PATH.relative_to(ROOT)), "sha256": sha256(SAMPLES_PATH), "rows": len(samples)},
            "audio_jsonl": {"path": str(AUDIO_PATH.relative_to(ROOT)), "sha256": sha256(AUDIO_PATH), "rows": len(audios)},
            "entity_database": {"path": str(LEXICON_PATH.relative_to(ROOT)), "sha256": sha256(LEXICON_PATH), "definitions": dictionary_count},
            "selector_source": {str(path.relative_to(ROOT)): sha256(path) for path in all_code},
        },
        "selector": {"class": "system.entity_audio_repair.GenericEntitySelector", "definitions_loaded": len(definitions)},
        "screening": {"selector_calls_with_nonblank_initial_and_absent_prompt_terms": scanned, "eligible_candidate_recordings_in_hash_scan_pool": unique_candidate_recordings, "hash_scan_pool_capped": unique_candidate_recordings >= HASH_SCAN_TARGET},
        "queue_counts": {
            "total": len(queue), "actual_accepted_conflicts": len(actual), "new_hypotheses": len(new),
            "priority": dict(sorted(by_priority.items())),
            "actually_accepted": sum(int(row["actually_accepted"]) for row in queue),
            "by_entity_type": dict(sorted(by_type.items())),
        },
        "integrity_checks": {
            "samples_and_audio_ids_exact_match": True,
            "four_expected_actual_conflicts_present": len(actual) == 4,
            "one_new_candidate_per_recording": True,
            "excluded_conflict_recordings_absent_from_hypotheses": True,
            "all_human_verdicts_null": True,
            "all_weak_labels_reference_absent_candidate": True,
            "queue_sha256": sha256(QUEUE_PATH),
        },
        "notes": [
            "Reference absence is only a weak text-match label, not audio truth.",
            "Actual accepted changes and newly retrieved hypotheses are separately marked.",
            "Human listening and adjudication are required before any learning use.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["queue_counts"], ensure_ascii=False, sort_keys=True))


class CounterLike(dict):
    def __missing__(self, key):
        self[key] = 0
        return 0


if __name__ == "__main__":
    main()
