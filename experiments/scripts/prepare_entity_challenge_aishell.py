"""Prepare a reference-only AISHELL entity challenge pool (no ASR inference)."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path("/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell")
ANNOTATIONS = ROOT / "results/entity_library_20261003/source/aishell_ner_transcript.txt"
TRANSCRIPT = CORPUS / "transcript/aishell_transcript_v0.8.txt"
OUT = ROOT / "results/entity_challenge_20261014"
POOL = OUT / "aishell_pool.jsonl"
STATS = OUT / "aishell_pool_stats.json"
NER = re.compile(r"\[([^\[\]]+)\]|\(([^()]+)\)|<([^<>]+)>")
TYPES = ("PERSON", "LOC", "ORG")
GENERIC = {
    "公司", "银行", "法院", "记者", "医生", "患者", "中国人", "国人", "企业", "学生",
    "单位", "部门", "政府", "政府部门", "工作人员", "负责人", "代表", "委员", "领导",
    "市民", "群众", "居民", "村民", "网友", "专家", "老师", "教授", "同学", "朋友",
    "先生", "女士", "先生们", "女士们", "团队", "小组", "协会", "委员会", "研究院",
}
SEED = "entity-challenge-aishell-20261014-v1"
MAX_ROWS = 8000


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] not in {"P", "Z", "C"})


def is_han(text: str) -> bool:
    return bool(text) and all(unicodedata.name(ch, "").startswith(
        ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")) for ch in text)


def read_pairs(path: Path) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed pair at {path}:{line_no}")
        pairs[fields[0]] = fields[1]
    return pairs


def parse_ner(text: str):
    """Return unmarked transcript and native spans using the existing AISHELL convention."""
    parts, entities, previous, offset = [], [], 0, 0
    for match in NER.finditer(text):
        prefix = text[previous:match.start()]
        parts.append(prefix)
        offset += len(prefix)
        idx = next(i for i, value in enumerate(match.groups()) if value is not None)
        surface = match.group(idx + 1)
        entities.append({"text": surface, "type": TYPES[idx],
                         "plain_start": offset, "plain_end": offset + len(surface)})
        parts.append(surface)
        offset += len(surface)
        previous = match.end()
    parts.append(text[previous:])
    return "".join(parts), entities


def reference_offsets(plain: str, reference: str) -> dict[int, int]:
    """Map annotation offsets to official-reference offsets, ignoring whitespace."""
    mapping, p, r = {}, 0, 0
    while p < len(plain) and r < len(reference):
        if plain[p].isspace():
            p += 1
            continue
        if reference[r].isspace():
            r += 1
            continue
        if normalize(plain[p]) != normalize(reference[r]):
            raise ValueError(f"Cannot align characters at plain={p}, reference={r}")
        mapping[p] = r
        p += 1
        r += 1
    while p < len(plain) and plain[p].isspace():
        p += 1
    while r < len(reference) and reference[r].isspace():
        r += 1
    if p != len(plain) or r != len(reference):
        raise ValueError("Unconsumed characters while aligning NER and reference")
    return mapping


def load_exclusions() -> tuple[set[str], set[str], dict[str, dict[str, int]]]:
    """Read source IDs and PCM hashes only; never open the protected reference sidecar."""
    recording_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    main_samples = ROOT / "results/large_entity_benchmark_v1_20261010/dataset/samples.jsonl"
    if main_samples.exists():
        for line in main_samples.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for source_row in row.get("source_records", []):
                if "AISHELL-1" in source_row.get("source", ""):
                    rid = source_row.get("recording_id") or source_row.get("source_id")
                    if rid:
                        recording_ids.add(rid)
        source_counts["large_entity_benchmark_source_recording_ids"] = len(recording_ids)

    # The 2,506 inference set includes the large benchmark pool. Read its source IDs
    # from the frozen sample metadata, but do not inspect its scoring references.
    inference = ROOT / "results/entity_modes_dataset_20261013/inference_inputs.jsonl"
    if inference.exists():
        ids = {json.loads(line)["id"] for line in inference.read_text(encoding="utf-8").splitlines() if line}
        for line in main_samples.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("id") in ids:
                for source_row in row.get("source_records", []):
                    if "AISHELL-1" in source_row.get("source", ""):
                        rid = source_row.get("recording_id") or source_row.get("source_id")
                        if rid:
                            recording_ids.add(rid)
        source_counts["entity_modes_2506_aishell_recording_ids"] = len(recording_ids)

    # This protected set is represented here by metadata only. Using source_recording_id
    # is exact for AISHELL's WAV filename/recording_id and avoids opening any gold refs.
    protected_samples = ROOT / "results/entity_context_confirmation_20261012/unseen/samples.jsonl"
    protected_pcm_hashes: set[str] = set()
    if protected_samples.exists():
        protected_ids = set()
        for line in protected_samples.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("source") == "AISHELL-1 test":
                rid = row.get("source_id")
                if rid:
                    protected_ids.add(rid)
                if row.get("pcm_sha256"):
                    protected_pcm_hashes.add(row["pcm_sha256"])
        recording_ids.update(protected_ids)
        source_counts["protected_new181_aishell_source_ids"] = len(protected_ids)
        source_counts["protected_new181_aishell_pcm_hashes"] = len(protected_pcm_hashes)
    return recording_ids, protected_pcm_hashes, dict(source_counts)


def speaker_id(recording_id: str) -> str:
    match = re.search(r"S\d+", recording_id)
    return match.group(0) if match else ""


def main() -> None:
    annotations = read_pairs(ANNOTATIONS)
    references = read_pairs(TRANSCRIPT)
    annotation_sha = hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest()
    transcript_sha = hashlib.sha256(TRANSCRIPT.read_bytes()).hexdigest()
    split_manifest_hashes = {
        split: hashlib.sha256((CORPUS / f"manifests/{split}_zh-CN.json").read_bytes()).hexdigest()
        for split in ("train", "dev", "test")
    }

    # Frequency is computed over the complete NER annotation file before audio,
    # split, exclusion, reference deduplication, or candidate filtering.
    person_freq: Counter[str] = Counter()
    parsed: dict[str, tuple[str, list[dict]]] = {}
    for rid, annotated in annotations.items():
        plain, entities = parse_ner(annotated)
        parsed[rid] = (plain, entities)
        for ent in entities:
            name = normalize(ent["text"])
            if ent["type"] == "PERSON" and len(ent["text"]) >= 2 and is_han(ent["text"]):
                person_freq[name] += 1

    excluded_ids, protected_pcm_hashes, exclusion_counts = load_exclusions()
    audio_root = CORPUS / "wav"
    candidate_rows: list[dict] = []
    seen_reference: set[str] = set()
    duplicate_reference_count = 0
    aligned_count = 0
    mismatch_ids: list[str] = []
    audio_missing = 0
    all_recordings: Counter[str] = Counter()
    eligible_by_split: Counter[str] = Counter()
    eligible_before_dedup_by_split: Counter[str] = Counter()
    entity_mentions: Counter[str] = Counter()
    excluded_by_split: Counter[str] = Counter()
    for split in ("train", "dev", "test"):
        for audio_path in sorted((audio_root / split).rglob("*.wav")):
            all_recordings[split] += 1
            rid = audio_path.stem
            if rid not in annotations:
                continue
            if rid not in references or rid not in parsed:
                mismatch_ids.append(rid)
                continue
            plain, raw_entities = parsed[rid]
            official = references[rid]
            if normalize(plain) != normalize(official):
                mismatch_ids.append(rid)
                continue
            char_map = reference_offsets(plain, official)
            aligned_count += 1

            valid_entities = []
            for ent in raw_entities:
                name, kind = ent["text"], ent["type"]
                if not (len(name) >= 2 and is_han(name)) or name in GENERIC:
                    continue
                start = char_map.get(ent["plain_start"])
                last = char_map.get(ent["plain_end"] - 1)
                if start is None or last is None:
                    continue
                end = last + 1
                surface = official[start:end]
                if "".join(surface.split()) != name:
                    raise ValueError(f"Entity boundary mismatch {rid}: {name!r} != {surface!r}")
                valid_entities.append({"text": name, "type": kind, "start": start, "end": end})
            if not valid_entities:
                continue
            if rid in excluded_ids:
                excluded_by_split[split] += 1
                continue

            eligible_before_dedup_by_split[split] += 1
            ref_key = normalize(official)
            if ref_key in seen_reference:
                duplicate_reference_count += 1
                continue
            seen_reference.add(ref_key)
            eligible_by_split[split] += 1
            entity_mentions.update(ent["type"] for ent in valid_entities)

            frequencies = [person_freq.get(normalize(ent["text"]), 0)
                           for ent in valid_entities if ent["type"] == "PERSON"]
            rarest_person_frequency = min(frequencies) if frequencies else None
            # Log inverse frequency keeps scores numerically stable; slight length
            # bonus prioritizes a rare three-character name over a two-character one.
            rarity = (math.log1p(1_000_000 / rarest_person_frequency) if rarest_person_frequency
                      else 0.0)
            if any(ent["type"] == "PERSON" and len(ent["text"]) == 3 for ent in valid_entities):
                rarity += 0.25
            if any(ent["type"] == "PERSON" for ent in valid_entities):
                rarity += 100.0
            tie_hash = hashlib.sha256(f"{SEED}\0{rid}".encode("utf-8")).hexdigest()
            candidate_rows.append({
                "reference": official,
                "gold_entities": valid_entities,
                "audio_path": str(audio_path),
                "source_recording_id": rid,
                "speaker_id": speaker_id(rid),
                "rarity_score": round(rarity, 8),
                "rarest_person_annotation_frequency": rarest_person_frequency,
                "domain": "人名机构地名",
                "split": split,
                "rarity_tie_hash": tie_hash,
            })

    if mismatch_ids:
        raise ValueError(f"{len(mismatch_ids)} annotation/reference mismatches; examples={mismatch_ids[:10]}")

    candidate_rows.sort(key=lambda row: (-row["rarity_score"], row["rarity_tie_hash"]))
    available_count = len(candidate_rows)
    selected = candidate_rows[:MAX_ROWS]
    OUT.mkdir(parents=True, exist_ok=True)
    with POOL.open("w", encoding="utf-8") as f:
        for row in selected:
            row = dict(row)
            row.pop("rarity_tie_hash")
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    selected_types: Counter[str] = Counter()
    selected_splits: Counter[str] = Counter()
    selected_person_mention_rows = 0
    selected_unique_refs = set()
    for row in selected:
        selected_splits[row["split"]] += 1
        selected_unique_refs.add(normalize(row["reference"]))
        selected_types.update(ent["type"] for ent in row["gold_entities"])
        selected_person_mention_rows += any(ent["type"] == "PERSON" for ent in row["gold_entities"])

    stats = {
        "dataset": "entity_challenge_aishell_20261014",
        "selection_seed": SEED,
        "selection": "Reference-only; sort by rare PERSON frequency score, then stable SHA-256 tie hash; cap at 8000.",
        "official_source": {
            "corpus": "AISHELL-1",
            "audio_home": str(audio_root),
            "official_annotation_source": str(ANNOTATIONS),
            "official_annotation_rows": len(annotations),
            "official_annotation_sha256": annotation_sha,
            "official_transcript": str(TRANSCRIPT),
            "official_transcript_rows": len(references),
            "official_transcript_sha256": transcript_sha,
            "official_split_manifests_sha256": split_manifest_hashes,
            "provenance_note": "AISHELL-1 corpus and AISHELL-NER annotation/transcript source used by the existing AISHELL entity benchmark preparation.",
        },
        "alignment": {
            "aligned_audio_recordings_with_annotation": aligned_count,
            "annotation_reference_mismatches": len(mismatch_ids),
            "all_audio_recordings_by_split": dict(all_recordings),
            "audio_missing_or_unannotated_count": sum(all_recordings.values()) - aligned_count,
        },
        "exclusions": {
            "source_recording_ids_excluded": len(excluded_ids),
            "counts_by_origin": exclusion_counts,
            "excluded_records_by_split": dict(excluded_by_split),
            "protected_new181_reference_accessed": False,
            "protected_pcm_hashes_read_from_metadata": len(protected_pcm_hashes),
            "protected_aishell_ids_excluded_by_exact_recording_id": True,
        },
        "candidate_counts": {
            "available_after_valid_entity_filter_and_exclusions_and_reference_dedup": available_count,
            "selected": len(selected),
            "cap": MAX_ROWS,
            "duplicate_normalized_references_removed": duplicate_reference_count,
            "unique_normalized_references_selected": len(selected_unique_refs),
            "selected_rows_with_person": selected_person_mention_rows,
            "available_unique_references_by_split": dict(eligible_by_split),
            "eligible_rows_before_reference_dedup_by_split": dict(eligible_before_dedup_by_split),
        },
        "selected_entity_mentions_by_type": dict(selected_types),
        "selected_recordings_by_split": dict(selected_splits),
        "entity_policy": {
            "types": list(TYPES),
            "retained_native_annotations": "Every valid native PERSON/LOC/ORG span on a selected row is retained; filtering only removes single-character, non-Han, and explicit generic spans.",
            "generic_terms_exact_match": sorted(GENERIC),
            "span_convention": "Zero-based Unicode code-point offsets [start,end) into official reference; whitespace-free reference slice equals the annotated entity spelling.",
            "deduplication": "NFKC/casefold normalization with Unicode punctuation, whitespace, and control characters removed; keep one recording per normalized complete reference.",
            "rarity": "PERSON name frequency counted over all 141600 official annotation rows, before split/exclusion; inverse-frequency logarithmic score, +100 PERSON preference and +0.25 for any 3-character PERSON.",
            "audio_processing": "No audio decoding, conversion, or ASR performed; audio_path points to source WAV.",
        },
        "output": {
            "pool_jsonl": str(POOL),
            "pool_sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(),
            "stats_json": str(STATS),
        },
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
