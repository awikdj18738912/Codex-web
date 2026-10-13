"""Build the AISHELL-1 staging pool for the large entity benchmark."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path("/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell")
ANNOTATIONS = ROOT / "results/entity_library_20261003/source/aishell_ner_transcript.txt"
OUT = ROOT / "results/large_entity_benchmark_v1_20261010/staging/aishell"
NER = re.compile(r"\[([^\[\]]+)\]|\(([^()]+)\)|<([^<>]+)>")
TYPES = ("PERSON", "LOC", "ORG")
GENERIC = {"公司", "银行", "法院", "记者", "医生", "患者", "中国人", "国人", "企业", "学生"}
HISTORICAL_ASR = (
    ROOT / "results/entity_library_20261003/asr.jsonl",
    ROOT / "results/entity_context_20261006/fresh_asr.jsonl",
    ROOT / "results/entity_context_20261006/name_holdout_asr.jsonl",
)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] not in {"P", "Z", "C"})


def is_han(text: str) -> bool:
    return bool(text) and all(unicodedata.name(ch, "").startswith(
        ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")) for ch in text)


def read_pairs(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        result[key] = value
    return result


def parse_ner(text: str):
    parts, entities, previous, offset = [], [], 0, 0
    for match in NER.finditer(text):
        prefix = text[previous:match.start()]
        parts.append(prefix)
        offset += len(prefix)
        idx = next(i for i, value in enumerate(match.groups()) if value is not None)
        surface = match.group(idx + 1)
        entities.append({"text": surface, "type": TYPES[idx], "plain_start": offset,
                         "plain_end": offset + len(surface)})
        parts.append(surface)
        offset += len(surface)
        previous = match.end()
    parts.append(text[previous:])
    return "".join(parts), entities


def reference_offsets(plain: str, reference: str) -> dict[int, int]:
    """Map transcript offsets while ignoring whitespace token separators."""
    mapping, p, r = {}, 0, 0
    while p < len(plain) and r < len(reference):
        if plain[p].isspace():
            p += 1
            continue
        if reference[r].isspace():
            r += 1
            continue
        if norm(plain[p]) != norm(reference[r]):
            raise ValueError(f"Cannot align characters at plain={p}, ref={r}")
        mapping[p] = r
        p += 1
        r += 1
    while p < len(plain) and plain[p].isspace():
        p += 1
    while r < len(reference) and reference[r].isspace():
        r += 1
    if p != len(plain) or r != len(reference):
        raise ValueError("Unconsumed characters while aligning NER and official transcript")
    return mapping


def historical_ids() -> set[str]:
    found = set()
    for path in HISTORICAL_ASR:
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                found.add(json.loads(line)["id"])
    if len(found) != 328:
        raise ValueError(f"Expected 328 historical evaluated recordings, found {len(found)}")
    return found


def main() -> None:
    annotations = read_pairs(ANNOTATIONS)
    transcript = read_pairs(CORPUS / "transcript/aishell_transcript_v0.8.txt")
    excluded = historical_ids()
    candidates, mismatches = [], []
    type_by_name, evidence_by_name = defaultdict(set), defaultdict(list)
    type_counts, sample_counts = Counter(), Counter()
    excluded_present, excluded_hashes = 0, set()

    for audio_path in sorted((CORPUS / "wav/test").rglob("*.wav")):
        rid = audio_path.stem
        if rid not in annotations:
            continue
        plain, raw_entities = parse_ner(annotations[rid])
        official = transcript.get(rid)
        if official is None or norm(plain) != norm(official):
            mismatches.append(rid)
            continue
        char_map = reference_offsets(plain, official)
        audio_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        with wave.open(str(audio_path), "rb") as wav:
            rate, channels, width, frames = (wav.getframerate(), wav.getnchannels(),
                                             wav.getsampwidth(), wav.getnframes())
            if rate != 16000 or channels != 1 or width != 2:
                raise ValueError(f"Expected mono 16 kHz PCM16, got rate={rate}, channels={channels}, width={width}: {audio_path}")
            pcm_bytes = wav.readframes(frames)
        pcm16_hash = hashlib.sha256(pcm_bytes).hexdigest()
        pcm_float32 = np.frombuffer(pcm_bytes, dtype="<i2").astype("<f4") / 32768.0
        pcm_hash = hashlib.sha256(pcm_float32.astype("<f4", copy=False).tobytes()).hexdigest()
        if rid in excluded:
            excluded_present += 1
            excluded_hashes.add(audio_hash)
            continue

        entities = []
        for ent in raw_entities:
            name, kind = ent["text"], ent["type"]
            if not (2 <= len(name) <= 24 and is_han(name)) or name in GENERIC:
                continue
            start = char_map[ent["plain_start"]]
            end = char_map[ent["plain_end"] - 1] + 1
            if "".join(official[start:end].split()) != name:
                raise ValueError(f"Entity not complete in reference span: {rid} {name!r} {official[start:end]!r}")
            entities.append({"text": name, "type": kind, "start": start, "end": end})
            type_by_name[name].add(kind)
            type_counts[kind] += 1
            evidence_by_name[name].append({"sample_id": rid, "type": kind, "start": start,
                                           "end": end, "reference_span": official[start:end]})
        candidates.append({"id": rid, "source": "AISHELL-1 test / AISHELL-NER",
                           "recording_id": rid, "reference": official,
                           "audio_path": str(audio_path), "duration_seconds": frames / rate,
                           "audio_sha256": audio_hash, "source_pcm16_sha256": pcm16_hash,
                           "pcm_sha256": pcm_hash,
                           "entities": entities})
        sample_counts["with_entities" if entities else "without_entities"] += 1

    if mismatches:
        raise ValueError(f"{len(mismatches)} NER/reference mismatches; examples={mismatches[:10]}")
    if excluded_present != len(excluded):
        raise ValueError(f"Only {excluded_present}/{len(excluded)} historical IDs found in aligned test audio")
    if excluded_hashes.intersection(row["audio_sha256"] for row in candidates):
        raise ValueError("Source audio hash overlaps a historically evaluated recording")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    entity_rows = []
    for name in sorted(evidence_by_name):
        kinds, evidence = sorted(type_by_name[name]), evidence_by_name[name]
        entity_rows.append({"canonical": name,
                            "entity_type": kinds[0] if len(kinds) == 1 else "CONFLICT",
                            "observed_entity_types": kinds, "type_conflict": len(kinds) > 1,
                            "supporting_sample_ids": sorted({x["sample_id"] for x in evidence}),
                            "reference_evidence": evidence})
    (OUT / "entities.json").write_text(json.dumps(entity_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    generic_review = ["女排", "国家队", "董事会", "事故调查组", "家用空调事业部",
                      "压缩机事业部", "猜评团", "交通枢纽商圈"]
    boundary_review = ["蚌飞市", "多塞特群", "首竟贸站", "中国证卷报", "上海迪尼士",
                       "香港迪尼士", "掠队", "华江瑶族乡十四届人大代表会"]
    by_name = {row["canonical"]: row for row in entity_rows}
    quality_review = {
        "purpose": "Human screening aid only; no item is automatically removed, corrected, aliased, or merged.",
        "generic_or_noncanonical_candidates": [by_name[n] for n in generic_review if n in by_name],
        "possible_spelling_or_boundary_candidates": [by_name[n] for n in boundary_review if n in by_name],
        "cross_type_conflicts": [row for row in entity_rows if row["type_conflict"]],
        "notes": [
            "Candidate flags are based on a lightweight lexical review of official AISHELL-NER strings, not external alias lookup.",
            "Review per-sample reference_evidence before excluding or changing any name.",
            "Cross-type labels are retained as observed; country names may refer to a place, state, team, or organization in context.",
        ],
    }
    (OUT / "entity_quality_review.json").write_text(
        json.dumps(quality_review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stats = {
        "source": "AISHELL-1 test audio with official AISHELL-NER annotations",
        "annotation_source": str(ANNOTATIONS),
        "official_transcript": str(CORPUS / "transcript/aishell_transcript_v0.8.txt"),
        "historical_ids_excluded": len(excluded), "historical_ids_found_and_excluded": excluded_present,
        "aligned_test_recordings_available": len(candidates), "aligned_mismatches": len(mismatches),
        "samples_with_eligible_entities": sample_counts["with_entities"],
        "samples_without_eligible_entities": sample_counts["without_entities"],
        "eligible_distinct_canonical_names_global": len(entity_rows),
        "eligible_distinct_names_by_type": {kind: sum(kind in type_by_name[n] for n in type_by_name) for kind in TYPES},
        "names_with_cross_type_conflict": sum(row["type_conflict"] for row in entity_rows),
        "entity_mentions_by_type": dict(type_counts),
        "eligibility": "Official NER spans; 2-24 Han characters; explicit exact-match generic terms removed; no alias generation",
        "audio_format": "mono 16 kHz PCM16 source WAV",
        "audio_sha256_definition": "SHA-256 of source WAV file bytes",
        "source_pcm16_sha256_definition": "SHA-256 of little-endian signed PCM16 frame bytes read from source WAV",
        "pcm_sha256_definition": "SHA-256 of float32 little-endian PCM bytes from np.frombuffer(raw,dtype='<i2').astype('<f4') / 32768.0",
        "span_convention": "Unicode code-point offsets [start,end) in official reference; removing whitespace from reference[start:end] yields the complete entity spelling",
    }
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = f"""# AISHELL-1 staging pool

This is the full aligned, previously unused AISHELL-1 test recording pool, derived from official AISHELL-NER spans and the official AISHELL-1 reference transcript. No aliases were generated.

- {len(candidates):,} aligned unused recordings; {sample_counts['with_entities']:,} contain at least one eligible entity.
- {len(entity_rows):,} globally deduplicated eligible canonical names. A name seen under multiple NER types is retained once and marked `CONFLICT`, with all types and per-sample evidence.
- Eligibility: 2–24 Han characters. Explicit generic words are removed by exact match. Short proper names such as 李娜、上海、天津 remain eligible.
- The 328 historically evaluated recording IDs were excluded. Recording IDs and source audio hashes were checked for overlap.
- `start`/`end` are zero-based Unicode code-point offsets into the official reference; end is exclusive. Removing whitespace from the span yields the complete entity spelling because the reference uses spaces between tokens.
- Audio is mono 16 kHz PCM16. `audio_sha256` hashes source WAV bytes; `source_pcm16_sha256` hashes decoded little-endian signed PCM16 samples. `pcm_sha256` follows the evaluation convention: interpret PCM16 with `np.frombuffer(raw, dtype='<i2')`, cast to `<f4`, divide by 32768.0, and hash the resulting float32 little-endian bytes.
- `entity_quality_review.json` lists lightweight generic/boundary review candidates and all cross-type conflicts. These are review flags only; no aliases, corrections, exclusions, or merges are applied.

AISHELL contributes PERSON/LOC/ORG entities from news-domain speech. Its measured distinct-name count does not represent the multi-domain benchmark total.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
