"""Freeze the multi-source large entity benchmark from completed staging pools.

This writes only results/large_entity_benchmark_v1_20261010/dataset/.
The build is atomic and refuses to overwrite an existing frozen dataset.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "results/large_entity_benchmark_v1_20261010"
STAGING = BASE / "staging"
AISHELL_CORPUS = Path("/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell")
AISHELL_NER = STAGING / "aishell/entities.json"
AISHELL_NER_SOURCE = ROOT / "results/entity_library_20261003/source/aishell_ner_transcript.txt"
QUALITY = BASE / "quality_policy.json"
QUALITY_READY = BASE / "quality_policy.ready"
OUTPUT = BASE / "dataset"
SPEECHIO_ROOT = ROOT / "results/speechio_professional_entity_eval_20261010/source"
HISTORICAL_EXCLUSION_FILES = (
    ROOT / "results/entity_library_20261003/asr.jsonl",
    ROOT / "results/entity_context_20261006/fresh_asr.jsonl",
    ROOT / "results/entity_context_20261006/name_holdout_asr.jsonl",
    ROOT / "results/generic_entity_blind_20261010/metadata.json",
    ROOT / "results/generic_entity_optimization_20261010/holdout/metadata.json",
)
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/usr/bin/ffprobe"
NER_PATTERN = re.compile(r"\[([^\[\]]+)\]|\(([^()]+)\)|<([^<>]+)>")
TYPES = ("PERSON", "LOC", "ORG")
TOPIC_AISHELL = "新闻人物组织地理"
TOPIC_MEDICAL = "医疗健康"
SELECTION_SEED = "large-entity-benchmark-v1-20261010-negative-sha256"


def json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_commonvoice(source: str) -> bool:
    return source.lower().replace("_", "").startswith("commonvoice")


def canonicalizer():
    """Load OpenCC from the available agentic-asr environment if needed."""
    try:
        from opencc import OpenCC
    except ImportError:
        site = Path("/home/aim0/data/conda/envs/agentic-asr/lib/python3.12/site-packages")
        if site.is_dir():
            sys.path.insert(0, str(site))
        try:
            from opencc import OpenCC
        finally:
            if site.is_dir() and str(site) in sys.path:
                sys.path.remove(str(site))
    converter = OpenCC("t2s")

    def canonical_key(text: str) -> str:
        text = unicodedata.normalize("NFKC", converter.convert(text)).casefold()
        return "".join(ch for ch in text if unicodedata.category(ch)[0] not in {"P", "Z", "C"})

    return canonical_key


def read_aishell_raw_annotations(path: Path):
    annotations = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rid, text = line.split(maxsplit=1)
        pieces, entities, prev, offset = [], [], 0, 0
        for m in NER_PATTERN.finditer(text):
            prefix = text[prev:m.start()]
            pieces.append(prefix)
            offset += len(prefix)
            idx = next(i for i, value in enumerate(m.groups()) if value is not None)
            surface = m.group(idx + 1)
            entities.append({"text": surface, "type": TYPES[idx], "start": offset, "end": offset + len(surface)})
            pieces.append(surface)
            offset += len(surface)
            prev = m.end()
        pieces.append(text[prev:])
        annotations[rid] = {"plain": "".join(pieces), "entities": entities}
    return annotations


def compact_with_offset_map(text: str):
    chars, before = [], [0] * (len(text) + 1)
    count = 0
    for i, char in enumerate(text):
        before[i] = count
        if not char.isspace():
            chars.append(char)
            count += 1
    before[len(text)] = count
    return "".join(chars), before


def read_historical_exclusions():
    old_aishell = set()
    for path in (ROOT / "results/entity_library_20261003/asr.jsonl",
                 ROOT / "results/entity_context_20261006/fresh_asr.jsonl",
                 ROOT / "results/entity_context_20261006/name_holdout_asr.jsonl"):
        old_aishell.update(row["id"] for row in jsonl_load(path))
    if len(old_aishell) != 328:
        raise ValueError(f"Historical AISHELL exclusion set changed: expected 328, found {len(old_aishell)}")

    old_cv_ids, old_cv_refs = set(), set()
    for path in (ROOT / "results/generic_entity_blind_20261010/metadata.json",
                 ROOT / "results/generic_entity_optimization_20261010/holdout/metadata.json"):
        if path.exists():
            for row in json_load(path)["samples"]:
                if is_commonvoice(row.get("source", "")):
                    old_cv_ids.add(row.get("location", {}).get("sentence_id", ""))
                    old_cv_refs.add(row.get("reference", ""))
    old_cv_ids.discard("")
    old_cv_refs.discard("")
    if len(old_cv_ids) != 112:
        raise ValueError(f"Historical Common Voice exclusion set changed: expected 112, found {len(old_cv_ids)}")

    old_speechio_ids, old_speechio_source_ids, old_speechio_refs = set(), set(), set()
    for path in (ROOT / "results/speechio_professional_entity_eval_20261010").glob("*/protocol.json"):
        for row in json_load(path)["samples"]:
            if row.get("source_id"):
                old_speechio_ids.add(row["source_id"])
                old_speechio_source_ids.add(row["source_id"])
            if row.get("id"):
                old_speechio_ids.add(row["id"])
            if row.get("reference"):
                old_speechio_refs.add(row["reference"])
    if len(old_speechio_source_ids) != 205:
        raise ValueError(f"Historical SpeechIO source ID set changed: expected 205, found {len(old_speechio_source_ids)}")
    return old_aishell, old_cv_ids, old_cv_refs, old_speechio_ids, old_speechio_source_ids, old_speechio_refs


def sample_source_info(row, tag: str):
    source = row["source"]
    loc = row.get("location") or {}
    if tag == "aishell":
        path = Path(row["audio_path"]).resolve()
        return {"group_key": "path:" + str(path), "source_path": str(path),
                "source_id": row["recording_id"], "recording_id": row["recording_id"],
                "sample_id": row["id"], "source": source, "tag": tag}
    if tag == "medical":
        path = Path(row["audio_path"]).resolve()
        source_id = str(row.get("recording_id", row.get("sample_id", "")))
        normalized_source = "commonvoice_zh-CN_test" if is_commonvoice(source) else source
        return {"group_key": "path:" + str(path), "source_path": str(path),
                "source_id": source_id, "recording_id": source_id,
                "sample_id": row.get("sample_id", source_id), "source": normalized_source, "tag": tag}
    if source == "SpeechIO":
        sid = str(loc.get("segment_id") or row["sample_id"])
        return {"group_key": "speechio:" + sid, "source_path": None, "source_id": sid,
                "recording_id": sid, "sample_id": row["sample_id"], "source": source,
                "tag": tag, "parquet": str(Path(loc["parquet"]).resolve()),
                "row_index": int(loc["row_index"]), "segment_id": sid}
    path = Path(loc.get("audio_path") or row["audio_path"]).resolve()
    sid = str(loc.get("sentence_id") or row.get("sample_id") or row.get("recording_id", ""))
    normalized_source = "commonvoice_zh-CN_test" if is_commonvoice(source) else source
    return {"group_key": "path:" + str(path), "source_path": str(path), "source_id": sid,
            "recording_id": sid, "sample_id": row.get("sample_id", sid), "source": normalized_source,
            "tag": tag}


def read_speechio_payloads(source_groups):
    import pyarrow.parquet as pq

    by_parquet = defaultdict(list)
    for group in source_groups:
        for info in group["source_records"]:
            if info["source"] == "SpeechIO":
                by_parquet[info["parquet"]].append(info)
    payloads = {}
    for parquet_path, records in by_parquet.items():
        pf = pq.ParquetFile(parquet_path)
        offsets, total = [], 0
        for rg in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            offsets.append((total, total + n, rg))
            total += n
        cache = {}
        for info in records:
            index = info["row_index"]
            lo, hi, rg = next((x for x in offsets if x[0] <= index < x[1]), (None, None, None))
            if rg is None:
                raise IndexError(f"SpeechIO row index out of range: {parquet_path} {index}")
            if rg not in cache:
                cache[rg] = pf.read_row_group(rg, columns=["segment_id", "text", "audio"])
            batch = cache[rg]
            local = index - lo
            row = batch.slice(local, 1).to_pylist()[0]
            if row["segment_id"] != info["segment_id"]:
                raise ValueError(f"SpeechIO source_id mismatch at {parquet_path}:{index}")
            if not row.get("audio") or not row["audio"].get("bytes"):
                raise ValueError(f"SpeechIO embedded audio payload missing at {parquet_path}:{index}")
            payloads[info["group_key"]] = row["audio"]["bytes"]
    return payloads


def ffprobe_duration(audio_path: Path) -> float:
    proc = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                          text=True, capture_output=True, check=True)
    value = proc.stdout.strip()
    if not value or value == "N/A":
        raise ValueError(f"ffprobe returned no duration for {audio_path}")
    duration = float(value)
    if duration <= 0:
        raise ValueError(f"Invalid ffprobe duration {duration} for {audio_path}")
    return duration


def convert_audio(job):
    sample_id, audio_path, pcm_path = job
    duration = ffprobe_duration(audio_path)
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio_path),
           "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le",
           "-f", "f32le", str(pcm_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    if pcm_path.stat().st_size == 0 or pcm_path.stat().st_size % 4:
        raise ValueError(f"Invalid float32 PCM byte count for {sample_id}: {pcm_path.stat().st_size}")
    return sample_id, sha256_file(pcm_path), pcm_path.stat().st_size, duration


def add_gold(group, name, kind, span, method, layer, source, topic):
    start, end = span
    reference = group["reference"]
    if start < 0 or end <= start or end > len(reference):
        raise ValueError(f"Bad gold span for {name!r}: {start}:{end} in {reference!r}")
    surface = reference[start:end]
    key = group["canonical_key_fn"](name)
    if group["canonical_key_fn"](surface) != key:
        raise ValueError(f"Gold surface mismatch: {name!r} != {surface!r} at {start}:{end}")
    group["entities"].append({"name": name, "key": key, "source_type": kind,
                              "start": start, "end": end, "surface": surface,
                              "annotation_method": method, "annotation_layer": layer,
                              "source": source, "topic": topic})


def merge_source_record(groups, info, reference, reference_original=None, canonical_key_fn=None):
    key = info["group_key"]
    group = groups.get(key)
    if group is None:
        group = {"group_key": key, "reference": reference,
                 "reference_original": reference_original, "source_records": [],
                 "entities": [], "topics": set(), "cohorts": set(), "canonical_key_fn": canonical_key_fn}
        groups[key] = group
    elif group["reference"] != reference:
        raise ValueError(f"Same source recording has inconsistent reference strings: {key}")
    if reference_original is not None:
        if group["reference_original"] not in (None, reference_original):
            raise ValueError(f"Conflicting AISHELL original references for {key}")
        group["reference_original"] = reference_original
    marker = (info.get("source"), info.get("source_id"), info.get("source_path"),
              info.get("parquet"), info.get("row_index"), info.get("sample_id"))
    if not any((record.get("source"), record.get("source_id"), record.get("source_path"),
                record.get("parquet"), record.get("row_index"), record.get("sample_id")) == marker
               for record in group["source_records"]):
        group["source_records"].append(info)
    return group


def staging_entity_entries(key_fn, aishell_rows, medical_rows, professional_rows, policy, qtype_conflicts):
    raw_entries = []
    for row in aishell_rows:
        name = row["canonical"]
        for observed_type in row.get("observed_entity_types", [row["entity_type"]]):
            raw_entries.append({"canonical": name, "key": key_fn(name), "entity_type": observed_type,
                                "source": "AISHELL-NER", "topic": TOPIC_AISHELL,
                                "method": "official_aishell_ner", "layer": "native_ner"})
    for row in medical_rows:
        name = row["canonical"]
        raw_entries.append({"canonical": name, "key": key_fn(name), "entity_type": row.get("type", "TERM"),
                            "source": row.get("source", "CMeEE_train.json"), "topic": TOPIC_MEDICAL,
                            "method": "cmeee_annotated_reference_exact_match",
                            "layer": "reference_curated"})
    for row in professional_rows:
        name = row["canonical"]
        raw_entries.append({"canonical": name, "key": key_fn(name), "entity_type": row.get("type", "TERM"),
                            "source": row.get("source", "professional_curated_seed"),
                            "topic": row["topic"], "method": row.get("review_method", "reference_curated"),
                            "layer": "reference_curated"})

    reject_ner = {key_fn(k): (k, reason) for k, reason in policy["reject_ner"].items()}
    reject_terms = {key_fn(k): (k, reason) for k, reason in policy["reject_terms"].items()}
    by_key = defaultdict(list)
    for entry in raw_entries:
        by_key[entry["key"]].append(entry)
    conflict_keys = set(qtype_conflicts)
    for key, entries in by_key.items():
        ner_types = {entry["entity_type"] for entry in entries if entry["layer"] == "native_ner"}
        if len(ner_types) > 1:
            conflict_keys.add(key)
    rejected = set(reject_ner) | set(reject_terms) | conflict_keys
    return raw_entries, by_key, rejected, reject_ner, reject_terms, conflict_keys


def get_quarantine(policy, reject_ner, reject_terms, conflict_keys, by_key, raw_annotations,
                   aishell_samples_by_id, canonical_key_fn):
    rows = []
    for category, items in (("quality_policy.reject_ner", reject_ner),
                            ("quality_policy.reject_terms", reject_terms)):
        for key, (original, reason) in sorted(items.items()):
            source_matches = [{"source": entry["source"], "source_name": entry["canonical"],
                               "type": entry["entity_type"]} for entry in by_key.get(key, [])]
            raw_evidence = []
            for rid, annotation in raw_annotations.items():
                if rid not in aishell_samples_by_id:
                    continue
                sample = aishell_samples_by_id[rid]
                for ent in annotation["entities"]:
                    if canonical_key_fn(ent["text"]) == key:
                        raw_evidence.append({"recording_id": rid, "entity_type": ent["type"],
                                             "reference_original": sample["reference"],
                                             "raw_annotation_span": [ent["start"], ent["end"]]})
                        if len(raw_evidence) >= 5:
                            break
                if len(raw_evidence) >= 5:
                    break
            rows.append({"canonical_candidate": original, "canonical_key": key,
                         "reason_code": category, "reason": reason,
                         "source_matches": source_matches, "raw_aishell_evidence_count_at_least": len(raw_evidence),
                         "raw_aishell_evidence": raw_evidence})
    for key in sorted(conflict_keys):
        entries = by_key.get(key, [])
        ner = [entry for entry in entries if entry["layer"] == "native_ner"]
        if len({e["entity_type"] for e in ner}) <= 1:
            continue
        rows.append({"canonical_candidate": sorted(e["canonical"] for e in ner)[0],
                     "canonical_key": key, "reason_code": "ner_cross_type_conflict",
                     "reason": "Same normalized canonical spelling has multiple official NER entity types.",
                     "observed_ner_types": sorted({e["entity_type"] for e in ner}),
                     "source_matches": [{"source": e["source"], "source_name": e["canonical"],
                                         "type": e["entity_type"]} for e in entries]})
    return rows


def load_inputs(key_fn):
    aishell_dir, medical_dir, professional_dir = (STAGING / name for name in ("aishell", "medical", "professional"))
    quality = json_load(QUALITY)
    if quality.get("policy_status") != "FINAL_PRE_BUILD_REVIEW" or not QUALITY_READY.is_file():
        raise ValueError("Final pre-build quality policy and quality_policy.ready marker are required")
    aishell_entities = json_load(aishell_dir / "entities.json")
    medical_entities = json_load(medical_dir / "entities.json")
    professional_entities = json_load(professional_dir / "entities.json")
    aishell_samples = jsonl_load(aishell_dir / "samples.jsonl")
    medical_samples = [row for row in jsonl_load(medical_dir / "samples.jsonl") if row.get("cohort") == "new_main"]
    regression_samples = [row for row in jsonl_load(medical_dir / "samples.jsonl") if row.get("cohort") == "seen_regression"]
    professional_samples = jsonl_load(professional_dir / "samples.jsonl")
    # Unreviewed draft candidates are intentionally never read into the main dataset.
    draft = json_load(professional_dir / "draft_candidates.json")
    if draft.get("status") != "not_reviewed" or draft.get("candidates"):
        raise ValueError("Professional draft candidates must remain outside the main entities")
    return (quality, aishell_entities, medical_entities, professional_entities,
            aishell_samples, medical_samples, regression_samples, professional_samples)


def quarantine_policy_recordings(policy, aishell_samples, medical_samples, professional_samples):
    ids = set(policy.get("quarantine_recording_ids", []))
    found = defaultdict(dict)
    candidate_keys = set()
    for tag, rows in (("aishell", aishell_samples), ("medical", medical_samples),
                      ("professional", professional_samples)):
        for row in rows:
            info = sample_source_info(row, tag)
            identifiers = {str(info.get(k, "")) for k in ("source_id", "recording_id", "sample_id")}
            identifiers.discard("")
            matched = identifiers & ids
            if not matched:
                continue
            rec = found[info["group_key"]]
            rec.update({"source": info["source"], "recording_id": info.get("recording_id"),
                        "sample_id": info.get("sample_id"), "source_id": info.get("source_id"),
                        "source_path": info.get("source_path"), "reference": row.get("reference"),
                        "matched_policy_ids": sorted(matched)})
            source_entities = list(row.get("entities", []))
            if row.get("canonical"):
                source_entities.append({"canonical": row["canonical"]})
            for ent in source_entities:
                name = ent.get("canonical", ent.get("text"))
                if name:
                    candidate_keys.add(name)
                    rec.setdefault("candidate_entities", set()).add(name)
    if ids and len({x for row in found.values() for x in row.get("matched_policy_ids", [])}) != len(ids):
        missing = ids - {x for row in found.values() for x in row.get("matched_policy_ids", [])}
        raise ValueError(f"Policy quarantine recording IDs absent from staging: {sorted(missing)}")
    group_keys = set(found)
    audio_hashes = set()
    for key, row in found.items():
        path = row.get("source_path")
        if path:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            audio_hashes.add(sha256_file(path))
    rows = []
    for key, row in sorted(found.items()):
        rows.append({"canonical_candidate": None, "canonical_key": None,
                     "reason_code": "quarantine_recording_pending_audio_review",
                     "reason": policy.get("quarantine_recording_reason", "Policy-excluded recording pending review."),
                     "source_group_key": key, "source": row.get("source"),
                     "recording_id": row.get("recording_id"), "sample_id": row.get("sample_id"),
                     "source_id": row.get("source_id"), "source_path": row.get("source_path"),
                     "matched_policy_ids": row.get("matched_policy_ids", []),
                     "reference": row.get("reference"),
                     "candidate_entities": sorted(row.get("candidate_entities", set()))})
    return group_keys, audio_hashes, rows, candidate_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen dataset: {OUTPUT}")
    if not Path(FFMPEG).is_file() or not Path(FFPROBE).is_file():
        raise FileNotFoundError("ffmpeg and ffprobe are required")
    key_fn = canonicalizer()
    (quality, aishell_entities, medical_entities, professional_entities,
     aishell_samples, medical_samples, regression_samples, professional_samples) = load_inputs(key_fn)
    old_aishell, old_cv_ids, old_cv_refs, old_speech_ids, old_speech_source_ids, old_speech_refs = read_historical_exclusions()

    aishell_by_id = {row["recording_id"]: row for row in aishell_samples}
    raw_annotations = read_aishell_raw_annotations(AISHELL_NER_SOURCE)
    quarantined_group_keys, quarantined_audio_hashes, quarantine_recording_rows, quarantine_candidate_names = (
        quarantine_policy_recordings(quality, aishell_samples, medical_samples, professional_samples))
    q_conflict_keys = {key_fn(row["canonical"]) for row in aishell_entities if row.get("type_conflict")}
    raw_entries, by_key, rejected_keys, reject_ner, reject_terms, conflict_keys = staging_entity_entries(
        key_fn, aishell_entities, medical_entities, professional_entities, quality, q_conflict_keys)

    # Build a normalized-name registry only from sample-backed staging entries.
    # Entries in draft_candidates.json are excluded before this point.
    groups = {}
    for row in aishell_samples:
        rid = row["recording_id"]
        if rid in old_aishell:
            raise ValueError(f"Previously evaluated AISHELL recording leaked into staging: {rid}")
        original = row["reference"]
        reference, offset_map = compact_with_offset_map(original)
        info = sample_source_info(row, "aishell")
        if info["group_key"] in quarantined_group_keys:
            continue
        group = merge_source_record(groups, info, reference, original, key_fn)
        group["topics"].add(TOPIC_AISHELL)
        group["cohorts"].add("aishell_test")
        raw = raw_annotations.get(rid, {"entities": []})
        for ent in row["entities"]:
            name = ent["text"]
            canon_key = key_fn(name)
            if canon_key in rejected_keys:
                continue
            start, end = offset_map[ent["start"]], offset_map[ent["end"]]
            add_gold(group, name, ent["type"], (start, end), "official_aishell_ner", "native_ner",
                     "AISHELL-NER", TOPIC_AISHELL)
        group["aishell_original_recording_id"] = rid
        group["aishell_source_ner_entities"] = raw["entities"]
        group["aishell_entities_filtered"] = row["entities"]

    for row in medical_samples:
        info = sample_source_info(row, "medical")
        if is_commonvoice(row["source"]):
            sid = info["source_id"]
            if sid in old_cv_ids or row["reference"] in old_cv_refs:
                raise ValueError(f"Previously evaluated Common Voice item leaked into medical staging: {sid}")
        if info["group_key"] in quarantined_group_keys:
            continue
        group = merge_source_record(groups, info, row["reference"], None, key_fn)
        group["topics"].add(TOPIC_MEDICAL)
        group["cohorts"].add("medical_new_main")
        for ent in row["entities"]:
            name = ent["canonical"]
            canon_key = key_fn(name)
            if canon_key in rejected_keys:
                continue
            add_gold(group, name, "TERM", (ent["start"], ent["end"]),
                     "cmeee_annotated_reference_exact_match", "reference_curated",
                     "CMeEE_train.json", TOPIC_MEDICAL)

    for row in professional_samples:
        info = sample_source_info(row, "professional")
        if is_commonvoice(row["source"]):
            if info["source_id"] in old_cv_ids or row["reference"] in old_cv_refs:
                raise ValueError(f"Previously evaluated Common Voice item leaked into professional staging: {info['source_id']}")
        if row["source"] == "SpeechIO":
            if info["segment_id"] in old_speech_source_ids or row["reference"] in old_speech_refs:
                raise ValueError(f"Previously evaluated SpeechIO item leaked into professional staging: {info['segment_id']}")
        if info["group_key"] in quarantined_group_keys:
            continue
        group = merge_source_record(groups, info, row["reference"], None, key_fn)
        topic = row["topic"]
        group["topics"].add(topic)
        group["cohorts"].add("professional_reference_curated")
        span = row.get("entity_span")
        if span is None:
            span = row.get("span") or [row["start"], row["end"]]
        name = row["canonical"]
        canon_key = key_fn(name)
        if canon_key in rejected_keys:
            continue
        method = row.get("review_method", "reference_curated")
        add_gold(group, name, row.get("type", "TERM"), span, method, "reference_curated",
                 row["source"], topic)

    # Remove duplicate mentions created when the same recording is represented by
    # multiple staging rows for the same curated standard name/span.
    for group in groups.values():
        unique = {}
        for ent in group["entities"]:
            marker = (ent["key"], ent["start"], ent["end"], ent["annotation_method"], ent["source_type"])
            unique[marker] = ent
        group["entities"] = list(unique.values())

    # Attach exact SpeechIO rows only for positive groups that survived policy filtering.
    source_candidate_groups = [group for group in groups.values() if group["entities"]]
    speechio_payloads = read_speechio_payloads(source_candidate_groups)
    for group in source_candidate_groups:
        for info in group["source_records"]:
            if info["source"] == "SpeechIO":
                raw_payload = speechio_payloads[info["group_key"]]
                info["embedded_audio_bytes"] = raw_payload
                info["source_audio_sha256"] = sha256_bytes(raw_payload)
                info["embedded_wave_payload"] = True
                break
        else:
            info = group["source_records"][0]
            source_path = Path(info["source_path"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            info["source_audio_sha256"] = sha256_file(source_path)
            info["embedded_wave_payload"] = False
        group["source_audio_sha256"] = info["source_audio_sha256"]
        group["audio_info"] = info

    # De-duplicate identical bytes even when two staging records use different paths.
    deduped, by_audio_hash = [], {}
    quarantined_hash_matches = []
    for group in sorted(groups.values(), key=lambda x: x["group_key"]):
        if not group["entities"]:
            continue
        h = group["source_audio_sha256"]
        if h in quarantined_audio_hashes:
            quarantined_hash_matches.append({"canonical_candidate": None, "canonical_key": None,
                "reason_code": "duplicate_of_quarantine_recording",
                "reason": "Raw source audio bytes match a policy-quarantined recording.",
                "source_group_key": group["group_key"],
                "candidate_entities": sorted({ent["name"] for ent in group["entities"]}),
                "source_records": [{k: v for k, v in r.items() if k != "embedded_audio_bytes"}
                                   for r in group["source_records"]]})
            quarantine_candidate_names.update(ent["name"] for ent in group["entities"])
            continue
        previous = by_audio_hash.get(h)
        if previous is None:
            by_audio_hash[h] = group
            deduped.append(group)
            continue
        if previous["reference"] != group["reference"]:
            raise ValueError(f"Duplicate source audio hash has inconsistent references: {h}")
        previous["source_records"].extend(group["source_records"])
        unique_records = {}
        for record in previous["source_records"]:
            marker = (record.get("source"), record.get("source_id"), record.get("source_path"),
                      record.get("parquet"), record.get("row_index"), record.get("sample_id"))
            unique_records[marker] = record
        previous["source_records"] = list(unique_records.values())
        previous["topics"].update(group["topics"])
        previous["cohorts"].update(group["cohorts"])
        previous["entities"].extend(group["entities"])
        for field in ("aishell_original_recording_id", "aishell_source_ner_entities", "aishell_entities_filtered"):
            if field in group and field not in previous:
                previous[field] = group[field]

    # Stable new sample IDs are derived from source bytes and normalized reference.
    for group in deduped:
        group["id"] = "leb1_" + hashlib.sha256(
            (group["source_audio_sha256"] + "\0" + key_fn(group["reference"])).encode("utf-8")).hexdigest()[:20]
    if len({g["id"] for g in deduped}) != len(deduped):
        raise ValueError("Deterministic sample ID collision")

    # Filter any source candidate whose entire sample annotation was rejected.
    positive_groups = [g for g in deduped if g["entities"]]
    all_entity_keys = {entry["key"] for g in positive_groups for entry in g["entities"]}
    if not all_entity_keys:
        raise ValueError("No accepted sample-backed entities")

    # Do not promote a shorter curated item when it is only nested inside a
    # longer span in the same utterance. Keep separate valid evidence elsewhere.
    nested_q = []
    for group in positive_groups:
        ents = group["entities"]
        remove = set()
        for i, shorter in enumerate(ents):
            if shorter["annotation_layer"] != "reference_curated":
                continue
            for j, longer in enumerate(ents):
                if i == j or shorter["key"] == longer["key"]:
                    continue
                if longer["start"] <= shorter["start"] and longer["end"] >= shorter["end"] and (
                    longer["start"] < shorter["start"] or longer["end"] > shorter["end"]):
                    remove.add(i)
                    nested_q.append({"canonical_candidate": shorter["name"], "canonical_key": shorter["key"],
                                     "reason_code": "nested_reference_curated_span",
                                     "reason": "This utterance only supports the shorter term inside a longer curated entity span.",
                                     "sample_group_key": group["group_key"], "sample_id": group["id"],
                                     "outer_canonical": longer["name"], "span": [shorter["start"], shorter["end"]],
                                     "outer_span": [longer["start"], longer["end"]]})
                    break
        if remove:
            group["entities"] = [ent for i, ent in enumerate(ents) if i not in remove]
    positive_groups = [g for g in positive_groups if g["entities"]]

    # Choose one display spelling per normalized canonical. An official NER
    # spelling wins; otherwise the manually curated seed spelling wins CMeEE.
    entity_mentions = defaultdict(list)
    for group in positive_groups:
        for ent in group["entities"]:
            entity_mentions[ent["key"]].append((group, ent))
    entity_records = []
    entity_by_key = {}
    for key, mentions in sorted(entity_mentions.items()):
        ner_names = sorted({ent["name"] for _, ent in mentions if ent["annotation_layer"] == "native_ner"})
        curated_names = sorted({ent["name"] for _, ent in mentions if ent["annotation_layer"] == "reference_curated"})
        name = (ner_names or curated_names)[0]
        ner_types = {ent["source_type"] for _, ent in mentions if ent["annotation_layer"] == "native_ner"}
        if len(ner_types) > 1:
            raise ValueError(f"Unquarantined NER conflict in accepted entity: {name} {ner_types}")
        entity_type = next(iter(ner_types)) if ner_types else "TERM"
        topics = sorted({ent["topic"] for _, ent in mentions})
        sources = sorted({ent["source"] for _, ent in mentions})
        methods = sorted({ent["annotation_method"] for _, ent in mentions})
        layers = sorted({ent["annotation_layer"] for _, ent in mentions})
        variants = sorted({ent["name"] for _, ent in mentions})
        evidence = []
        for group, ent in sorted(mentions, key=lambda pair: (pair[0]["id"], pair[1]["start"], pair[1]["end"])):
            evidence.append({"sample_id": group["id"], "start": ent["start"], "end": ent["end"],
                             "surface": ent["surface"], "source_type": ent["source_type"],
                             "annotation_method": ent["annotation_method"],
                             "annotation_layer": ent["annotation_layer"], "source": ent["source"],
                             "topic": ent["topic"]})
        row = {"canonical_text": name, "canonical_key": key, "entity_type": entity_type,
               "domain": "general", "normalization_policy": "normalize", "aliases": [],
               "source": "; ".join(sources), "topics": topics,
               "observed_standard_spellings": variants, "observed_source_types": sorted({ent["source_type"] for _, ent in mentions}),
               "annotation_layers": layers, "annotation_methods": methods,
               "evidence_count": len(evidence), "supporting_sample_ids": sorted({x["sample_id"] for x in evidence}),
               "evidence": evidence}
        entity_by_key[key] = row
        entity_records.append(row)

    # Re-key sample mentions to the selected global display spelling/type.
    for group in positive_groups:
        for ent in group["entities"]:
            global_entity = entity_by_key[ent["key"]]
            ent["canonical"] = global_entity["canonical_text"]
            ent["entity_type"] = global_entity["entity_type"]
            ent["topics"] = global_entity["topics"]
        group["entities"].sort(key=lambda e: (e["start"], e["end"], e["canonical"]))

    # Only AISHELL recordings can supply negatives. Distinguish truly empty
    # source NER from NER-bearing samples whose labels are not in the registry.
    aishell_positive_ids = {g.get("aishell_original_recording_id") for g in positive_groups
                            if g.get("aishell_original_recording_id")}
    registered_keys = set(entity_by_key)
    negative_candidates = []
    for row in aishell_samples:
        rid = row["recording_id"]
        if rid in old_aishell or rid in aishell_positive_ids:
            continue
        reference_original = row["reference"]
        reference = compact_with_offset_map(reference_original)[0]
        normalized_ref = key_fn(reference)
        if any(key in normalized_ref for key in registered_keys):
            continue
        raw_entities = raw_annotations.get(rid, {"entities": []})["entities"]
        audio_path = Path(row["audio_path"]).resolve()
        negative_candidates.append({"row": row, "rid": rid, "reference": reference,
                                    "reference_original": reference_original,
                                    "raw_entities": raw_entities, "audio_path": audio_path,
                                    "audio_sha256": row["audio_sha256"]})
    negative_candidates.sort(key=lambda x: hashlib.sha256(
        f"{SELECTION_SEED}:{x['rid']}".encode("utf-8")).hexdigest())
    if len(negative_candidates) < 300:
        raise ValueError(f"Only {len(negative_candidates)} eligible AISHELL negatives; at least 300 required")
    negative_groups = []
    for item in negative_candidates[:300]:
        rid, row = item["rid"], item["row"]
        group = {"id": "leb1_" + hashlib.sha256((item["audio_sha256"] + "\0" + key_fn(item["reference"])).encode("utf-8")).hexdigest()[:20],
                 "reference": item["reference"], "reference_original": item["reference_original"],
                 "entities": [], "topics": [TOPIC_AISHELL], "cohorts": ["aishell_test"],
                 "sample_class": "negative",
                 "negative_subtype": "negative_no_source_ner" if not item["raw_entities"] else "negative_uncovered_source_ner",
                 "source_ner_entities": item["raw_entities"],
                 "source_records": [sample_source_info(row, "aishell")],
                 "audio_info": {"source_path": str(item["audio_path"]), "source_audio_sha256": item["audio_sha256"],
                                "source": row["source"], "tag": "aishell", "embedded_wave_payload": False},
                 "source_audio_sha256": item["audio_sha256"], "aishell_original_recording_id": rid}
        if any(key in key_fn(group["reference"]) for key in registered_keys):
            raise AssertionError("Negative sample contains a registered canonical surface")
        negative_groups.append(group)

    main_groups = sorted(positive_groups + negative_groups, key=lambda g: g["id"])
    if len({g["id"] for g in main_groups}) != len(main_groups):
        raise ValueError("Positive/negative sample ID collision")
    if len({g["source_audio_sha256"] for g in main_groups}) != len(main_groups):
        raise ValueError("Duplicate raw audio bytes remain in main sample set")
    positive_count = len(positive_groups)
    if positive_count < 2000:
        raise ValueError(f"Only {positive_count} unique positive recordings; expected at least around 2,000")
    if len(entity_records) < quality["min_unique_entities"]:
        raise ValueError(f"Only {len(entity_records)} unique entities; minimum is {quality['min_unique_entities']}")
    term_count = sum(row["entity_type"] == "TERM" for row in entity_records)
    if term_count < quality["min_professional_terms"]:
        raise ValueError(f"Only {term_count} TERM entities; minimum is {quality['min_professional_terms']}")
    topic_names = sorted({topic for row in entity_records for topic in row["topics"]})
    if len(topic_names) < quality["min_topics"]:
        raise ValueError(f"Only {len(topic_names)} topics; minimum is {quality['min_topics']}")

    qrows = get_quarantine(quality, reject_ner, reject_terms, conflict_keys, by_key,
                           raw_annotations, aishell_by_id, key_fn)
    qrows.extend(quarantine_recording_rows)
    qrows.extend(quarantined_hash_matches)
    for candidate_name in sorted(quarantine_candidate_names):
        candidate_key = key_fn(candidate_name)
        if candidate_key not in entity_by_key:
            qrows.append({"canonical_candidate": candidate_name, "canonical_key": candidate_key,
                          "reason_code": "only_support_in_quarantined_audio",
                          "reason": "This standard name has no remaining non-quarantined evidence in the main set."})
    qrows.extend(nested_q)
    # Preserve a full audit trail for all names removed by policy/conflict.
    qrows.sort(key=lambda x: (x["reason_code"], x["canonical_key"], x.get("canonical_candidate", "")))

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    temp_root = Path(tempfile.mkdtemp(prefix=".dataset_build_", dir=BASE))
    try:
        audio_dir, pcm_dir, library_dir = temp_root / "audio", temp_root / "pcm", temp_root / "library"
        audio_dir.mkdir()
        pcm_dir.mkdir()
        library_dir.mkdir()

        # Extract the exact selected SpeechIO parquet rows. Other sources stay at
        # their verified original paths.
        build_audio_paths = {}
        for group in main_groups:
            info = group["audio_info"]
            if info.get("embedded_wave_payload"):
                selected_audio = audio_dir / f"{group['id']}.wav"
                selected_audio.write_bytes(info["embedded_audio_bytes"])
                info["selected_audio_path"] = str(selected_audio.resolve())
                info["selected_audio_sha256"] = sha256_file(selected_audio)
            else:
                source_path = Path(info["source_path"])
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                info["selected_audio_path"] = str(source_path)
                info["selected_audio_sha256"] = sha256_file(source_path)
            build_audio_paths[group["id"]] = Path(info["selected_audio_path"])

        jobs = []
        for group in main_groups:
            pcm_path = pcm_dir / f"{group['id']}.f32le"
            jobs.append((group["id"], Path(group["audio_info"]["selected_audio_path"]), pcm_path))
        pcm_meta = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_audio, job): job[0] for job in jobs}
            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                sid, pcm_hash, pcm_bytes, duration = future.result()
                pcm_meta[sid] = {"pcm_sha256": pcm_hash, "pcm_bytes": pcm_bytes,
                                 "duration_seconds": duration}
                if i % 250 == 0 or i == len(jobs):
                    print(f"PCM {i}/{len(jobs)}", flush=True)

        # Construct frozen sample rows and check audio metadata against ffprobe.
        final_samples = []
        for group in main_groups:
            sid = group["id"]
            info = group["audio_info"]
            build_audio_path = build_audio_paths[sid]
            pcm = pcm_meta[sid]
            duration = pcm["duration_seconds"]
            sample_rate = 16000
            decoded_duration = pcm["pcm_bytes"] / (4 * sample_rate)
            if abs(decoded_duration - duration) > max(0.15, duration * 0.02):
                raise ValueError(f"ffprobe/decoded PCM duration mismatch for {sid}: {duration} vs {decoded_duration}")
            if group.get("tag") == "aishell" or any(x.get("tag") == "aishell" for x in group.get("source_records", [])):
                # Random-check the staging float32 hash convention against ffmpeg output.
                aishell_record = next((x for x in group["source_records"] if x.get("tag") == "aishell"), None)
                if aishell_record:
                    source_row = aishell_by_id.get(aishell_record["recording_id"])
                    if source_row and source_row.get("pcm_sha256") and source_row["pcm_sha256"] != pcm["pcm_sha256"]:
                        raise ValueError(f"AISHELL pre-verified float32 PCM hash changed: {aishell_record['recording_id']}")
            source_records = []
            for rec in group.get("source_records", []):
                source_records.append({k: v for k, v in rec.items()
                                       if k not in {"embedded_audio_bytes", "embedded_wave_payload"}})
            if group.get("sample_class") == "negative":
                topics, cohort = group["topics"], "negative"
                gold = []
            else:
                topics = sorted(group["topics"])
                cohort = "positive"
                gold = [{"canonical": e["canonical"], "canonical_key": e["key"],
                         "entity_type": e["entity_type"], "source_type": e["source_type"],
                         "surface": e["surface"], "start": e["start"], "end": e["end"],
                         "annotation_method": e["annotation_method"],
                         "annotation_layer": e["annotation_layer"], "topic": e["topic"]}
                        for e in group["entities"]]
            final_audio_path = (OUTPUT / "audio" / f"{sid}.wav") if info.get("embedded_wave_payload") else Path(info["selected_audio_path"])
            row = {"id": sid, "source": source_records[0]["source"] if len({x["source"] for x in source_records}) == 1 else "merged_sources",
                   "sources": sorted({x["source"] for x in source_records}),
                   "recording_id": source_records[0]["recording_id"] if len(source_records) == 1 else None,
                   "source_records": source_records,
                   "audio_path": str(final_audio_path), "pcm_path": str((OUTPUT / "pcm" / f"{sid}.f32le").resolve()),
                   "audio_sha256": info["selected_audio_sha256"], "source_audio_sha256": group["source_audio_sha256"],
                   "pcm_sha256": pcm["pcm_sha256"], "duration_seconds": duration,
                   "sample_rate": 16000, "channels": 1, "sample_format": "f32le",
                   "reference": group["reference"], "topics": topics, "cohort": cohort,
                   "cohorts": sorted(group.get("cohorts", [])), "gold_entities": gold}
            if group.get("reference_original") is not None:
                row["reference_original"] = group["reference_original"]
            if group.get("sample_class") == "negative":
                row["negative_subtype"] = group["negative_subtype"]
                row["source_ner_entities"] = group["source_ner_entities"]
            final_samples.append(row)

        # Seen FAQ rows remain isolated and do not contribute to the library.
        regression_rows = []
        for row in regression_samples:
            path = Path(row["audio_path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            regression_rows.append({"id": "regression_" + str(row["recording_id"]),
                                    "source": row["source"], "recording_id": row["recording_id"],
                                    "audio_path": str(path), "audio_sha256": sha256_file(path),
                                    "reference": row["reference"], "duration_seconds": ffprobe_duration(path),
                                    "cohort": "seen_regression", "included_in_main": False,
                                    "gold_entities": []})

        # Independent semantic validation before publishing any files.
        ids = [row["id"] for row in final_samples]
        if len(ids) != len(set(ids)):
            raise AssertionError("Duplicate sample IDs")
        if len({row["source_audio_sha256"] for row in final_samples}) != len(final_samples):
            raise AssertionError("Duplicate underlying recordings remain")
        all_reference_by_id = {row["id"]: row for row in final_samples}
        evidence_sample_ids = set()
        canonical_coverage = Counter()
        for row in final_samples:
            audio_path = build_audio_paths[row["id"]]
            pcm_path = pcm_dir / f"{row['id']}.f32le"
            if not audio_path.is_file() or not pcm_path.is_file():
                raise AssertionError(f"Missing final selected file: {row['id']}")
            if sha256_file(audio_path) != row["audio_sha256"] or sha256_file(pcm_path) != row["pcm_sha256"]:
                raise AssertionError(f"Final audio hash mismatch: {row['id']}")
            compact_ref, _ = compact_with_offset_map(row.get("reference_original", row["reference"]))
            if row.get("reference_original") is not None and compact_ref != row["reference"]:
                raise AssertionError(f"AISHELL reference compaction mismatch: {row['id']}")
            if row["cohort"] == "negative":
                if row["gold_entities"]:
                    raise AssertionError(f"Negative has gold entities: {row['id']}")
                normalized_ref = key_fn(row["reference"])
                if any(key in normalized_ref for key in registered_keys):
                    raise AssertionError(f"Negative contains a registered canonical: {row['id']}")
            else:
                if not row["gold_entities"]:
                    raise AssertionError(f"Positive missing gold: {row['id']}")
                for ent in row["gold_entities"]:
                    if not (0 <= ent["start"] < ent["end"] <= len(row["reference"])):
                        raise AssertionError(f"Span out of range: {row['id']} {ent}")
                    surface = row["reference"][ent["start"]:ent["end"]]
                    if key_fn(surface) != ent["canonical_key"]:
                        raise AssertionError(f"Span does not support canonical: {row['id']} {ent}")
                    canonical_coverage[ent["canonical_key"]] += 1
                    evidence_sample_ids.add(row["id"])
        if set(entity_by_key) != set(canonical_coverage):
            missing = set(entity_by_key) - set(canonical_coverage)
            extra = set(canonical_coverage) - set(entity_by_key)
            raise AssertionError(f"Entity evidence coverage mismatch missing={len(missing)} extra={len(extra)}")
        for record in entity_records:
            if not record["supporting_sample_ids"] or not all(sid in all_reference_by_id for sid in record["supporting_sample_ids"]):
                raise AssertionError(f"Invalid support list for {record['canonical_text']}")
        negative_counts = Counter(row["negative_subtype"] for row in final_samples if row["cohort"] == "negative")
        if sum(negative_counts.values()) != 300:
            raise AssertionError("Negative sample count changed")
        for row in final_samples:
            for source_record in row["source_records"]:
                if source_record["tag"] == "aishell" and source_record["recording_id"] in old_aishell:
                    raise AssertionError("Historical AISHELL ID in output")
                if is_commonvoice(source_record["source"]):
                    if source_record["source_id"] in old_cv_ids or row["reference"] in old_cv_refs:
                        raise AssertionError("Historical Common Voice sample/ref in output")
                if source_record["source"] == "SpeechIO":
                    if source_record["source_id"] in old_speech_source_ids or row["reference"] in old_speech_refs:
                        raise AssertionError("Historical SpeechIO sample/ref in output")

        # Write finalized JSON/JSONL artifacts.
        (temp_root / "entities.json").write_text(json.dumps(entity_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temp_root / "quarantine.json").write_text(json.dumps(qrows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temp_root / "references.json").write_text(json.dumps([
            {"sample_id": row["id"], "reference": row["reference"], **({"reference_original": row["reference_original"]} if "reference_original" in row else {})}
            for row in final_samples], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temp_root / "sample_entities.json").write_text(json.dumps([
            {"sample_id": row["id"], "cohort": row["cohort"], "gold_entities": row["gold_entities"]}
            for row in final_samples], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (temp_root / "samples.jsonl").open("w", encoding="utf-8") as f:
            for row in final_samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (temp_root / "regression_samples.jsonl").open("w", encoding="utf-8") as f:
            for row in regression_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Create isolated EntityStore-compatible databases. Every entity is in
        # domain=general for UI compatibility; per-topic DBs are independent.
        from system.entity_store import EntityStore
        main_store = EntityStore(library_dir / "entities.db")
        for row in entity_records:
            main_store.upsert_entity(row["canonical_text"], entity_type=row["entity_type"],
                                     domain="general", normalization_policy="normalize",
                                     aliases=(), source=row["source"] or "large_entity_benchmark_v1")
        for index, topic in enumerate(topic_names, start=1):
            topic_store = EntityStore(library_dir / f"topic_{index:02d}_{topic}.db")
            for row in entity_records:
                if topic in row["topics"]:
                    topic_store.upsert_entity(row["canonical_text"], entity_type=row["entity_type"],
                                              domain="general", normalization_policy="normalize",
                                              aliases=(), source=row["source"] or "large_entity_benchmark_v1")
        # Validate SQLite row counts and absence of aliases.
        with sqlite3.connect(library_dir / "entities.db") as db:
            db_count = db.execute("SELECT COUNT(*) FROM protected_entities WHERE domain='general' AND enabled=1").fetchone()[0]
            alias_count = db.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0]
        if db_count != len(entity_records) or alias_count != 0:
            raise AssertionError(f"Main EntityStore mismatch: rows={db_count}, aliases={alias_count}")

        # Summary and frozen hash manifest.
        type_counts = Counter(row["entity_type"] for row in entity_records)
        topic_entity_counts = {topic: sum(topic in row["topics"] for row in entity_records) for topic in topic_names}
        source_entity_counts = Counter()
        for row in entity_records:
            for source in row["source"].split("; "):
                source_entity_counts[source] += 1
        layer_entity_counts = {layer: sum(layer in row["annotation_layers"] for row in entity_records)
                               for layer in ("native_ner", "reference_curated")}
        sample_source_counts = Counter(src for row in final_samples for src in row["sources"])
        cohort_counts = Counter(row["cohort"] for row in final_samples)
        total_seconds = sum(row["duration_seconds"] for row in final_samples)
        source_variant_collisions = sum(len(row["observed_standard_spellings"]) > 1 for row in entity_records)
        canonical_key_counts = Counter(key_fn(entry["canonical"]) for entry in raw_entries)
        duplicate_source_rows = sum(max(0, n - 1) for n in canonical_key_counts.values())
        nested_count = len(nested_q)
        summary = {
            "dataset": "large_entity_benchmark_v1_20261010",
            "positive_recordings": positive_count,
            "positive_target": quality["target_positive_audio"],
            "positive_shortfall": max(0, quality["target_positive_audio"] - positive_count),
            "negative_recordings": sum(cohort_counts.values()) - positive_count,
            "negative_by_subtype": dict(negative_counts),
            "total_main_recordings": len(final_samples),
            "distinct_canonical_entities": len(entity_records),
            "distinct_entities_by_type": dict(type_counts),
            "minimum_entities_met": len(entity_records) >= quality["min_unique_entities"],
            "professional_term_count": type_counts["TERM"],
            "minimum_terms_met": type_counts["TERM"] >= quality["min_professional_terms"],
            "topics": topic_names,
            "distinct_entities_by_topic": topic_entity_counts,
            "minimum_topics_met": len(topic_names) >= quality["min_topics"],
            "entities_by_annotation_layer_distinct_nonexclusive": layer_entity_counts,
            "annotation_method_entity_counts_distinct_nonexclusive": {
                method: sum(method in row["annotation_methods"] for row in entity_records)
                for method in sorted({method for row in entity_records for method in row["annotation_methods"]})
            },
            "entity_sources_distinct": dict(source_entity_counts),
            "samples_by_source": dict(sample_source_counts),
            "sample_cohorts": dict(cohort_counts),
            "total_audio_duration_seconds": round(total_seconds, 3),
            "seen_regression_samples_separate": len(regression_rows),
            "quarantine_entries": len(qrows),
            "policy_reject_ner_names": len(reject_ner),
            "policy_reject_term_names": len(reject_terms),
            "cross_type_conflict_keys": len(conflict_keys),
            "nested_curated_span_occurrences_quarantined": nested_count,
            "policy_quarantined_recording_ids": sorted(quality.get("quarantine_recording_ids", [])),
            "policy_quarantined_recording_count": len(quality.get("quarantine_recording_ids", [])),
            "quarantine_only_supported_name_count": sum(row["reason_code"] == "only_support_in_quarantined_audio" for row in qrows),
            "canonical_dedup_stats": {
                "raw_staging_entity_rows": len(raw_entries),
                "distinct_nfkc_t2s_canonical_keys_before_rejects": len(canonical_key_counts),
                "duplicate_rows_collapsed_by_canonical_key": duplicate_source_rows,
                "distinct_keys_with_multiple_display_spellings": source_variant_collisions,
                "ner_conflicts_removed": sum(1 for key in conflict_keys if any(e["key"] == key and e["layer"] == "native_ner" for e in raw_entries)),
                "same_name_across_topic_counted_once": True,
            },
            "quality_layers": {
                "native_ner": "AISHELL-NER source spans with reference alignment; no per-item human audio listening claim",
                "reference_curated": "Professional curated seeds and CMeEE-annotated terms matched to supplied source references; audio not individually listened to",
                "derivative_reference_match_distinct_entities": sum("derivative_reference_match" in row["annotation_methods"] for row in entity_records),
                "unreviewed_draft_candidates_included": 0,
            },
            "history_exclusions": {"aishell_recordings": len(old_aishell), "commonvoice_sample_ids": len(old_cv_ids),
                                   "speechio_source_ids": len(old_speech_source_ids),
                                   "old_faq_regression_not_in_main": True},
            "audio": {"format": "mono 16 kHz float32 little-endian raw PCM", "pcm_extension": ".f32le",
                      "conversion": "ffmpeg -ac 1 -ar 16000 -c:a pcm_f32le -f f32le",
                      "duration_source": "ffprobe per selected source/extracted audio"},
            "limitations": [
                "Positive audio target was 2,500; only unique source-backed aligned/curated recordings are included, with no duplication to fill the shortfall.",
                "AISHELL topics represent news PERSON/ORG/LOC coverage and are separate from the eight professional seed topics.",
                "Reference evidence and supplied annotations were checked; every audio was decoded, but not every item was manually listened to.",
                "Canonical names are source spellings selected deterministically; no authoritative external entity linking or unverified aliases are claimed.",
                "Distinct counts are by NFKC + simplified canonical spelling, not by real-world identity; alias resolution is incomplete and aliases remain empty.",
                "Two policy-quarantined high-rate Common Voice recordings are excluded pending separate audio review; names supported only by them are quarantined too.",
            ],
        }
        (temp_root / "SUMMARY.md").write_text(render_summary(summary), encoding="utf-8")
        (temp_root / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        input_hashes = {}
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                input_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
        input_sources = [QUALITY, QUALITY_READY, AISHELL_NER_SOURCE, AISHELL_CORPUS / "transcript/aishell_transcript_v0.8.txt",
                         Path(__file__).resolve(), *HISTORICAL_EXCLUSION_FILES,
                         *sorted((ROOT / "results/speechio_professional_entity_eval_20261010").glob("*/protocol.json"))]
        for path in input_sources:
            if path.is_file():
                try:
                    label = str(path.relative_to(ROOT))
                except ValueError:
                    label = str(path)
                input_hashes[label] = sha256_file(path)
        output_hashes = {}
        for path in sorted(temp_root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                output_hashes[str(path.relative_to(temp_root))] = sha256_file(path)
        manifest = {
            "dataset": "large_entity_benchmark_v1_20261010",
            "created_by": "experiments/scripts/build_large_entity_benchmark.py",
            "selection_seed": SELECTION_SEED,
            "frozen": True,
            "manifest_hash_scope": "All listed input/output files; manifest.json excludes its own hash to avoid recursion.",
            "quality_policy_sha256": sha256_file(QUALITY),
            "staging_input_sha256": input_hashes,
            "output_sha256": output_hashes,
            "sample_audio_sha256": {row["id"]: row["audio_sha256"] for row in final_samples},
            "sample_pcm_sha256": {row["id"]: row["pcm_sha256"] for row in final_samples},
            "sample_count": len(final_samples), "entity_count": len(entity_records),
        }
        (temp_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Final manifest and all file hashes are rechecked immediately before the
        # atomic publish. Existing output is never overwritten.
        for rel, expected in output_hashes.items():
            if sha256_file(temp_root / rel) != expected:
                raise AssertionError(f"Output changed before freeze: {rel}")
        if OUTPUT.exists():
            raise FileExistsError(OUTPUT)
        os.replace(temp_root, OUTPUT)
        print(json.dumps({"output": str(OUTPUT), "positive": positive_count,
                          "negative": len(final_samples) - positive_count,
                          "entities": len(entity_records), "terms": type_counts["TERM"],
                          "topics": len(topic_names), "duration_seconds": round(total_seconds, 3)},
                         ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        raise


def render_summary(stats):
    by_type = "、".join(f"{key}: {value}" for key, value in sorted(stats["distinct_entities_by_type"].items()))
    by_topic = "\n".join(f"- {topic}: {count} 个实体" for topic, count in stats["distinct_entities_by_topic"].items())
    by_source = "、".join(f"{key}: {value}" for key, value in sorted(stats["entity_sources_distinct"].items()))
    by_layer = "、".join(f"{key}: {value}" for key, value in stats["entities_by_annotation_layer_distinct_nonexclusive"].items())
    limitations = "\n".join(f"- {item}" for item in stats["limitations"])
    return f"""# Large Entity Benchmark v1

本目录是待主代理复核的冻结候选数据集，未写入生产代码或正式用户实体库。

## 覆盖

- 主样本：{stats['positive_recordings']} 条正例录音，{stats['negative_recordings']} 条负例；正例目标为 2,500，实际不足部分不通过重复录音补齐。
- 全局唯一标准名：{stats['distinct_canonical_entities']} 个；按主实体类型：{by_type}。
- TERM：{stats['professional_term_count']} 个；话题：{len(stats['topics'])} 个。
- 总音频时长：{stats['total_audio_duration_seconds']:.1f} 秒；旧 FAQ regression 单独保留 {stats['seen_regression_samples_separate']} 条，不进入主集。
- 标注分层（非互斥去重实体数）：{by_layer}。
- 来源实体覆盖：{by_source}。
- 规范名按 NFKC + 简体转换去重；同名多话题只计一次，跨来源优先采用官方 NER 类型，保留来源观察值。别名均为空。

## 话题覆盖

{by_topic}

## 质量与局限

AISHELL NER spans、CMeEE 词条到参考文本的精确匹配、人工整理的专业术语 seed 分层记录在 entities.json 和 sample_entities.json 中。`draft_candidates.json` 未进入主库。负例按固定 SHA-256 顺序抽取，参考文本不含任何已注册规范名，并区分原 NER 为空与含未注册 NER。所有选中音频均经 ffprobe 与 ffmpeg 解码检查，保存 16 kHz 单声道 float32 little-endian PCM；这不代表逐条人工听审。

{limitations}

质量拒绝与 NER 跨类型冲突记录在 quarantine.json。manifest.json 冻结输入 staging、质量政策和所有输出文件 SHA-256。实体库为独立 SQLite 文件，topic 数据库也各自使用 domain=general 以兼容现有 UI。
"""


if __name__ == "__main__":
    main()
