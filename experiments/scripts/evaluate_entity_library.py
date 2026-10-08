"""Reproducible entity-dictionary evaluation on real AISHELL-1 recordings.

Run with the project root on PYTHONPATH. The dictionary is built exclusively
from train/dev annotations, before decoding the held-out test audio. No test
transcription is used to populate aliases or tune matching thresholds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
NER_PATTERN = re.compile(r"\[([^\[\]]+)\]|\(([^()]+)\)|<([^<>]+)>")
TYPES = ("PERSON", "LOC", "ORG")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] not in {"P", "Z", "C"})


def parse_ner(text):
    entities = []
    parts = []
    previous = 0
    for match in NER_PATTERN.finditer(text):
        parts.append(text[previous:match.start()])
        index = next(i for i, group in enumerate(match.groups()) if group is not None)
        surface = match.group(index + 1)
        start = sum(map(len, parts))
        parts.append(surface)
        entities.append(dict(text=surface, entity_type=TYPES[index], start=start, end=start+len(surface)))
        previous = match.end()
    parts.append(text[previous:])
    return "".join(parts), entities


def balanced_sample(rows, count, rng):
    """Round-robin over speakers, with all ordering fixed by the seed."""
    by_speaker = {}
    for row in sorted(rows, key=lambda item: item["id"]):
        by_speaker.setdefault(row["speaker"], []).append(row)
    for pool in by_speaker.values():
        rng.shuffle(pool)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    selected = []
    while len(selected) < count:
        progressed = False
        for speaker in speakers:
            if by_speaker[speaker] and len(selected) < count:
                selected.append(by_speaker[speaker].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"pool only contains {len(selected)} samples, requested {count}")
    return selected


def prepare(args):
    from system.entity_store import EntityStore
    annotations = {}
    for line in args.annotations.read_text().splitlines():
        record_id, text = line.split(maxsplit=1)
        annotations[record_id] = parse_ner(text)
    transcript = {}
    for line in (args.corpus / "transcript/aishell_transcript_v0.8.txt").read_text().splitlines():
        record_id, text = line.split(maxsplit=1)
        transcript[record_id] = "".join(text.split())
    frequencies = Counter()
    for split in ("train", "dev"):
        for path in sorted((args.corpus / "wav" / split).rglob("*.wav")):
            for entity in annotations.get(path.stem, ("", []))[1]:
                if len(entity["text"]) >= 3:
                    frequencies[(entity["text"], entity["entity_type"])] += 1
    ranked = sorted(frequencies, key=lambda key: (-frequencies[key], key))
    vocabulary = []
    seen = set()
    for text, kind in ranked:
        if text in seen:
            continue
        seen.add(text)
        vocabulary.append(dict(canonical_text=text, entity_type=kind,
                               train_dev_occurrences=frequencies[(text, kind)]))
        if len(vocabulary) == args.vocabulary_size:
            break
    known = {row["canonical_text"] for row in vocabulary}
    pools = {"covered_entity": [], "uncovered_entity": [], "no_entity": []}
    mismatches = []
    for path in sorted((args.corpus / "wav/test").rglob("*.wav")):
        if path.stem not in annotations:
            continue
        ner_reference, entities = annotations[path.stem]
        reference = transcript[path.stem]
        if normalized(ner_reference) != normalized(reference):
            mismatches.append(path.stem)
            continue
        for entity in entities:
            entity["covered"] = entity["text"] in known
        group = ("covered_entity" if any(e["covered"] for e in entities)
                 else "uncovered_entity" if entities else "no_entity")
        pools[group].append(dict(id=path.stem, speaker=path.parent.name,
                                audio_path=str(path), reference=reference,
                                entities=entities, group=group))
    rng = random.Random(args.seed)
    selected = []
    for group, count in (("covered_entity", 60), ("uncovered_entity", 20), ("no_entity", 20)):
        selected.extend(balanced_sample(pools[group], count, rng))
    write_json(args.output / "vocabulary.json", vocabulary)
    write_json(args.output / "selection.json", dict(seed=args.seed, vocabulary_size=len(vocabulary),
        vocabulary_source="AISHELL-1 train + dev only; >=3 characters; frequency-ranked",
        test_pool_sizes={key: len(value) for key, value in pools.items()},
        annotation_transcript_mismatches=mismatches, sample_count=len(selected),
        noise_snr_db=args.noise_snr, samples=selected))
    for variant in ("typed", "native", "all_term"):
        path = args.output / f"{variant}.db"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}; use a fresh output directory")
        store = EntityStore(path)
        for row in vocabulary:
            store.upsert_entity(row["canonical_text"],
                entity_type=(row["entity_type"] if variant == "typed" or
                    (variant == "native" and row["entity_type"] != "LOC") else "TERM"),
                normalization_policy="normalize", priority=0,
                source="AISHELL-NER train/dev; all_term is an explicitly labeled diagnostic")
    live_db = args.live_db
    if live_db and live_db.exists():
        source = sqlite3.connect(f"file:{live_db}?mode=ro", uri=True)
        destination = sqlite3.connect(args.output / "current_snapshot.db")
        source.backup(destination)
        destination.close()
        source.close()
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in tracked if (ROOT/name).is_file() and (ROOT/name).suffix in {".py", ".sh", ".html", ".json", ".cc", ".h"}
              and not name.startswith("audio.cpp/external/")}
    write_json(args.output / "provenance.json", dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        initial_git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        annotation_url="https://raw.githubusercontent.com/Alibaba-NLP/AISHELL-NER/main/data/aishell_ner_transcript.txt",
        annotation_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        corpus=str(args.corpus), source_hashes=hashes))
    print("PREPARED", len(selected), "recordings", len(vocabulary), "entities", flush=True)


def audio_bytes(row, variant, args):
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(row["audio_path"], dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("expected mono 16kHz AISHELL audio")
    if variant == "noise":
        seed = int(hashlib.sha256(f'{args.seed}:{row["id"]}'.encode()).hexdigest()[:8], 16)
        noise = np.random.default_rng(seed).standard_normal(audio.shape).astype("float32")
        rms = np.sqrt(np.mean(audio ** 2))
        audio = audio + noise * rms / (10 ** (args.noise_snr / 20)) / np.sqrt(np.mean(noise**2))
        peak = float(np.max(np.abs(audio)))
        if peak > 0.99:
            audio *= 0.99 / peak
    return audio.astype("<f4").tobytes(), len(audio) / rate


def decode(args):
    import httpx
    selection = json.loads((args.output / "selection.json").read_text())
    args.seed = selection["seed"]
    args.noise_snr = selection["noise_snr_db"]
    completed = {(row["id"], row["variant"]) for row in read_jsonl(args.output / "asr.jsonl")}
    with httpx.Client(timeout=180) as client:
        health = client.get(args.asr_url + "/health").raise_for_status().json()
        write_json(args.output / "asr_health.json", health)
        for index, row in enumerate(selection["samples"]):
            for variant in ("clean", "noise"):
                if (row["id"], variant) in completed:
                    continue
                pcm, duration = audio_bytes(row, variant, args)
                started = time.perf_counter()
                sid = client.post(args.asr_url + "/stream/start", params=dict(language="Chinese", chunk_seconds=0.5)).raise_for_status().json()["session_id"]
                payloads = []
                try:
                    # Match the browser's float32 transport, with one-second chunks.
                    for offset in range(0, len(pcm), 64000):
                        payloads.append(client.post(args.asr_url + "/stream/chunk", params=dict(session_id=sid), content=pcm[offset:offset+64000]).raise_for_status().json())
                    final = client.post(args.asr_url + "/stream/finish", params=dict(session_id=sid)).raise_for_status().json()
                except BaseException:
                    client.post(args.asr_url + "/stream/cancel", params=dict(session_id=sid))
                    raise
                result = dict(**row, variant=variant, noise_snr_db=args.noise_snr if variant == "noise" else None,
                              duration_seconds=duration, asr_elapsed_seconds=time.perf_counter()-started,
                              pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                              raw_text=final["text"], chunk_payloads=payloads, final_payload=final)
                append_jsonl(args.output / "asr.jsonl", result)
                print(f'ASR {index+1}/{len(selection["samples"])} {variant} {row["id"]}: {final["text"]}', flush=True)


def dictionary_outputs(text, definitions, mode, *, prepared_tools=None):
    from system.entity_matcher import EntityCandidateMatcher
    from system.entity_pipeline import prepare_entity_segment
    from system.protection import EntityProtector
    if prepared_tools is None:
        protector = EntityProtector(definitions, protect_numeric_spans=False)
        matcher = EntityCandidateMatcher(definitions, mode=mode) if definitions and mode != "off" else None
    else:
        protector, matcher = prepared_tools
    prepared = prepare_entity_segment(text, protector, matcher, allow_auto=True)
    return dict(text=prepared.baseline_text, normalizations=list(prepared.normalizations),
                hints=list(prepared.hints), candidates=[m.public_dict() for m in prepared.report.matches],
                matcher_latency_ms=prepared.report.matcher_latency_ms)


def match(args):
    from system.entity_matcher import EntityCandidateMatcher
    from system.entity_store import EntityStore
    from system.protection import EntityProtector
    databases = {key: EntityStore(args.output / f"{key}.db").list_entities(domain="general")
                 for key in ("typed", "native", "all_term", "current_snapshot") if (args.output/f"{key}.db").exists()}
    rows = read_jsonl(args.output / "asr.jsonl")
    destination = args.output / "entity_only.jsonl"
    completed = {(r["id"],r["variant"]):r for r in read_jsonl(destination)}
    prepared_tools = {(name, mode): (EntityProtector(definitions, protect_numeric_spans=False),
        EntityCandidateMatcher(definitions, mode=mode) if mode != "off" else None)
        for name, definitions in databases.items() for mode in ("off", "shadow", "hint", "auto")}
    expected_conditions = {"no_dictionary", *(f"{name}_{mode}" for name in databases
        for mode in ("off", "shadow", "hint", "auto"))}
    for index, row in enumerate(rows):
        key = (row["id"],row["variant"])
        previous = completed.get(key, {}).get("conditions", {})
        if expected_conditions.issubset(previous):
            continue
        conditions = dict(previous)
        if "no_dictionary" not in conditions:
            conditions["no_dictionary"] = dictionary_outputs(row["raw_text"], (), "off")
        for db_name, definitions in databases.items():
            for mode in ("off", "shadow", "hint", "auto"):
                if f"{db_name}_{mode}" in conditions:
                    continue
                conditions[f"{db_name}_{mode}"] = dictionary_outputs(row["raw_text"], definitions, mode,
                    prepared_tools=prepared_tools[(db_name, mode)])
        append_jsonl(destination, dict(id=row["id"],variant=row["variant"],conditions=conditions))
        if (index+1) % 20 == 0:
            print("MATCH", index+1, "/", len(rows), flush=True)
    # Retain one current record per audio variant after a resumed run that
    # introduced additional conditions. Until replacement, appended complete
    # records also support recovery if this process is interrupted.
    latest = {(r["id"],r["variant"]):r for r in read_jsonl(destination)}
    staged = destination.with_suffix(".tmp")
    staged.write_text("".join(json.dumps(latest[(r["id"],r["variant"])],ensure_ascii=False)+"\n" for r in rows))
    staged.replace(destination)


def diagnostics(args):
    """Explicit text stress cases, kept separate from real-audio statistics."""
    from system.entity_store import EntityStore
    from system.entity_matcher import EntityCandidateMatcher
    from system.entity_pipeline import prepare_entity_segment, finalize_entity_segment
    from system.protection import EntityProtector
    definitions = EntityStore(args.output/"current_snapshot.db").list_entities(domain="general")
    protector = EntityProtector(definitions, protect_numeric_spans=False)
    records = []
    cases = [
        ("exact_alias", "乌冬得水电站开始蓄水。", "乌东德水电站开始蓄水。", "auto", True, None),
        ("fuzzy_positive", "鬼灵们是入口。", "鬼灵门是入口。", "auto", True, None),
        ("context_negative", "外出务工人员回乡。", "外出务工人员回乡。", "auto", True, None),
        ("context_negative_low_confidence", "外出务工人员回乡。", "外出务工人员回乡。", "auto", True, 0.01),
        ("shadow_negative", "外出务工人员回乡。", "外出务工人员回乡。", "shadow", True, None),
        ("off_still_normalizes_exact_alias", "乌冬得水电站开始蓄水。", "乌东德水电站开始蓄水。", "off", True, None),
        ("partial_contract", "鬼灵们是入口。", "鬼灵们是入口。", "auto", False, None),
    ]
    for name, text, expected, mode, allow_auto, confidence in cases:
        matcher = EntityCandidateMatcher(definitions, mode=mode) if mode != "off" else None
        prepared = prepare_entity_segment(text,protector,matcher,allow_auto=allow_auto,confidence=confidence)
        finalized = finalize_entity_segment(prepared.protection.masked_text,prepared,protector)
        damaged = finalize_entity_segment("丢失受保护的实体。",prepared,protector)
        records.append(dict(name=name,kind="controlled_text_not_audio",source=text,expected=expected,
            actual=finalized.text,correct=finalized.text==expected,mode=mode,allow_auto=allow_auto,
            confidence=confidence,changes=list(prepared.normalizations),
            candidates=[m.public_dict() for m in prepared.report.matches],
            damaged_refiner_fallback=damaged.text,damaged_refiner_reasons=list(damaged.reject_reasons)))
    # A small dedicated store exercises the public dictionary API without
    # mutating the user's database. Existing pytest covers the HTTP endpoints.
    control = EntityStore(args.output/"control_crud.db")
    entity = control.upsert_entity("北京航空航天大学",entity_type="ORG",domain="education",
        normalization_policy="normalize",aliases=("北航",))
    p = EntityProtector(control.list_entities(domain="education"),protect_numeric_spans=False)
    alias = prepare_entity_segment("北航开展实验。",p,None,allow_auto=True)
    general_names = [e.canonical_text for e in control.list_entities(domain="general")]
    updated = control.update_entity(entity.entity_id,entity.canonical_text,entity_type="ORG",domain="education",
        normalization_policy="preserve",aliases=("北航",))
    kept = prepare_entity_segment("北航开展实验。",EntityProtector((updated,)),None,allow_auto=True)
    control.set_enabled_by_id(entity.entity_id,enabled=False)
    disabled_count = len(control.list_entities(domain="education"))
    deleted = control.delete_entity(entity.entity_id)
    write_json(args.output/"text_diagnostics.json",dict(cases=records,crud=dict(
        normalized_alias=alias.baseline_text,preserve_alias=kept.baseline_text,
        general_domain_names=general_names,disabled_active_count=disabled_count,deleted=deleted),
        note="Constructed text diagnostics are excluded from all audio CER/recall tables."))
    for record in records:
        print(record["name"],record["source"],"=>",record["actual"],"correct",record["correct"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "asr", "match", "diagnostics"))
    parser.add_argument("--output", type=Path, default=ROOT/"results/entity_library_20261003")
    parser.add_argument("--corpus", type=Path, default=Path("/home/aim0/data/datasets/ASR/aishell1/data/AISHELL-1/data_aishell"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--live-db", type=Path, default=ROOT.parent/"AgenticASR_entity_e5bedb3/data/entities.db")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--vocabulary-size", type=int, default=500)
    parser.add_argument("--noise-snr", type=float, default=10.0)
    parser.add_argument("--asr-url", default="http://127.0.0.1:8766")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.annotations = args.annotations or args.output/"source/aishell_ner_transcript.txt"
    {"prepare":prepare, "asr":decode, "match":match, "diagnostics":diagnostics}[args.phase](args)


if __name__ == "__main__":
    main()
