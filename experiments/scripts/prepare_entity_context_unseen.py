#!/usr/bin/env python3
"""Freeze a fresh, source-auditable entity confirmation set before ASR.

Selection is deterministic and uses only audio metadata, official references,
and the existing 1,346-entry entity library. References and literal annotations
are kept separate from the audio/source review input list.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from pypinyin import lazy_pinyin

from prepare_large_entity_aishell import historical_ids

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/large_entity_benchmark_v1_20261010"
OUT = ROOT / "results/entity_context_confirmation_20261012/unseen"
CV_ROOT = Path("/home/aim0/data/datasets/ASR/commonvoice_zh-CN/cv-corpus-26.0-2026-06-12/zh-CN")
SEED = "entity-context-unseen-20261012-v1"
OLD_BATCHES = ("holdout", "holdout_news", "holdout_v2")
TOPIC_TARGET = 30
POSITIVE_MAX = 90
LIBRARY_OUT_MAX = 35
NEGATIVE_TARGET = 50


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compact(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and unicodedata.category(c)[0] not in {"P", "C"}).casefold()


def token_hits(text: str, terms: list[str]) -> list[dict]:
    """Longest non-overlapping literal matching against fixed canonicals."""
    trie: dict = {}
    for term in terms:
        node = trie
        for char in term:
            node = node.setdefault(char, {})
        node[None] = term
    found, i = [], 0
    while i < len(text):
        node, j, best = trie, i, None
        while j < len(text) and text[j] in node:
            node = node[text[j]]
            j += 1
            if None in node:
                best = (node[None], j)
        if best:
            term, end = best
            found.append({"canonical": term, "entity_type": "TERM", "start": i,
                          "end": end, "annotation_method": "reference_longest_surface_match"})
            i = end
        else:
            i += 1
    return found


def levenshtein_one(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] == 1


def near_library_forms(surface: str, pinyin_by_term: dict[str, tuple[str, ...]]) -> list[dict]:
    if len(surface) < 2:
        return []
    phones = tuple(lazy_pinyin(surface))
    output = []
    for canonical, target in pinyin_by_term.items():
        if levenshtein_one(phones, target):
            output.append({"canonical": canonical, "pinyin_edit_distance": 1})
            if len(output) == 5:
                break
    return output


def old_cv_exclusions() -> tuple[set[str], set[str]]:
    ids, refs = set(), set()
    for name in ("generic_entity_blind_20261010/metadata.json",
                 "generic_entity_optimization_20261010/holdout/metadata.json"):
        payload = json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        for row in payload["samples"]:
            if row.get("source", "").startswith("commonvoice"):
                sid = row.get("location", {}).get("sentence_id")
                if sid:
                    ids.add(sid)
                if row.get("reference"):
                    refs.add(compact(row["reference"]))
    return ids, refs


def read_old_sets() -> dict:
    large = [json.loads(line) for line in (BASE / "dataset/samples.jsonl").read_text(encoding="utf-8").splitlines()]
    old = [json.loads(line) for batch in OLD_BATCHES
           for line in (ROOT / f"results/entity_confirmation_20261011/{batch}/samples.jsonl").read_text(encoding="utf-8").splitlines()]
    all_rows = large + old
    paths, ids, audio_hashes, pcm_hashes, refs = set(), set(), set(), set(), set()
    for row in all_rows:
        if row.get("audio_path"):
            paths.add(str(Path(row["audio_path"]).resolve()))
        for key in ("id", "source_id", "recording_id"):
            if row.get(key):
                ids.add(row[key])
        for key in ("audio_sha256", "source_audio_sha256"):
            if row.get(key):
                audio_hashes.add(row[key])
        if row.get("pcm_sha256"):
            pcm_hashes.add(row["pcm_sha256"])
        if row.get("reference"):
            refs.add(compact(row["reference"]))
        for source in row.get("source_records", []):
            for key in ("source_id", "recording_id", "sample_id"):
                if source.get(key):
                    ids.add(source[key])
            if source.get("source_path"):
                paths.add(str(Path(source["source_path"]).resolve()))
    return {"paths": paths, "ids": ids, "audio_hashes": audio_hashes,
            "pcm_hashes": pcm_hashes, "refs": refs, "large_count": len(large),
            "old_count": len(old), "old_rows": old,
            "large_manifest_sha256": sha(BASE / "dataset/manifest.json"),
            "old_manifest_sha256": {b: sha(ROOT / f"results/entity_confirmation_20261011/{b}/manifest.json") for b in OLD_BATCHES}}


def order(row: dict, cohort: str) -> str:
    return hashlib.sha256((SEED + "\0" + cohort + "\0" + row["source_id"]).encode()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen unseen set: {OUT}")
    entities = json.loads((BASE / "dataset/entities.json").read_text(encoding="utf-8"))
    by_canonical = {row["canonical_text"]: row for row in entities}
    terms = sorted(by_canonical, key=lambda x: (-len(x), x))
    pinyin_by_term = {term: tuple(lazy_pinyin(term)) for term in terms}
    exclusions = read_old_sets()
    cv_old_ids, cv_old_refs = old_cv_exclusions()
    aishell_historical = historical_ids()
    pools: dict[str, list[dict]] = defaultdict(list)

    # CV references provide the broadest topic mix. No ASR output is read.
    durations_path = CV_ROOT / "clip_durations.tsv"
    with durations_path.open(encoding="utf-8", newline="") as stream:
        durations = {r["clip"]: float(r["duration[ms]"]) / 1000 for r in csv.DictReader(stream, delimiter="\t")}
    with (CV_ROOT / "test.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            sid, original = row["sentence_id"], row["sentence"].strip()
            audio = (CV_ROOT / "clips" / row["path"]).resolve()
            ref = compact(original)
            duration = durations.get(row["path"], 0.0)
            if (sid in cv_old_ids or ref in cv_old_refs or sid in exclusions["ids"] or
                    str(audio) in exclusions["paths"] or ref in exclusions["refs"] or
                    not audio.is_file() or not 3 <= duration <= 20):
                continue
            gold = token_hits(ref, terms)
            if not gold:
                pools["negative"].append({"source": "Common Voice zh-CN test", "source_id": sid,
                    "audio_path": str(audio), "reference": ref, "reference_original": original,
                    "duration_seconds": duration, "gold_entities": [], "topics": ["negative_control"],
                    "cohort": "negative"})
                continue
            topics = sorted({topic for item in gold for topic in by_canonical[item["canonical"]].get("topics", [])})
            # The fixed library's dominant news topic is intentionally omitted for topic-balanced CV selection.
            topics = [t for t in topics if t != "新闻人物组织地理"]
            if not topics:
                continue
            pools["positive"].append({"source": "Common Voice zh-CN test", "source_id": sid,
                "audio_path": str(audio), "reference": ref, "reference_original": original,
                "duration_seconds": duration, "gold_entities": gold, "topics": topics,
                "cohort": "positive"})

    # AISHELL official NER supplies out-of-library entity forms and ordinary controls.
    staging = BASE / "staging/aishell/samples.jsonl"
    for line in staging.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sid, audio = row["id"], Path(row["audio_path"]).resolve()
        ref = compact(row["reference"])
        if (sid in aishell_historical or sid in exclusions["ids"] or str(audio) in exclusions["paths"] or
                ref in exclusions["refs"] or row["audio_sha256"] in exclusions["audio_hashes"] or
                row["pcm_sha256"] in exclusions["pcm_hashes"] or not 3 <= row["duration_seconds"] <= 20):
            continue
        source_entities = row.get("entities", [])
        known, unknown = [], []
        for ent in source_entities:
            canonical = ent["text"]
            start = len(compact(row["reference"][:ent["start"]]))
            end = len(compact(row["reference"][:ent["end"]]))
            item = {"canonical": canonical, "entity_type": ent["type"], "start": start, "end": end,
                    "annotation_method": "official_aishell_ner"}
            (known if canonical in by_canonical else unknown).append(item)
        if known:
            topics = sorted({t for e in known for t in by_canonical[e["canonical"]].get("topics", [])})
            pools["positive"].append({"source": "AISHELL-1 test", "source_id": sid,
                "audio_path": str(audio), "reference": ref, "reference_original": row["reference"],
                "duration_seconds": row["duration_seconds"], "gold_entities": known,
                "topics": topics or ["新闻人物组织地理"], "cohort": "positive"})
        elif unknown:
            near = []
            for entity in unknown:
                surface = entity["canonical"]
                matches = near_library_forms(surface, pinyin_by_term)
                if matches:
                    near.append({"surface": surface, "span": [entity["start"], entity["end"]],
                                 "nearest_library_forms": matches})
            cohort = "near_sound" if near else "library_out"
            pools[cohort].append({"source": "AISHELL-1 test", "source_id": sid,
                "audio_path": str(audio), "reference": ref, "reference_original": row["reference"],
                "duration_seconds": row["duration_seconds"], "gold_entities": unknown,
                "topics": ["库外实体"] + (["近音词候选"] if near else []),
                "near_sound_candidates": near, "cohort": cohort})
        else:
            pools["negative"].append({"source": "AISHELL-1 test", "source_id": sid,
                "audio_path": str(audio), "reference": ref, "reference_original": row["reference"],
                "duration_seconds": row["duration_seconds"], "gold_entities": [],
                "topics": ["新闻普通负例"], "cohort": "negative"})

    # Deduplicate candidates by source, normalized reference and physical path.
    for key in list(pools):
        dedup, seen = [], set()
        for row in pools[key]:
            k = (str(Path(row["audio_path"]).resolve()), row["reference"])
            if k not in seen:
                seen.add(k)
                dedup.append(row)
        pools[key] = dedup

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pcm").mkdir(parents=True, exist_ok=True)

    selected, seen_paths, seen_refs = [], set(exclusions["paths"]), set(exclusions["refs"])
    seen_audio, seen_pcm = set(exclusions["audio_hashes"]), set(exclusions["pcm_hashes"])
    used_source_ids = set()

    def accept(row: dict) -> bool:
        path = Path(row["audio_path"])
        if row["source_id"] in used_source_ids or str(path.resolve()) in seen_paths or row["reference"] in seen_refs:
            return False
        audio_hash = sha(path)
        if audio_hash in seen_audio:
            return False
        pcm_path = OUT / "pcm" / ("ecu_" + hashlib.sha256((SEED + audio_hash).encode()).hexdigest()[:20] + ".f32le")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-map", "0:a:0",
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", "-f", "f32le", str(pcm_path)], check=True)
        pcm_hash = sha(pcm_path)
        actual_duration = pcm_path.stat().st_size / 64000
        if pcm_hash in seen_pcm or not 3 <= actual_duration <= 20:
            pcm_path.unlink(missing_ok=True)
            return False
        for ent in row["gold_entities"]:
            if ent["canonical"] in by_canonical and row["reference"][ent["start"]:ent["end"]] != ent["canonical"]:
                pcm_path.unlink(missing_ok=True)
                raise ValueError(f"Reference span mismatch: {row['source_id']} {ent}")
        row.update(id=pcm_path.stem, audio_sha256=audio_hash, pcm_sha256=pcm_hash,
                   pcm_path=str(pcm_path.resolve()), duration_seconds=actual_duration,
                   sample_rate=16000, channels=1, sample_format="f32le")
        selected.append(row)
        seen_paths.add(str(path.resolve())); seen_refs.add(row["reference"])
        seen_audio.add(audio_hash); seen_pcm.add(pcm_hash); used_source_ids.add(row["source_id"])
        return True

    # Diverse positives: first cover each available non-news topic, then fill to cap.
    topic_order = sorted({t for row in pools["positive"] for t in row["topics"] if t != "新闻人物组织地理"})
    positive_order = sorted(pools["positive"], key=lambda r: order(r, "positive"))
    for topic in topic_order:
        count = 0
        for row in positive_order:
            if topic in row["topics"] and row not in selected and count < TOPIC_TARGET:
                if accept(row):
                    count += 1
    for row in positive_order:
        if sum(r["cohort"] == "positive" for r in selected) >= POSITIVE_MAX:
            break
        if row not in selected:
            accept(row)

    # Prefer near-sound candidates, then other library-out entity evidence.
    for cohort, quota in (("near_sound", 20), ("library_out", LIBRARY_OUT_MAX)):
        count = 0
        for row in sorted(pools[cohort], key=lambda r: order(r, cohort)):
            if count >= quota:
                break
            if accept(row):
                count += 1
    negative_count = 0
    for row in sorted(pools["negative"], key=lambda r: order(r, "negative")):
        if negative_count >= NEGATIVE_TARGET:
            break
        if accept(row):
            negative_count += 1

    OUT.mkdir(parents=True, exist_ok=True)
    # PCM files are created during selection, so initialize the output directory first.
    # (The directory is created before candidate decoding below by the caller's setup.)
    summary = {
        "seed": SEED, "pre_ASR_selection": True, "fixed_library_size": len(entities),
        "library_sha256": sha(BASE / "dataset/entities.json"),
        "source_pool_counts": {k: len(v) for k, v in sorted(pools.items())},
        "available_positive_topics": dict(Counter(t for row in pools["positive"] for t in row["topics"])),
        "accepted_by_cohort": dict(Counter(r["cohort"] for r in selected)),
        "accepted_by_topic": dict(Counter(t for r in selected for t in r["topics"])),
        "samples": len(selected),
        "dedup_exclusions": {"large_benchmark_samples": exclusions["large_count"],
            "prior_20261011_samples": exclusions["old_count"], "aishell_historical_ids": len(aishell_historical),
            "cv112_source_ids_and_references": {"ids": len(cv_old_ids), "references": len(cv_old_refs)}},
        "limitations": ["Gold spans are literal reference matches or official AISHELL-NER; they do not establish semantic NER or audio correctness.",
            "Near-sound labels are candidate-generation hints based on tone-insensitive pinyin edit distance one; they are not aliases.",
            "The audio and references remain from the same public source corpora; this does not establish speaker or model-pretraining independence."],
    }
    # The review input list deliberately contains no transcript or entity labels.
    review_inputs = [{k: row[k] for k in ("id", "source", "source_id", "audio_path", "pcm_path",
                                           "audio_sha256", "pcm_sha256", "duration_seconds",
                                           "sample_rate", "channels", "sample_format")} for row in selected]
    sample_index = [{k: v for k, v in row.items() if k not in ("reference", "reference_original", "gold_entities", "near_sound_candidates")} for row in selected]
    references = [{"id": row["id"], "reference": row["reference"],
                   "reference_original": row["reference_original"], "gold_entities": row["gold_entities"],
                   **({"near_sound_candidates": row["near_sound_candidates"]} if row.get("near_sound_candidates") else {})}
                  for row in selected]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review_inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review_inputs), encoding="utf-8")
    (OUT / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample_index), encoding="utf-8")
    (OUT / "references.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in references), encoding="utf-8")
    summary.update({"samples_sha256": sha(OUT / "samples.jsonl"),
                    "review_inputs_sha256": sha(OUT / "review_inputs.jsonl"),
                    "references_sha256": sha(OUT / "references.jsonl"),
                    "pcm_sha256": {r["id"]: r["pcm_sha256"] for r in selected}})
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"frozen": True, "selection": summary,
        "large_benchmark_manifest_sha256": exclusions["large_manifest_sha256"],
        "prior_20261011_manifest_sha256": exclusions["old_manifest_sha256"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "READY").write_text("Frozen before ASR; reference and literal gold are stored separately.\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
