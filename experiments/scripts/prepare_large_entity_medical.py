#!/usr/bin/env python3
"""Prepare a CMeEE-grounded medical entity set from public audio references.

This script only writes the medical staging directory.  It does not invoke ASR
or alter any production dictionary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CV_ROOT = Path("/home/aim0/data/datasets/ASR/commonvoice_zh-CN/cv-corpus-26.0-2026-06-12/zh-CN")
WENET_JSON = Path("/home/aim0/data/datasets/ASR/wenetspeech_test_net/wenetspeech_test_net.json")
CME_FILE = ROOT / "results/medical_entity_20261003/source/CMeEE_train.json"
FAQ_MANIFEST = ROOT / "results/chinese_er_faq_entity_eval_20261010/manifest.json"
OLD_CV_MANIFESTS = (
    ROOT / "results/generic_entity_blind_20261010/metadata.json",
    ROOT / "results/generic_entity_optimization_20261010/holdout/metadata.json",
)
OUT = ROOT / "results/large_entity_benchmark_v1_20261010/staging/medical"
HAN = re.compile(r"^[\u4e00-\u9fff]+$")

# CMeEE includes descriptive fragments and broad labels.  These are rejected
# even when tagged because they are poor standard-form ASR targets.
REJECT = set("""
医生 患者 检查 治疗 手术 细胞 组织 疾病 症状 表现 功能 药物 人体
不活跃 体积较小 先天性 大小不等 无反应 有光泽 此外还有 某些病毒 混合型
窒息死亡 表面光滑 衍生物 粒细胞性 神经系 传感器 橄榄油 蒲公英 遗传学
病理科 肿瘤科 综合征 微生物 寄生虫 生殖器 免疫系统 蛋白质 容易发 冒金星
压力降低 全身发抖 四肢无力 小心脏 打自己 扮鬼脸 控设备 擦眼睛 眼前发黑
积极治疗 脸色灰 表情呆滞 体重增加 肢端肥大 手术治疗 碱性溶液 无机盐
氨基酸 血氧水平 血液循环
""".split())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cmeee_terms():
    categories: dict[str, set[str]] = defaultdict(set)
    for row in read_json(CME_FILE):
        for entity in row.get("entities", []):
            word = entity.get("entity", "")
            if 3 <= len(word) <= 24 and HAN.fullmatch(word) and word not in REJECT:
                categories[word].add(entity["type"])
    return categories


def build_trie(terms):
    root: dict = {}
    for word in terms:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node[None] = word
    return root


def longest_nonoverlap(text: str, trie: dict):
    """Left-to-right matching, choosing the longest term at each start."""
    matches = []
    i = 0
    while i < len(text):
        node = trie
        j = i
        best = None
        best_end = i
        while j < len(text) and text[j] in node:
            node = node[text[j]]
            j += 1
            if None in node:
                best, best_end = node[None], j
        if best is None:
            i += 1
        else:
            matches.append({"start": i, "end": best_end, "canonical": best})
            i = best_end
    return matches


def prior_cv_exclusions():
    sentence_ids, references = set(), set()
    for path in OLD_CV_MANIFESTS:
        if not path.exists():
            continue
        for sample in read_json(path).get("samples", []):
            if sample.get("source") != "commonvoice_zh-CN_test":
                continue
            loc = sample.get("location", {})
            if loc.get("sentence_id"):
                sentence_ids.add(loc["sentence_id"])
            if sample.get("reference"):
                references.add(sample["reference"])
    return sentence_ids, references


def cv_rows(sentence_ids, references, trie):
    test_tsv = CV_ROOT / "test.tsv"
    durations_tsv = CV_ROOT / "clip_durations.tsv"
    with durations_tsv.open(encoding="utf-8", newline="") as stream:
        durations = {r["clip"]: float(r["duration[ms]"]) / 1000 for r in csv.DictReader(stream, delimiter="\t")}
    out = []
    seen_references = set()
    with test_tsv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            ref = row["sentence"]
            sid = row["sentence_id"]
            path = CV_ROOT / "clips" / row["path"]
            duration = durations.get(row["path"], 0.0)
            if sid in sentence_ids or ref in references or ref in seen_references or duration < 3:
                continue
            if not path.is_file():
                continue
            spans = longest_nonoverlap(ref, trie)
            if spans:
                seen_references.add(ref)
                out.append({
                    "source": "commonvoice_zh-CN_test",
                    "recording_id": sid,
                    "sample_id": "medical_cv_" + sid,
                    "audio_path": str(path.resolve()),
                    "reference": ref,
                    "duration_seconds": duration,
                    "audio_accessible": True,
                    "spans": spans,
                    "speaker_id": row.get("client_id", ""),
                    "sentence_domain": row.get("sentence_domain", ""),
                })
    return out


def wenet_rows(trie):
    out = []
    with WENET_JSON.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            duration = float(row["duration"])
            path = Path(row["audio_filepath"])
            ref = row["text"]
            if duration < 3 or not path.is_file():
                continue
            spans = longest_nonoverlap(ref, trie)
            if spans:
                rid = path.stem
                out.append({
                    "source": "wenetspeech_TEST_NET",
                    "recording_id": rid,
                    "sample_id": "medical_wenet_" + rid,
                    "audio_path": str(path.resolve()),
                    "reference": ref,
                    "duration_seconds": duration,
                    "audio_accessible": True,
                    "spans": spans,
                    "lang": row.get("lang", ""),
                })
    return out


def faq_regression_rows(trie):
    if not FAQ_MANIFEST.exists():
        return []
    out = []
    for row in read_json(FAQ_MANIFEST).get("samples", []):
        duration = float(row.get("duration_seconds", 0))
        path = Path(row["audio_path"])
        if duration < 3 or not path.is_file():
            continue
        ref = row["reference"]
        spans = longest_nonoverlap(ref, trie)
        out.append({
            "source": "medical_FAQ_seen_regression",
            "recording_id": row["id"],
            "sample_id": row["id"],
            "audio_path": str(path.resolve()),
            "reference": ref,
            "duration_seconds": duration,
            "audio_accessible": True,
            "spans": spans,
            "cohort": "seen_regression",
        })
    return out


def choose_main_samples(candidates):
    """Choose one deterministic reference per term, packing terms per clip."""
    by_term: dict[str, list[dict]] = defaultdict(list)
    for sample in candidates:
        for span in sample["spans"]:
            by_term[span["canonical"]].append(sample)
    unassigned = set(by_term)
    chosen: dict[str, dict] = {}
    while unassigned:
        best_sample = None
        best_new = set()
        best_key = None
        # Prefer CV references where coverage is tied; Wenet supplements terms
        # absent from CV.  Stable IDs make reruns reproducible.
        for sample in candidates:
            new = {s["canonical"] for s in sample["spans"]} & unassigned
            if not new:
                continue
            priority = 0 if sample["source"] == "commonvoice_zh-CN_test" else 1
            key = (-len(new), priority, hashlib.sha256(sample["sample_id"].encode()).hexdigest())
            if best_key is None or key < best_key:
                best_key, best_sample, best_new = key, sample, new
        if best_sample is None:
            break
        chosen[best_sample["sample_id"]] = best_sample
        for term in best_new:
            unassigned.remove(term)
            by_term[term] = [best_sample]
    return list(chosen.values()), by_term, unassigned


def probe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main():
    out_dir = OUT.resolve()
    terms = cmeee_terms()
    trie = build_trie(terms)
    old_ids, old_refs = prior_cv_exclusions()
    cv = cv_rows(old_ids, old_refs, trie)
    wenet = wenet_rows(trie)
    all_candidates = cv + wenet
    selected, evidence, unassigned = choose_main_samples(all_candidates)
    if unassigned:
        raise RuntimeError(f"Could not assign audio reference to {len(unassigned)} terms")

    # Keep only spans actually represented by the chosen samples, and ensure
    # every selected sample remains a whole reference segment.
    # The chosen set is based on one occurrence per term; report the definitive
    # terms from those occurrences rather than any incidental overlap.
    main_terms = set(evidence)
    sample_by_id = {s["sample_id"]: s for s in selected}

    for sample in selected:
        if not Path(sample["audio_path"]).is_file():
            raise RuntimeError(f"Selected audio missing: {sample['audio_path']}")
        sample["sampled_duration_seconds"] = (
            probe_duration(sample["audio_path"])
            if sample["source"] == "commonvoice_zh-CN_test"
            else sample["duration_seconds"]
        )
        sample["audio_accessible"] = sample["sampled_duration_seconds"] >= 3
        if not sample["audio_accessible"]:
            raise RuntimeError(f"Selected reference shorter than 3 seconds: {sample['sample_id']}")
        if sample["source"] == "wenetspeech_TEST_NET":
            sample["audio_sha256"] = digest_file(Path(sample["audio_path"]))
        sample["domain"] = "medical"
        sample["cohort"] = "new_main"
        # Drop CV speaker metadata so the staging set does not duplicate personal IDs.
        sample.pop("speaker_id", None)
        sample.pop("sentence_domain", None)
        sample.pop("lang", None)

    entities = []
    for canonical in sorted(main_terms):
        source_samples = evidence[canonical]
        hits = []
        for sample in source_samples:
            span = next(s for s in sample["spans"] if s["canonical"] == canonical)
            hits.append({"sample_id": sample["sample_id"], **span})
        entities.append({
            "canonical": canonical,
            "type": "TERM",
            "domain": "medical",
            "source": "CMeEE_train.json",
            "source_categories": sorted(terms[canonical]),
            "evidence": hits,
        })

    faq = faq_regression_rows(trie)
    # Recheck FAQ audio paths and retain the full reference.  Seen rows never
    # contribute to the new-main entity vocabulary or coverage total.
    for sample in faq:
        sample["sampled_duration_seconds"] = sample["duration_seconds"]
        if not sample["audio_accessible"]:
            raise RuntimeError(f"FAQ regression audio missing: {sample['sample_id']}")
        sample["entities"] = sample.pop("spans")
    for sample in selected:
        sample["entities"] = sample.pop("spans")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in selected + faq:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    (out_dir / "entities.json").write_text(json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stats = {
        "dataset": "large_entity_benchmark_v1_20261010 medical staging",
        "primary_target_reached": 100 <= len(entities) <= 200,
        "main_unique_entity_count": len(entities),
        "main_sample_count": len(selected),
        "main_samples_by_source": {
            source: sum(s["source"] == source for s in selected)
            for source in ("commonvoice_zh-CN_test", "wenetspeech_TEST_NET")
        },
        "main_occurrences": sum(len(s["entities"]) for s in selected),
        "seen_regression_sample_count": len(faq),
        "seen_regression_entity_spans": sum(len(s["entities"]) for s in faq),
        "seen_regression_excluded_from_main_count": True,
        "cmeee_candidate_count_after_length_script_filter_and_generic_rejects": len(terms),
        "cv_prior_excluded_sentence_id_count": len(old_ids),
        "cv_prior_excluded_reference_count": len(old_refs),
        "matching": "Exact CMeEE standard-form substring matches; left-to-right longest non-overlapping; half-open Python character offsets.",
        "reference_status": "Matched against supplied standard references, not native speech NER labels; no ASR output used to select terms.",
        "audio_review": "Selected file paths checked; Common Voice durations checked with ffprobe; selected WenetSpeech clips SHA-256 hashed. No per-item human listening audit performed.",
        "medical_faq_status": "Prior emergency-room FAQ retained only as seen_regression cohort; never counted toward new-main entity coverage.",
        "known_limitations": [
            "Common Voice and WenetSpeech reference matches establish textual evidence, not independently audited acoustic realization.",
            "CMeEE category labels are preserved as source provenance; benchmark entities are uniformly exported as TERM.",
            "MultiMed is excluded from the main set due to incomplete long-reference/audio-range concerns.",
        ],
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: stats[k] for k in (
        "main_unique_entity_count", "main_sample_count", "main_samples_by_source",
        "seen_regression_sample_count", "primary_target_reached")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
