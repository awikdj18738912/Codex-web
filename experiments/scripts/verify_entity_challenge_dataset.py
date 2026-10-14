"""Independent offline verifier for the frozen entity challenge dataset.

The verifier reads the dataset, its independent SQLite libraries, and only the
PCM hash map from the protected holdout summary. It never opens protected audio
or references and never contacts an ASR service.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import unicodedata

import numpy as np
from opencc import OpenCC
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "results/entity_challenge_20261014/dataset"
DEFAULT_PROTECTED_SUMMARY = ROOT / "results/entity_context_confirmation_20261012/unseen/summary.json"
EXPECTED_RECORDINGS = 2000
MIN_DOMAIN_RECORDINGS = 30
MIN_RECALL = 0.495
MAX_RECALL = 0.505
RATE = 16000
BYTES_PER_SAMPLE = 4


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_ieee_float32_wav(path: Path) -> tuple[dict, bytes]:
    """Read RIFF/WAVE chunks and return format metadata plus raw float32 data."""
    blob = path.read_bytes()
    if len(blob) < 12 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    declared_size = struct.unpack_from("<I", blob, 4)[0] + 8
    if declared_size != len(blob):
        raise ValueError(f"RIFF size {declared_size} != file size {len(blob)}")

    fmt = None
    data_chunks = []
    offset = 12
    while offset + 8 <= len(blob):
        chunk_id, chunk_size = struct.unpack_from("<4sI", blob, offset)
        start = offset + 8
        end = start + chunk_size
        if end > len(blob):
            raise ValueError(f"truncated {chunk_id!r} chunk")
        chunk = blob[start:end]
        if chunk_id == b"fmt ":
            if len(chunk) < 16:
                raise ValueError("short fmt chunk")
            fmt = struct.unpack_from("<HHIIHH", chunk, 0)
        elif chunk_id == b"data":
            data_chunks.append(chunk)
        offset = end + (chunk_size & 1)
    if offset != len(blob):
        raise ValueError("malformed RIFF chunk padding")
    if fmt is None or not data_chunks:
        raise ValueError("missing fmt or data chunk")
    if len(data_chunks) != 1:
        raise ValueError(f"expected one data chunk, found {len(data_chunks)}")
    code, channels, rate, byte_rate, block_align, bits = fmt
    metadata = dict(format_code=code, channels=channels, sample_rate=rate,
                    byte_rate=byte_rate, block_align=block_align, bits_per_sample=bits)
    if (code, channels, rate, byte_rate, block_align, bits) != (3, 1, RATE, RATE * 4, 4, 32):
        raise ValueError(f"expected mono 16 kHz IEEE_FLOAT32 WAV, got {metadata}")
    payload = data_chunks[0]
    if len(payload) % 4:
        raise ValueError("WAV float32 payload length is not divisible by four")
    return metadata, payload


def compact_whitespace(value: str) -> str:
    return "".join(c for c in value if not c.isspace())


def find_leaks(value, location="") -> list[str]:
    forbidden = {
        "reference", "reference_original", "gold", "gold_entities", "entities",
        "canonical", "canonical_key", "canonical_text", "surface", "text",
        "raw_text", "transcript", "hypothesis", "asr_payload", "asr_output",
        "glossary", "vocabulary", "vocabulary_terms", "entity_names", "entity_context",
        "context_text", "context_terms", "lexicon", "dictionary",
    }
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_s = str(key)
            lower = key_s.casefold()
            if (lower in forbidden or lower.startswith(("reference_", "gold_", "canonical_", "surface_"))):
                found.append(f"{location}.{key_s}" if location else key_s)
            found.extend(find_leaks(child, f"{location}.{key_s}" if location else key_s))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_leaks(child, f"{location}[{index}]"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"dataset directory (default: {DEFAULT_DATASET})")
    parser.add_argument("--protected-summary", type=Path, default=DEFAULT_PROTECTED_SUMMARY,
                        help="read only this file's pcm_sha256 map")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    errors: list[dict] = []
    checks: dict[str, dict] = {}

    def error(code: str, message: str, sample_id: str | None = None):
        item = {"code": code, "message": message}
        if sample_id is not None:
            item["sample_id"] = sample_id
        errors.append(item)

    def check(name: str, passed: bool, details=None):
        checks[name] = {"passed": bool(passed), "details": details or {}}
        if not passed:
            error("check_failed", name)

    report_path = dataset / "verification.json"
    if not dataset.is_dir():
        print(json.dumps({"status": "FAIL", "dataset": str(dataset),
                          "errors": [{"code": "missing_dataset", "message": "dataset directory does not exist"}]},
                         ensure_ascii=False, indent=2))
        return 2

    required = ["manifest.json", "summary.json", "inference_inputs.jsonl", "references.json",
                "baseline_raw.jsonl", "samples.jsonl"]
    for name in required:
        if not (dataset / name).is_file():
            error("missing_file", f"missing {name}")

    manifest = summary = refs_doc = None
    inputs = baselines = samples = []
    for name, path in (("manifest", dataset / "manifest.json"),
                       ("summary", dataset / "summary.json"),
                       ("references", dataset / "references.json"),
                       ("inference_inputs", dataset / "inference_inputs.jsonl"),
                       ("baseline_raw", dataset / "baseline_raw.jsonl"),
                       ("samples", dataset / "samples.jsonl")):
        if not path.is_file():
            continue
        try:
            value = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
            if name == "manifest": manifest = value
            elif name == "summary": summary = value
            elif name == "references": refs_doc = value
            elif name == "inference_inputs": inputs = value
            elif name == "baseline_raw": baselines = value
            elif name == "samples": samples = value
        except Exception as exc:
            error("read_error", f"{name}: {type(exc).__name__}: {exc}")

    labels = refs_doc.get("samples", {}) if isinstance(refs_doc, dict) else {}
    input_by_id = {}
    sample_by_id = {}
    baseline_by_id = {}
    for field, rows, dest in (("inference_inputs", inputs, input_by_id),
                              ("samples", samples, sample_by_id),
                              ("baseline_raw", baselines, baseline_by_id)):
        for row in rows:
            sid = row.get("id")
            if not sid:
                error("missing_id", f"{field} row has no id")
            elif sid in dest:
                error("duplicate_id", f"duplicate id in {field}", sid)
            else:
                dest[sid] = row
    ids = set(input_by_id)
    labels_ids = set(labels) if isinstance(labels, dict) else set()
    sample_ids = set(sample_by_id)
    baseline_ids = set(baseline_by_id)
    check("recording_count", len(inputs) == EXPECTED_RECORDINGS,
          {"actual": len(inputs), "expected": EXPECTED_RECORDINGS})
    check("sidecar_ids_match", ids == labels_ids == sample_ids == baseline_ids,
          {"inference": len(ids), "references": len(labels_ids), "samples": len(sample_ids),
           "baseline": len(baseline_ids), "missing_reference": sorted(ids - labels_ids)[:10],
           "extra_reference": sorted(labels_ids - ids)[:10]})
    sidecar_mismatches = []
    for sid in sorted(ids & labels_ids & sample_ids & baseline_ids):
        input_row = input_by_id[sid]
        sample_row = sample_by_id[sid]
        baseline_row = baseline_by_id[sid]
        label = labels[sid]
        if sample_row.get("pcm_sha256") != input_row.get("pcm_sha256") or baseline_row.get("pcm_sha256") != input_row.get("pcm_sha256"):
            sidecar_mismatches.append({"id": sid, "problem": "pcm_sha256_disagrees_across_sidecars"})
        if sample_row.get("reference") != label.get("reference") or sample_row.get("gold_entities") != label.get("gold_entities"):
            sidecar_mismatches.append({"id": sid, "problem": "reference_or_gold_disagrees_across_sidecars"})
    check("sidecar_contents_match", not sidecar_mismatches,
          {"mismatches": sidecar_mismatches[:30], "mismatch_count": len(sidecar_mismatches)})

    pcm_hashes = [row.get("pcm_sha256") for row in inputs]
    check("unique_ids", len(ids) == len(inputs), {"unique": len(ids), "rows": len(inputs)})
    check("unique_pcm_sha256", None not in pcm_hashes and len(set(pcm_hashes)) == len(pcm_hashes),
          {"rows": len(pcm_hashes), "unique": len(set(pcm_hashes))})

    converter = OpenCC("t2s")
    try:
        from system.zh_itn_bridge import ZhITNNormalizer
        itn = ZhITNNormalizer()
    except Exception as exc:
        error("normalizer_init", f"{type(exc).__name__}: {exc}")
        itn = None

    def norm(text: str) -> str:
        value = itn.normalize(text).text if itn is not None else text
        value = unicodedata.normalize("NFKC", converter.convert(value)).casefold()
        return "".join(c for c in value if unicodedata.category(c)[0] not in {"P", "Z", "C"})

    normalized_refs = []
    if isinstance(labels, dict):
        normalized_refs = [norm(label.get("reference", "")) for label in labels.values()]
    check("unique_normalized_references",
          len(normalized_refs) == EXPECTED_RECORDINGS and len(set(normalized_refs)) == len(normalized_refs),
          {"rows": len(normalized_refs), "unique": len(set(normalized_refs))})

    domains = Counter(row.get("domain") for row in inputs)
    check("domain_coverage", len(domains) >= 15 and all(n >= MIN_DOMAIN_RECORDINGS for n in domains.values()),
          {"domain_count": len(domains), "recordings": dict(domains), "minimum_per_domain": MIN_DOMAIN_RECORDINGS})

    leaks = []
    for row in inputs:
        for path in find_leaks(row):
            leaks.append({"sample_id": row.get("id"), "field": path})
    check("inference_inputs_have_no_reference_or_gold_leaks", not leaks, {"leaks": leaks[:50], "count": len(leaks)})

    # Exact per-mention rescoring and span support against the frozen sidecar.
    total_hits = total_mentions = total_errors = total_reference_chars = 0
    per_domain = Counter()
    per_domain_hits = Counter()
    per_domain_mentions = Counter()
    span_errors = 0
    score_errors = 0
    for sid in sorted(ids & labels_ids & baseline_ids):
        label = labels[sid]
        baseline = baseline_by_id[sid]
        reference = label.get("reference", "")
        raw_text = baseline.get("raw_text", "")
        ref_norm = norm(reference)
        hyp_norm = norm(raw_text)
        gold_counts = Counter(norm(entity.get("canonical", "")) for entity in label.get("gold_entities", []))
        if any(not word or ref_norm.count(word) < count for word, count in gold_counts.items()):
            error("gold_not_supported_by_reference", "canonical mention multiplicity exceeds reference support", sid)
        hits = sum(min(count, hyp_norm.count(word)) for word, count in gold_counts.items())
        expected = sum(gold_counts.values())
        edit_distance = Levenshtein.distance(ref_norm, hyp_norm)
        ref_chars = len(ref_norm)
        total_hits += hits
        total_mentions += expected
        total_errors += edit_distance
        total_reference_chars += ref_chars
        domain = input_by_id[sid].get("domain", "")
        per_domain[domain] += 1
        per_domain_hits[domain] += hits
        per_domain_mentions[domain] += expected

        stored = baseline.get("score", {})
        stored_pairs = (("hits", hits), ("expected_mentions", expected), ("errors", edit_distance),
                        ("reference_chars", ref_chars))
        if any(stored.get(key) != value for key, value in stored_pairs):
            score_errors += 1
            error("baseline_score_mismatch", "stored baseline score differs from independent recomputation", sid)

        for entity in label.get("gold_entities", []):
            start, end = entity.get("start"), entity.get("end")
            canonical = entity.get("canonical", "")
            surface = entity.get("surface", entity.get("text", ""))
            if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(reference)):
                span_errors += 1
                error("invalid_gold_boundary", f"invalid [{start}, {end}) for reference length {len(reference)}", sid)
                continue
            span = reference[start:end]
            accepted = {norm(canonical)}
            if surface:
                accepted.add(norm(surface))
            if norm(span) not in accepted:
                span_errors += 1
                error("gold_boundary_unsupported", f"reference slice {span!r} supports neither canonical nor surface", sid)

    recall = total_hits / total_mentions if total_mentions else 0.0
    cer = total_errors / total_reference_chars if total_reference_chars else None
    check("recomputed_raw_entity_recall", total_mentions > 0 and MIN_RECALL <= recall <= MAX_RECALL,
          {"hits": total_hits, "expected_mentions": total_mentions, "recall": recall,
           "required_range": [MIN_RECALL, MAX_RECALL], "stored_score_mismatch_rows": score_errors})
    check("gold_reference_boundaries", span_errors == 0,
          {"mentions_checked": sum(len(labels[s].get("gold_entities", [])) for s in ids & labels_ids),
           "boundary_errors": span_errors})
    check("recomputed_full_transcript_cer", cer is not None,
          {"edit_distance": total_errors, "normalized_reference_chars": total_reference_chars, "cer": cer})
    if summary is not None:
        check("summary_matches_recomputed_scores",
              summary.get("recordings") == len(ids) and summary.get("baseline_hits") == total_hits and
              summary.get("expected_mentions") == total_mentions,
              {"summary_recordings": summary.get("recordings"), "actual_recordings": len(ids),
               "summary_hits": summary.get("baseline_hits"), "recomputed_hits": total_hits,
               "summary_mentions": summary.get("expected_mentions"), "recomputed_mentions": total_mentions})
        domain_mismatches = {}
        for domain, count in per_domain.items():
            saved = summary.get("domains", {}).get(domain, {})
            actual = {"recordings": count, "hits": per_domain_hits[domain],
                      "expected_mentions": per_domain_mentions[domain]}
            if any(saved.get(k) != v for k, v in actual.items()):
                domain_mismatches[domain] = {"saved": saved, "recomputed": actual}
        check("summary_domain_scores_match", not domain_mismatches, {"mismatches": domain_mismatches})

    # Read only summary.json and its pcm_sha256 map from the protected holdout.
    protected_hashes = set()
    if not args.protected_summary.is_file():
        error("protected_summary_missing", str(args.protected_summary))
    else:
        try:
            protected_doc = read_json(args.protected_summary)
            protected_map = protected_doc.get("pcm_sha256", {})
            protected_hashes = set(protected_map.values()) if isinstance(protected_map, dict) else set()
            protected_overlaps = sorted(set(pcm_hashes) & protected_hashes)
            check("protected_pcm_disjoint", len(protected_hashes) == 181 and not protected_overlaps,
                  {"protected_hashes_read": len(protected_hashes), "expected_protected_hashes": 181,
                   "overlap_count": len(protected_overlaps), "overlap_examples": protected_overlaps[:10],
                   "read_scope": "summary.json pcm_sha256 values only"})
        except Exception as exc:
            error("protected_summary_read_error", f"{type(exc).__name__}: {exc}")

    # Manifest covers every frozen dataset artifact except itself and this report.
    manifest_files = manifest.get("files_sha256", {}) if isinstance(manifest, dict) else {}
    hash_errors = []
    for relative, expected_hash in manifest_files.items():
        path = dataset / relative
        if not path.is_file():
            hash_errors.append({"path": relative, "problem": "missing"})
        else:
            actual_hash = digest(path)
            if actual_hash != expected_hash:
                hash_errors.append({"path": relative, "problem": "sha256_mismatch",
                                    "expected": expected_hash, "actual": actual_hash})
    actual_artifacts = {str(p.relative_to(dataset)) for p in dataset.rglob("*")
                        if p.is_file() and p.name not in {"manifest.json", "verification.json"}}
    unlisted = sorted(actual_artifacts - set(manifest_files))
    check("manifest_file_hashes", not hash_errors and not unlisted,
          {"listed_files": len(manifest_files), "hash_errors": hash_errors[:30],
           "unlisted_files": unlisted[:30], "unlisted_count": len(unlisted)})

    # Verify the generated WAV inventory, RIFF format, and sample payload bytes.
    wav_dir = dataset / "audio"
    wav_paths = sorted(wav_dir.glob("*.wav")) if wav_dir.is_dir() else []
    wav_errors = []
    for sid in sorted(ids):
        row = input_by_id.get(sid)
        if row is None:
            continue
        expected_wav = (wav_dir / f"{sid}.wav").resolve()
        declared = row.get("evaluation_wav_path")
        if not declared or Path(declared).resolve() != expected_wav:
            wav_errors.append({"id": sid, "problem": "path_mismatch", "declared": declared,
                               "expected": str(expected_wav)})
            continue
        pcm_path = Path(str(row.get("pcm_path") or ""))
        wav_path = expected_wav
        if not pcm_path.is_file() or not wav_path.is_file():
            wav_errors.append({"id": sid, "problem": "missing_pcm_or_wav"})
            continue
        try:
            pcm = pcm_path.read_bytes()
            pcm_sha = hashlib.sha256(pcm).hexdigest()
            if pcm_sha != row.get("pcm_sha256"):
                wav_errors.append({"id": sid, "problem": "pcm_sha256_mismatch"})
                continue
            if len(pcm) == 0 or len(pcm) % BYTES_PER_SAMPLE:
                wav_errors.append({"id": sid, "problem": "invalid_pcm_byte_length"})
                continue
            values = np.frombuffer(pcm, dtype="<f4")
            if values.ndim != 1 or not np.isfinite(values).all():
                wav_errors.append({"id": sid, "problem": "pcm_not_finite_mono_float32"})
                continue
            duration = len(values) / RATE
            if abs(duration - float(row.get("duration_seconds", -1))) > (1.0 / RATE + 1e-9):
                wav_errors.append({"id": sid, "problem": "pcm_duration_mismatch",
                                   "pcm_duration": duration, "declared": row.get("duration_seconds")})
            _, payload = read_ieee_float32_wav(wav_path)
            if payload != pcm:
                wav_errors.append({"id": sid, "problem": "wav_payload_differs_from_pcm"})
            declared_wav_sha = row.get("evaluation_wav_sha256")
            if not declared_wav_sha or digest(wav_path) != declared_wav_sha:
                wav_errors.append({"id": sid, "problem": "wav_sha256_mismatch"})
        except Exception as exc:
            wav_errors.append({"id": sid, "problem": f"{type(exc).__name__}: {exc}"})
    wav_names = {p.name for p in wav_paths}
    expected_names = {f"{sid}.wav" for sid in ids}
    check("float32_wav_export", len(wav_paths) == EXPECTED_RECORDINGS and wav_names == expected_names and not wav_errors,
          {"wav_count": len(wav_paths), "expected_count": EXPECTED_RECORDINGS,
           "checked_payloads": len(ids) - len(wav_errors), "errors": wav_errors[:50],
           "error_count": len(wav_errors)})

    # Check independent SQLite libraries and every selected canonical gold name.
    library_meta = manifest.get("libraries", {}) if isinstance(manifest, dict) else {}
    library_errors = []
    library_paths = set()
    domain_names = sorted(domains)
    for domain in domain_names:
        metadata = library_meta.get(domain)
        if not isinstance(metadata, dict):
            library_errors.append({"domain": domain, "problem": "missing_manifest_library"})
            continue
        path = Path(str(metadata.get("path") or ""))
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        path = path.resolve()
        library_paths.add(str(path))
        configured_inputs = {Path(input_by_id[sid].get("library_path", "")).resolve()
                             for sid in ids if input_by_id[sid].get("domain") == domain}
        if configured_inputs != {path}:
            library_errors.append({"domain": domain, "problem": "inference_library_path_mismatch"})
        if not path.is_file():
            library_errors.append({"domain": domain, "problem": "library_missing", "path": str(path)})
            continue
        if digest(path) != metadata.get("sha256"):
            library_errors.append({"domain": domain, "problem": "library_sha256_mismatch"})
            continue
        try:
            uri = path.as_uri() + "?mode=ro"
            with sqlite3.connect(uri, uri=True) as db:
                stored = {compact_whitespace(row[0]) for row in db.execute(
                    "SELECT canonical_text FROM protected_entities WHERE enabled=1")}
            missing = sorted({compact_whitespace(e.get("canonical", ""))
                              for sid in ids if input_by_id[sid].get("domain") == domain
                              for e in labels[sid].get("gold_entities", [])} - stored)
            if missing:
                library_errors.append({"domain": domain, "problem": "gold_canonical_missing_from_library",
                                       "count": len(missing), "examples": missing[:20]})
            if metadata.get("entity_count") is not None and len(stored) != metadata["entity_count"]:
                library_errors.append({"domain": domain, "problem": "entity_count_mismatch",
                                       "manifest": metadata["entity_count"], "actual": len(stored)})
        except Exception as exc:
            library_errors.append({"domain": domain, "problem": f"sqlite_read_error: {type(exc).__name__}: {exc}"})
    check("independent_domain_libraries", not library_errors and len(library_paths) == len(domain_names),
          {"domains": len(domain_names), "unique_library_paths": len(library_paths),
           "errors": library_errors[:50], "error_count": len(library_errors)})

    selected_pcm = manifest.get("selected_pcm", {}) if isinstance(manifest, dict) else {}
    selected_pcm_mismatches = []
    if set(selected_pcm) != ids:
        selected_pcm_mismatches.append({"problem": "selected_pcm_id_set_mismatch",
                                        "manifest_ids": len(selected_pcm), "input_ids": len(ids)})
    for sid in sorted(ids & set(selected_pcm)):
        metadata = selected_pcm[sid]
        row = input_by_id[sid]
        if metadata.get("sha256") != row.get("pcm_sha256") or Path(str(metadata.get("path", ""))).resolve() != Path(str(row.get("pcm_path", ""))).resolve():
            selected_pcm_mismatches.append({"id": sid, "problem": "selected_pcm_manifest_mismatch"})
    check("manifest_selected_pcm_matches", not selected_pcm_mismatches,
          {"mismatches": selected_pcm_mismatches[:30], "mismatch_count": len(selected_pcm_mismatches)})

    status = "PASS" if not errors else "FAIL"
    report = {
        "schema_version": 1,
        "status": status,
        "dataset": str(dataset),
        "recordings": len(inputs),
        "domain_recordings": dict(domains),
        "recomputed_baseline": {"hits": total_hits, "expected_mentions": total_mentions,
                                 "entity_recall": recall, "edit_distance": total_errors,
                                 "normalized_reference_chars": total_reference_chars, "CER": cer},
        "protected_hashes_read": len(protected_hashes),
        "wav_files_checked": len(wav_paths),
        "normalization": "zh-ITN, OpenCC t2s, NFKC/casefold, strip Unicode P/Z/C",
        "protected_read_scope": "summary.json pcm_sha256 values only; no protected references or audio opened",
        "checks": checks,
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
