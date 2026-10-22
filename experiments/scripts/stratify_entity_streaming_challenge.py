#!/usr/bin/env python3
"""Build leakage-separated diagnostic groups for the frozen entity challenge.

This is an offline indexing/reporting tool. It never reads or modifies PCM contents,
and it does not invoke ASR inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "results/entity_challenge_20261014_scene_v2/dataset"
EVAL = ROOT / "results/entity_challenge_scene_eval_20261014_v2"
SCORER_PATH = ROOT / "experiments/scripts/score_current_entity_challenge.py"
sys.path.insert(0, str(SCORER_PATH.parent))
import score_current_entity_challenge as scorer  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bucket(duration: float) -> str:
    return "le_4s" if duration <= 4 else ("4_to_8s" if duration <= 8 else "gt_8s")


def target_positions(reference: str, gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Locate normalized canonical mentions, preferring annotation offsets when valid."""
    norm_ref = scorer.normalize(reference)
    out = []
    for ent in gold:
        canonical = str(ent.get("canonical") or ent.get("surface") or "")
        needle = scorer.normalize(canonical)
        if not needle:
            continue
        starts: list[int] = []
        raw_start, raw_end = ent.get("start"), ent.get("end")
        if isinstance(raw_start, int) and isinstance(raw_end, int) and 0 <= raw_start <= raw_end <= len(reference):
            annotated = scorer.normalize(reference[raw_start:raw_end])
            offset = len(scorer.normalize(reference[:raw_start]))
            if annotated == needle and norm_ref[offset:offset + len(needle)] == needle:
                starts = [offset]
        if not starts:
            cursor = 0
            while True:
                at = norm_ref.find(needle, cursor)
                if at < 0:
                    break
                starts.append(at)
                cursor = at + 1
        for at in starts:
            remain = len(norm_ref) - (at + len(needle))
            out.append({"canonical": canonical, "normalized_start": at,
                        "normalized_end": at + len(needle),
                        "remaining_normalized_chars": remain,
                        "tail_lt_8": remain < 8,
                        "position_source": "validated_reference_offsets" if starts and isinstance(raw_start, int) and isinstance(raw_end, int) and scorer.normalize(reference[raw_start:raw_end]) == needle else "canonical_search"})
    return out


def check_count(row: dict[str, Any]) -> int:
    repair = scorer.get_path(row, "result", "entity_audio_repair", default={}) or {}
    value = scorer.get_path(repair, "evaluation_backend", "checks")
    if value is None:
        value = repair.get("checks", 0)
    if isinstance(value, int): return value
    if isinstance(value, (list, dict)): return len(value)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--eval-dir", type=Path, default=EVAL)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/entity_streaming_testset_diagnostics_20261015")
    args = ap.parse_args()
    dataset, eval_dir, outdir = args.dataset.resolve(), args.eval_dir.resolve(), args.output_dir.resolve()
    source_files = [dataset / name for name in ("samples.jsonl", "references.json", "baseline_raw.jsonl", "inference_inputs.jsonl", "manifest.json")]
    source_hashes_before = {str(p): digest(p) for p in source_files}
    sample_rows = read_jsonl(dataset / "samples.jsonl")
    inputs = read_jsonl(eval_dir / "inference_inputs.jsonl")
    baseline_rows = read_jsonl(dataset / "baseline_raw.jsonl")
    result_rows = read_jsonl(eval_dir / "results.jsonl")
    traces = read_jsonl(eval_dir / "asr_trace.jsonl")
    refs_doc = json.loads((dataset / "references.json").read_text(encoding="utf-8"))
    refs = refs_doc["samples"]
    samples = {str(r["id"]): r for r in sample_rows}
    input_map = {str(r["id"]): r for r in inputs}
    baseline = {str(r["id"]): r for r in baseline_rows}
    results = {str(r["id"]): r for r in result_rows}
    if any(len(m) != len(rows) for m, rows in ((samples, sample_rows), (input_map, inputs), (baseline, baseline_rows), (results, result_rows))):
        raise ValueError("Duplicate IDs in fixed dataset/evaluation artifacts")
    ids = set(refs)
    if len(ids) != 2000 or any(set(m) != ids for m in (samples, input_map, baseline, results)):
        raise ValueError("Expected exactly 2000 identical IDs in references, samples, inference inputs, baseline, results")

    trace_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        if t.get("primary") and t.get("endpoint") in ("/stream/chunk", "/stream/finish"):
            trace_by_id[str(t.get("id"))].append(t)
    safe_keys = ("id", "pcm_path", "pcm_sha256", "library_path", "domain", "duration_seconds")
    inference_rows, scoring_rows = [], []
    for sid in sorted(ids):
        inp, ref_obj, row = input_map[sid], refs[sid], results[sid]
        ref = str(ref_obj.get("reference", ""))
        gold = ref_obj.get("gold_entities", [])
        positions = target_positions(ref, gold)
        dur = float(inp["duration_seconds"])
        tail_n = sum(bool(p["tail_lt_8"]) for p in positions)
        tail_group = "no_gold_position" if not positions else ("tail_only" if tail_n == len(positions) else ("tail_present" if tail_n else "non_tail"))
        base_hits = scorer.score_entities(baseline[sid].get("raw_text", ""), gold)["hits"]
        repair = scorer.get_path(row, "result", "entity_audio_repair", default={}) or {}
        repaired_text = scorer.get_path(row, "result", "raw_text", default="") or ""
        repaired_hits = scorer.score_entities(str(repaired_text), gold)["hits"]
        recovered, lost = scorer.mention_change(str(baseline[sid].get("raw_text", "")), str(repaired_text), gold)
        selected_trace = sorted(trace_by_id.get(sid, []), key=lambda t: float(t.get("seconds", 0) or 0))
        # Record observations at finish and last primary chunk; these are not entity-onset timings.
        chunks = [t for t in selected_trace if t.get("endpoint") == "/stream/chunk"]
        finishes = [t for t in selected_trace if t.get("endpoint") == "/stream/finish"]
        def obs(t: dict[str, Any] | None) -> dict[str, Any] | None:
            if not t: return None
            result = t.get("result") or {}
            return {"seconds": t.get("seconds"), "text": result.get("text"),
                    "active_segment": result.get("active_segment"),
                    "completed_segment_count": len(result.get("completed_segments", []) or [])}
        checks = check_count(row)
        module_error = bool(row.get("module_errors") or scorer.get_path(row, "result", "module_errors") or row.get("error") or scorer.get_path(row, "result", "error"))
        inference_rows.append({k: inp[k] for k in safe_keys if k in inp} | {"duration_bucket": bucket(dur)})
        scoring_rows.append({"id": sid, "duration_bucket": bucket(dur), "tail_position_group": tail_group,
            "target_positions": positions, "gold_mentions": len(positions), "baseline_hit_mentions": base_hits,
            "audio_repaired_hit_mentions": repaired_hits, "recovered_mentions": recovered,
            "lost_mentions": lost, "checks": checks, "module_error": module_error,
            "no_check_error": checks == 0 and module_error,
            "trace_observation": {"primary_chunk_count": len(chunks), "last_primary_chunk": obs(chunks[-1] if chunks else None),
                                  "finish": obs(finishes[-1] if finishes else None)}})

    def summarize(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in rows: groups[key_fn(row)].append(row)
        out = []
        for key, members in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            out.append({"group": list(key), "samples": len(members), "unique_ids": len({x["id"] for x in members}),
                "baseline_hit_mentions": sum(x["baseline_hit_mentions"] for x in members),
                "audio_repaired_hit_mentions": sum(x["audio_repaired_hit_mentions"] for x in members),
                "recovered_mentions": sum(x["recovered_mentions"] for x in members),
                "lost_mentions": sum(x["lost_mentions"] for x in members),
                "no_check_error_samples": sum(x["no_check_error"] for x in members),
                "checks": sum(x["checks"] for x in members)})
        return out

    length = summarize(scoring_rows, lambda r: (r["duration_bucket"],))
    cross = summarize(scoring_rows, lambda r: (r["duration_bucket"], r["tail_position_group"]))
    totals = summarize(scoring_rows, lambda r: ("all",))[0]
    notes = {
      "duration_buckets": {"le_4s": "duration <= 4 seconds", "4_to_8s": "4 < duration <= 8 seconds", "gt_8s": "duration > 8 seconds"},
      "tail_definition": "Text-position approximation: a gold canonical in reference has fewer than 8 remaining normalized characters after it. tail_present means at least one target mention is in this tail; tail_only means all located mentions are; non_tail means none are. Not audio timing and does not imply the actual implementation necessarily rejects the case; the 8-character right-context constraint applies to incomplete segments, while completed segments may be handled differently.",
      "trace_definition": "Observational primary last-chunk and finish fields from asr_trace.jsonl only. No entity first-occurrence time is inferred.",
      "repaired_hit_definition": "Offline scorer score_entities on result.raw_text (audio-repair output); this does not rerun inference.",
      "no_check_error_definition": "Samples with zero recorded entity repair checks and a recorded module/result error."}
    outdir.mkdir(parents=True, exist_ok=True)
    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_jsonl(outdir / "inference_groups.jsonl", inference_rows)
    write_jsonl(outdir / "scoring_groups.jsonl", scoring_rows)
    report = {"dataset": str(dataset), "eval_dir": str(eval_dir), "samples": len(scoring_rows), "totals": totals,
              "by_duration": length, "by_duration_and_tail_position": cross, "method_notes": notes}
    (outdir / "group_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (outdir / "group_report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["duration_bucket", "tail_position_group", "samples", "unique_ids", "baseline_hit_mentions", "audio_repaired_hit_mentions", "recovered_mentions", "lost_mentions", "no_check_error_samples", "checks"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for item in cross: w.writerow(dict(zip(("duration_bucket", "tail_position_group"), item["group"])) | {k: item[k] for k in fields[2:]})

    source_hashes_after = {str(p): digest(p) for p in source_files}
    allowed = set(safe_keys) | {"duration_bucket"}
    leakage_keys = {"reference", "gold_entities", "gold", "text", "raw_text", "target_positions", "tail_position_group", "baseline_hit_mentions", "audio_repaired_hit_mentions"}
    checks = {
      "sample_id_set_unchanged": set(samples) == ids and {r["id"] for r in inference_rows} == ids and {r["id"] for r in scoring_rows} == ids,
      "all_2000_ids_unique_once": len(inference_rows) == len(scoring_rows) == 2000 and len({r["id"] for r in inference_rows}) == len({r["id"] for r in scoring_rows}) == 2000,
      "no_input_id_loss": {r["id"] for r in inference_rows} == set(input_map),
      "inference_has_no_gold_or_text_leakage": all(not (set(r) & leakage_keys) and set(r) <= allowed for r in inference_rows),
      "original_references_baseline_unchanged_during_run": source_hashes_before == source_hashes_after,
      "input_file_hashes_before_output": source_hashes_before,
      "input_file_hashes_after_output": source_hashes_after,
      "all_groups_cover_all_samples": sum(x["samples"] for x in length) == 2000 and sum(x["samples"] for x in cross) == 2000,
      "every_group_unique_ids_equal_samples": all(x["samples"] == x["unique_ids"] for x in length + cross),
      "scoring_imported_from": str(SCORER_PATH),
      "source_data_files": source_hashes_before,
    }
    (outdir / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not all(v for k, v in checks.items() if k not in ("input_file_hashes_before_output", "source_data_files", "scoring_imported_from")):
        raise ValueError("Output verification failed")
    print(json.dumps({"output_dir": str(outdir), "totals": totals, "by_duration": length,
                      "by_duration_and_tail_position": cross, "verification": str(outdir / "verification.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
