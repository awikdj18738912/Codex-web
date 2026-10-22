#!/usr/bin/env python3
"""Analyze paired test and control outcomes for the frozen entity backend comparison."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Reuse the report's normalization/scoring implementation (and its single ITN instance).
from experiments.scripts.report_entity_backend_comparison import norm, score, read_jsonl  # noqa: E402

RUN_DEFAULT = ROOT / "results/entity_backend_comparison_20261015"
TOKENS_DEFAULT = Path("/home/aim0/data/models/ASR/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30/tokens.txt")
ARMS = {
    "qwen": "qwen_raw",
    "nativehotword": "native_hotword",
    "nativebeam": "native_beam",
    "hybrid": "qwen_ctc_sidecar",
}
PAIRS = (
    ("nativehotword_vs_qwen", "nativehotword", "qwen"),
    ("hybrid_vs_qwen", "hybrid", "qwen"),
    ("nativehotword_vs_nativebeam", "nativehotword", "nativebeam"),
    ("hybrid_vs_nativehotword", "hybrid", "nativehotword"),
)


def load_tokens(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rsplit(" ", 1)
            if len(parts) == 2:
                out.add(parts[0])
    return out


def barechar_supported(canonical: str, tokens: set[str]) -> bool:
    # Match native hotword_payload exactly: inspect the original spelling and
    # remove whitespace only. Normalizing here would overstate injectability.
    chars = [c for c in canonical if not c.isspace()]
    return bool(chars) and all(c in tokens for c in chars)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def nullable_text(row: dict[str, str], arm: str) -> str | None:
    value = row.get(arm + "_text")
    if value is not None and value != "":
        return value
    # CSV encodes both a missing hypothesis and a genuine empty transcription as
    # an empty field. The report's metric columns distinguish those cases.
    if row.get(arm + "_entity_hits") not in (None, ""):
        return ""
    return None


def paired_item(sample: dict[str, str], refs: dict[str, dict], pair: tuple[str, str, str]) -> dict[str, Any] | None:
    _, cand, base = pair
    ctext, btext = nullable_text(sample, ARMS[cand]), nullable_text(sample, ARMS[base])
    if ctext is None or btext is None:
        return None
    refobj = refs[str(sample["id"])]
    ref = str(refobj.get("reference", ""))
    gold = refobj.get("gold_entities", [])
    cs, bs = score(ref, ctext, gold), score(ref, btext, gold)
    gc = Counter(cs["gold_counts"])
    cn, bn = norm(ctext), norm(btext)
    recovered = lost = 0
    for canonical, mentions in gc.items():
        delta = min(mentions, cn.count(canonical)) - min(mentions, bn.count(canonical))
        recovered += max(delta, 0)
        lost += max(-delta, 0)
    return {
        "id": str(sample["id"]), "source_group": str(sample.get("source_group") or "unknown"),
        "recovered_hits": recovered, "lost_hits": lost,
        "candidate_hits": cs["entity_hits"], "baseline_hits": bs["entity_hits"],
        "gold_mentions": cs["entity_mentions"],
        "candidate_cer_errors": cs["cer_errors"], "baseline_cer_errors": bs["cer_errors"],
        "reference_chars": cs["reference_chars"],
        "cer_improved": cs["cer_errors"] < bs["cer_errors"],
        "cer_degraded": cs["cer_errors"] > bs["cer_errors"],
        "cer_equal": cs["cer_errors"] == bs["cer_errors"],
    }


def bootstrap_ci(items: list[dict[str, Any]], seed: int, reps: int = 1000) -> dict[str, Any]:
    """Cluster bootstrap; each replicate resamples source_groups with replacement."""
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_group[item["source_group"]].append(item)
    groups = sorted(by_group)
    if not groups:
        return {"clusters": 0, "replicates": reps, "entity_recall_difference_95ci": None, "cer_difference_95ci": None}
    rng = random.Random(seed)
    entity_diffs: list[float] = []
    cer_diffs: list[float] = []
    for _ in range(reps):
        chosen = [rng.choice(groups) for _ in groups]
        batch = [item for group in chosen for item in by_group[group]]
        mentions = sum(x["gold_mentions"] for x in batch)
        chars = sum(x["reference_chars"] for x in batch)
        if mentions:
            entity_diffs.append((sum(x["candidate_hits"] for x in batch) - sum(x["baseline_hits"] for x in batch)) / mentions)
        if chars:
            cer_diffs.append((sum(x["candidate_cer_errors"] for x in batch) - sum(x["baseline_cer_errors"] for x in batch)) / chars)
    def ci(vals: list[float]):
        if not vals:
            return None
        vals.sort()
        # Linear percentile interpolation, equivalent to common percentile CI definitions.
        def percentile(p: float) -> float:
            pos = (len(vals) - 1) * p
            lo = math.floor(pos); hi = math.ceil(pos)
            return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
        return [percentile(0.025), percentile(0.975)]
    return {"clusters": len(groups), "replicates": reps,
            "entity_recall_difference_95ci": ci(entity_diffs), "cer_difference_95ci": ci(cer_diffs)}


def library_names(inp: dict[str, Any], run: Path, cache: dict[str, list[str]]) -> list[str]:
    lib = Path(str(inp.get("library_path") or ""))
    if not lib.is_file():
        lib = run / "libraries" / lib.name
    if not lib.is_file():
        return []
    key = str(lib.resolve())
    if key not in cache:
        with sqlite3.connect(f"file:{key}?mode=ro", uri=True) as con:
            cache[key] = [str(r[0]) for r in con.execute("SELECT DISTINCT canonical_text FROM protected_entities WHERE enabled=1")]
    return cache[key]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=RUN_DEFAULT)
    ap.add_argument("--tokens", type=Path, default=TOKENS_DEFAULT)
    ap.add_argument("--allow-partial", action="store_true", help="permit fewer than 1829 paired test examples")
    args = ap.parse_args()
    run = args.run_dir.resolve()
    summary_path, samples_path = run / "summary.json", run / "sample_comparison.csv"
    inputs_path, refs_path, native_path = run / "inputs.jsonl", run / "references.json", run / "native_results.jsonl"
    for path in (summary_path, samples_path, inputs_path, refs_path, native_path, args.tokens):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # The report summary is the finalization marker and supplies the dev-selected values.
    if not isinstance(summary.get("selection"), dict) or not summary["selection"]:
        raise ValueError("summary.json lacks the dev-selected parameter record")
    inputs_rows = read_jsonl(inputs_path)
    inputs = {str(x["id"]): x for x in inputs_rows}
    if len(inputs) != len(inputs_rows):
        raise ValueError("duplicate input IDs")
    refs_doc = json.loads(refs_path.read_text(encoding="utf-8"))
    refs = refs_doc.get("samples", {})
    if set(inputs) != set(refs):
        raise ValueError("inputs/reference IDs do not match")
    rows = read_csv(samples_path)
    samples = {str(x["id"]): x for x in rows}
    if len(samples) != len(rows):
        raise ValueError("duplicate sample_comparison IDs")
    # Validate sample table entries against the frozen source grouping.
    for sid, sample in samples.items():
        if sid not in inputs:
            raise ValueError(f"sample comparison contains unknown ID {sid}")
        if sample.get("split") != inputs[sid].get("split"):
            raise ValueError(f"split mismatch for {sid}")
        sample["source_group"] = str(inputs[sid].get("source_group") or "unknown")
    test_ids = {sid for sid, x in inputs.items() if x.get("split") == "test"}
    test_rows = [samples[sid] for sid in sorted(test_ids & samples.keys())]
    if len(test_rows) != 1829 and not args.allow_partial:
        raise ValueError(f"formal analysis requires all 1829 test sample rows; found {len(test_rows)} (use --allow-partial for interim analysis)")
    test_pairs: dict[str, list[dict[str, Any]]] = {}
    for pair in PAIRS:
        items = [x for row in test_rows if (x := paired_item(row, refs, pair)) is not None]
        if len(items) != len(test_rows) and not args.allow_partial:
            raise ValueError(f"{pair[0]} missing paired hypotheses for {len(test_rows)-len(items)} test rows")
        test_pairs[pair[0]] = items

    pair_results = {}
    for pair in PAIRS:
        name = pair[0]; items = test_pairs[name]
        cand = sum(x["candidate_hits"] for x in items); base = sum(x["baseline_hits"] for x in items)
        mentions = sum(x["gold_mentions"] for x in items)
        cand_err = sum(x["candidate_cer_errors"] for x in items); base_err = sum(x["baseline_cer_errors"] for x in items)
        chars = sum(x["reference_chars"] for x in items)
        pair_results[name] = {
            "candidate_arm": pair[1], "baseline_arm": pair[2], "paired_samples": len(items),
            "entity_recovered_hits": sum(x["recovered_hits"] for x in items),
            "entity_lost_hits": sum(x["lost_hits"] for x in items),
            "entity_net_hits": cand - base,
            "cer_improved_samples": sum(x["cer_improved"] for x in items),
            "cer_degraded_samples": sum(x["cer_degraded"] for x in items),
            "cer_equal_samples": sum(x["cer_equal"] for x in items),
            "candidate_entity_recall": cand / mentions if mentions else None,
            "baseline_entity_recall": base / mentions if mentions else None,
            "entity_recall_difference": (cand - base) / mentions if mentions else None,
            "candidate_cer": cand_err / chars if chars else None,
            "baseline_cer": base_err / chars if chars else None,
            "cer_net_difference": (cand_err - base_err) / chars if chars else None,
            "cer_net_error_count": cand_err - base_err,
            "gold_mentions_denominator": mentions, "reference_chars_denominator": chars,
            "cluster_bootstrap": bootstrap_ci(items, 20261015),
        }

    tokens = load_tokens(args.tokens)
    lib_cache: dict[str, list[str]] = {}
    supports: dict[str, dict[str, float]] = {
        "supported": defaultdict(float), "unsupported": defaultdict(float),
        "not_in_library": defaultdict(float),
    }
    support_class_by_norm: dict[str, set[str]] = defaultdict(set)
    test_gold_mentions = 0
    for sid in sorted(test_ids & samples.keys()):
        sample, gold = samples[sid], refs[sid].get("gold_entities", [])
        texts = {name: nullable_text(sample, arm) for name, arm in ARMS.items() if name in ("qwen", "nativehotword", "hybrid")}
        gc = score(str(refs[sid].get("reference", "")), "", gold)["gold_counts"]
        # Determine injectability from each raw gold canonical (or surface
        # fallback), then map the status back to normalized hit-count keys.
        raw_supported_counts: Counter[tuple[str, str]] = Counter()
        available_names = set(library_names(inputs[sid], run, lib_cache))
        for ent in gold:
            raw = str(ent.get("canonical") or ent.get("surface") or "")
            key = norm(raw)
            if not key:
                continue
            if raw not in available_names:
                status = "not_in_library"
            elif barechar_supported(raw, tokens):
                status = "supported"
            else:
                status = "unsupported"
            raw_supported_counts[(key, status)] += 1
        raw_statuses: dict[str, set[str]] = defaultdict(set)
        raw_mentions_by_norm: Counter[str] = Counter()
        for (key, status), count in raw_supported_counts.items():
            raw_statuses[key].add(status)
            raw_mentions_by_norm[key] += count
        for canonical, count in gc.items():
            if raw_mentions_by_norm[canonical] != count:
                raise ValueError(f"raw gold entity count mismatch for {canonical!r} in {sid}: raw={raw_mentions_by_norm[canonical]}, normalized={count}")
            statuses = raw_statuses.get(canonical, set())
            if len(statuses) > 1:
                raise ValueError(f"ambiguous raw support classes for normalized gold entity {canonical!r} in {sid}: {sorted(statuses)}")
            group = next(iter(statuses), "not_in_library")
            support_class_by_norm[canonical].add(group)
            test_gold_mentions += count
            supports[group]["gold_mentions"] += count
            for arm, text in texts.items():
                if text is None:
                    continue
                supports[group][arm + "_scored_mentions"] += count
                supports[group][arm + "_hits"] += min(count, norm(text).count(canonical))
    ambiguous_norms = {key: sorted(classes) for key, classes in support_class_by_norm.items() if len(classes) > 1}
    # Cross-sample status differences are retained as explicit ambiguity notes;
    # each row still contributes a normalized entity's hits only once.
    expected_test_mentions = summary.get("splits", {}).get("test", {}).get("qwen_raw", {}).get("entity_mentions")
    if len(test_rows) == 1829 and expected_test_mentions is not None and test_gold_mentions != int(expected_test_mentions):
        raise ValueError(f"test gold mention denominator mismatch: analyzed={test_gold_mentions}, summary={expected_test_mentions}")
    support_results = {}
    for group in ("supported", "unsupported", "not_in_library"):
        d = supports[group]
        support_results[group] = {"gold_mentions": int(d["gold_mentions"]), "arms": {}}
        for arm in ("qwen", "nativehotword", "hybrid"):
            denom = d[arm + "_scored_mentions"]
            support_results[group]["arms"][arm] = {
                "scored_gold_mentions": int(denom), "entity_hits": int(d[arm + "_hits"]),
                "entity_recall": d[arm + "_hits"] / denom if denom else None,
            }

    native_rows = read_jsonl(native_path)
    native_hot = {}
    for row in native_rows:
        if row.get("condition") == "hotword" and row.get("score") is not None:
            # The report's dev selection is authoritative; use only its selected score.
            selected_score = summary["selection"].get("native_hotword_score")
            if selected_score is not None and float(row["score"]) == float(selected_score):
                native_hot[str(row.get("id"))] = row
    # Confirm the selected native setting is represented in final native records.
    if not native_hot:
        raise ValueError("native_results.jsonl has no hotword rows at the dev-selected score")

    control_ids = {sid for sid, x in inputs.items() if x.get("split") == "control"} & samples.keys()
    control_arms = {
        "qwen_raw": "qwen_raw", "ctc_greedy": "ctc_greedy",
        "native_greedy": "native_greedy", "native_beam": "native_beam",
        "native_hotword": "native_hotword", "hybrid": "qwen_ctc_sidecar",
    }
    control_results: dict[str, Any] = {}
    for name, arm in control_arms.items():
        err = chars = n = 0
        candidates: set[tuple[str, str]] = set()
        incremental_mentions = 0
        baseline_arm = "native_beam" if name == "native_hotword" else "qwen_raw"
        for sid in sorted(control_ids):
            row = samples[sid]
            text = nullable_text(row, arm)
            if text is None:
                continue
            ref = str(refs[sid].get("reference", "")); gold = refs[sid].get("gold_entities", [])
            sc = score(ref, text, gold)
            err += sc["cer_errors"]; chars += sc["reference_chars"]; n += 1
            baseline = nullable_text(row, baseline_arm)
            if baseline is None:
                continue
            refn, hypn, basen = norm(ref), norm(text), norm(baseline)
            for canonical in library_names(inputs[sid], run, lib_cache):
                needle = norm(canonical)
                if needle and needle not in refn:
                    added = max(0, hypn.count(needle) - basen.count(needle))
                    if added:
                        candidates.add((sid, canonical))
                        incremental_mentions += added
        control_results[name] = {
            "baseline_for_audit": baseline_arm,
            "samples": n, "cer_errors": err, "reference_chars": chars,
            "cer": err / chars if chars else None,
            "new_library_entity_audit_candidate_count": len(candidates),
            "new_library_entity_audit_candidate_mentions": incremental_mentions,
            "audit_note": "Automatic lexical candidates only; not human-reviewed precision.",
        }

    out = {
        "dataset": str(run), "status": "partial" if len(test_rows) < 1829 else "complete",
        "allow_partial": bool(args.allow_partial), "test_samples": len(test_rows),
        "required_test_samples": 1829, "dev_selection": summary["selection"],
        "normalization": "Reuses report_entity_backend_comparison.norm/score: ITN, OpenCC t2s, NFKC, casefold and removal of Unicode P/Z/C; entity counts clipped by gold mentions.",
        "pairwise_test": pair_results,
        "native_barechar_coverage": {
            "tokens_path": str(args.tokens), "definition": "An entity is injectable only when its original gold canonical (or surface fallback) is present verbatim in that sample's enabled library and every original non-whitespace character has a standalone token, matching native cjkchar encoding. Otherwise it is unsupported (in library but unencodable) or not_in_library. Hit counting remains normalized and mention-clipped.",
            "by_gold_entity_support": support_results,
            "analyzed_test_gold_mentions": test_gold_mentions,
            "summary_test_gold_mentions": expected_test_mentions,
            "normalized_entity_class_ambiguities": ambiguous_norms,
        },
        "control": control_results,
        "bootstrap": {"unit": "source_group cluster", "seed": 20261015, "replicates": 1000,
                      "method": "Resample source_group clusters with replacement; recompute pooled recall and CER differences using summed gold mentions/reference characters in every replicate.",
                      "difference_direction": "candidate minus baseline; negative CER difference is better."},
        "limitations": ["Control audit counts are automatically generated lexical candidates and must not be interpreted as human-reviewed precision.",
                        "Bare-character support is a token representability measure, not evidence that the recognizer will decode the entity correctly.",
                        "All results depend on the final paired sample table and the explicitly dev-selected settings in summary.json."],
    }
    (run / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Entity backend comparison analysis", "", f"Status: **{out['status']}** ({len(test_rows)}/1829 paired test rows). Test settings come from the dev selection recorded in `summary.json`.",
             "", "Pair differences are candidate minus baseline. Recall differences use summed gold mentions; CER differences use summed normalized reference characters. Negative CER is an improvement. Confidence intervals use 1,000 source-group cluster bootstrap replicates with seed 20261015.",
             "", "| Paired comparison | N | Recovered hits | Lost hits | CER improved samples | CER degraded samples | Recall difference | CER net difference | Recall 95% CI | CER 95% CI |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pair, _, _ in PAIRS:
        x = pair_results[pair]; b = x["cluster_bootstrap"]
        def fmt(v): return "NA" if v is None else f"{v:+.4%}"
        lines.append(f"| {pair} | {x['paired_samples']} | {x['entity_recovered_hits']} | {x['entity_lost_hits']} | {x['cer_improved_samples']} | {x['cer_degraded_samples']} | {fmt(x['entity_recall_difference'])} | {fmt(x['cer_net_difference'])} | {fmt_ci(b['entity_recall_difference_95ci'])} | {fmt_ci(b['cer_difference_95ci'])} |")
    lines += ["", "## Native bare-character token coverage", "", "Supported means the original gold canonical (or surface fallback) is present verbatim in that sample's enabled library and every original non-whitespace character is a standalone token in native tokens.txt. `Unsupported` means present in the library but not encodable; `not_in_library` is reported separately. Hits remain normalized and mention-clipped. Rates are mention-weighted.",
              "", "| Coverage class | Gold mentions | Qwen hits / recall | Native hotword hits / recall | Hybrid hits / recall |", "|---|---:|---:|---:|---:|"]
    for group in ("supported", "unsupported", "not_in_library"):
        x = support_results[group]
        cols = []
        for arm in ("qwen", "nativehotword", "hybrid"):
            a = x["arms"][arm]; cols.append(f"{a['entity_hits']}/{a['scored_gold_mentions']} ({fmt_pct(a['entity_recall'])})")
        lines.append(f"| {group} | {x['gold_mentions']} | " + " | ".join(cols) + " |")
    lines += ["", "## Control CER and lexical audit candidates", "", "For native hotword, the lexical baseline is native beam; other arms are compared with Qwen raw. Candidates are library canonicals absent from the reference whose occurrence count increased. These are automatic audit candidates, not human-reviewed precision.",
              "", "| Arm | Baseline | N | CER | New candidate entities | Added candidate mentions |", "|---|---|---:|---:|---:|---:|"]
    for arm, x in control_results.items():
        lines.append(f"| {arm} | {x['baseline_for_audit']} | {x['samples']} | {fmt_pct(x['cer'])} | {x['new_library_entity_audit_candidate_count']} | {x['new_library_entity_audit_candidate_mentions']} |")
    lines.append("")
    (run / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


def fmt_ci(values: list[float] | None) -> str:
    if values is None:
        return "NA"
    return f"[{values[0]:+.4%}, {values[1]:+.4%}]"


def fmt_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2%}"


if __name__ == "__main__":
    raise SystemExit(main())
