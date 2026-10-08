"""Summarize paired entity-dictionary evaluations without hiding regressions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
from functools import lru_cache

from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT, normalized, read_jsonl, write_json
from system.zh_itn_bridge import ZhITNNormalizer


def recovered_entities(row, text, *, covered_only=False):
    occurrences = Counter()
    recovered = total = 0
    hypothesis = normalized(text)
    for entity in row["entities"]:
        if covered_only and not entity["covered"]:
            continue
        surface = normalized(entity["text"])
        occurrences[surface] += 1
        total += 1
        recovered += hypothesis.count(surface) >= occurrences[surface]
    return recovered, total


def paired_bootstrap(differences, denominators):
    if not differences:
        return [0.0, 0.0]
    if not any(differences):
        return [0.0, 0.0]
    rng = random.Random(20261003)
    rates = []
    for _ in range(2000):
        indexes = [rng.randrange(len(differences)) for _ in differences]
        rates.append(100 * sum(differences[i] for i in indexes) / sum(denominators[i] for i in indexes))
    rates.sort()
    return [rates[49], rates[1949]]


def summarize(rows, outputs, baseline, normalizer):
    total_chars = errors = itn_chars = itn_errors = 0
    recovered = entity_total = covered_recovered = covered_total = 0
    improved = worsened = equal = exact = harmful_correct = 0
    # The clean and noisy recordings of one utterance are a paired cluster.
    # Resampling them independently would overstate the effective sample size.
    baseline_deltas = defaultdict(int)
    denominators = defaultdict(int)
    changes = hints = calls = rejected = unsupported = 0
    matcher_latencies = []
    examples = []
    for key, output in outputs.items():
        row = rows[key]
        text = output["text"]
        reference = normalized(row["reference"])
        hypothesis = normalized(text)
        ref_itn = normalized(normalizer.normalize(row["reference"]).text)
        hyp_itn = normalized(normalizer.normalize(text).text)
        base_itn = normalized(normalizer.normalize(baseline[key]["text"]).text)
        e = Levenshtein.distance(reference, hypothesis)
        ei = Levenshtein.distance(ref_itn, hyp_itn)
        eb = Levenshtein.distance(ref_itn, base_itn)
        total_chars += len(reference)
        errors += e
        itn_chars += len(ref_itn)
        itn_errors += ei
        denominators[row["id"]] += len(ref_itn)
        baseline_deltas[row["id"]] += ei-eb
        improved += ei < eb
        worsened += ei > eb
        equal += ei == eb
        exact += ei == 0
        harmful_correct += eb == 0 and ei > 0
        r, t = recovered_entities(row, text)
        recovered += r
        entity_total += t
        r, t = recovered_entities(row, text, covered_only=True)
        covered_recovered += r
        covered_total += t
        ns = output.get("normalizations", [])
        changes += len(ns)
        hints += len(output.get("hints", []))
        calls += output.get("calls", 0)
        rejected += bool(output.get("rejections"))
        unsupported += sum(normalized(n["canonical"]) not in reference for n in ns)
        matcher_latencies.append(output.get("matcher_latency_ms", 0.0))
        if ei != eb or ns:
            examples.append(dict(id=row["id"], variant=row["variant"], group=row["group"],
                reference=row["reference"], raw=row["raw_text"], baseline=baseline[key]["text"],
                output=text, delta_itn_errors=ei-eb, normalizations=ns,
                hints=output.get("hints",[]), rejections=output.get("rejections", [])))
    matcher_latencies.sort()
    return dict(samples=len(outputs), cer_errors=errors, reference_characters=total_chars,
        cer_percent=100*errors/total_chars if total_chars else None,
        itn_cer_errors=itn_errors, itn_reference_characters=itn_chars,
        itn_cer_percent=100*itn_errors/itn_chars if itn_chars else None,
        recovered_entities=recovered, total_entities=entity_total,
        entity_recall_percent=100*recovered/entity_total if entity_total else None,
        recovered_covered_entities=covered_recovered, total_covered_entities=covered_total,
        covered_entity_recall_percent=100*covered_recovered/covered_total if covered_total else None,
        improved_samples=improved, worsened_samples=worsened, equal_samples=equal,
        originally_correct_damaged=harmful_correct, exact_samples=exact,
        normalizations=changes, unsupported_canonical_normalizations=unsupported,
        hints=hints, refiner_calls=calls, samples_with_rejections=rejected,
        matcher_latency_p50_ms=statistics.median(matcher_latencies) if matcher_latencies else None,
        matcher_latency_p95_ms=matcher_latencies[min(len(matcher_latencies)-1,int(len(matcher_latencies)*.95))] if matcher_latencies else None,
        paired_delta_itn_cer_95ci_pp=paired_bootstrap(list(baseline_deltas.values()),list(denominators.values())), examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/entity_library_20261003")
    args = parser.parse_args()
    normalizer = ZhITNNormalizer()
    # Rules are deterministic; references and shared ASR baselines repeat
    # across conditions and subgroup tables.
    normalizer.normalize = lru_cache(maxsize=None)(normalizer.normalize)
    rows = {(r["id"],r["variant"]):r for r in read_jsonl(args.output/"asr.jsonl")}
    sources = dict(raw_asr={"raw_asr":{key:dict(text=row["raw_text"]) for key,row in rows.items()}})
    entity_outputs = defaultdict(dict)
    for row in read_jsonl(args.output/"entity_only.jsonl"):
        for condition, result in row["conditions"].items():
            entity_outputs[condition][(row["id"],row["variant"])] = result
    sources["entity_only"] = entity_outputs
    for mode in ("offline", "streaming", "online"):
        web_outputs = defaultdict(dict)
        for row in read_jsonl(args.output/f"web_{mode}.jsonl"):
            r = row["result"]
            web_outputs[row["condition"]][(row["id"],row["variant"])] = dict(
                text=r["clean_text"],normalizations=r.get("entity_normalizations",[]),
                hints=r.get("entity_refinement_hints",[]),
                calls=r.get("session_refiner_call_count",len(row["model_calls"])),
                rejections=r.get("refiner_reject_reasons",[]),
                matcher_latency_ms=r.get("entity_matcher_latency_ms",0))
        if web_outputs:
            sources[f"web_{mode}"] = web_outputs
    summary = {}
    for source, conditions in sources.items():
        baseline = conditions.get("no_dictionary", sources["raw_asr"]["raw_asr"])
        summary[source] = {}
        for condition, outputs in conditions.items():
            summary[source][condition] = {}
            for variant in ("clean", "noise", "both"):
                chosen = {key:value for key,value in outputs.items()
                          if key in baseline and (variant == "both" or key[1] == variant)}
                summary[source][condition][variant] = summarize(rows,chosen,baseline,normalizer)
            summary[source][condition]["by_group"] = {
                group:summarize(rows,{key:value for key,value in outputs.items()
                    if key in baseline and rows[key]["group"] == group},baseline,normalizer)
                for group in ("covered_entity", "uncovered_entity", "no_entity")}
    write_json(args.output/"summary.json", summary)
    for source, conditions in summary.items():
        print(source)
        for condition, variants in conditions.items():
            row = variants["both"]
            print(condition, "n",row["samples"], "ITN-CER",round(row["itn_cer_percent"],3),
                "entity",(round(row["entity_recall_percent"],2)
                          if row["entity_recall_percent"] is not None else "n/a"),
                "better/worse",row["improved_samples"],row["worsened_samples"],
                "calls",row["refiner_calls"])


if __name__ == "__main__":
    main()
