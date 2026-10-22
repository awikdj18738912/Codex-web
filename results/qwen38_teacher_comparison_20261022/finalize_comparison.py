#!/usr/bin/env python3
"""Validate blind reviews and assemble the Qwen3.8 teacher comparison."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
VERDICTS = {"PASS", "MINOR", "FAIL", "AMBIGUOUS"}
WINNERS = {"A", "B", "TIE", "UNRESOLVED"}
IDENTITIES = {"qwen30", "qwen38", "TIE", "UNRESOLVED"}
REVIEW_FILES = [
    "blind_review_001_067.jsonl",
    "blind_review_068_134.jsonl",
    "blind_review_135_200.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line: {path.name}:{line_no}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path.name}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object: {path.name}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_map(filename: str, field: str) -> dict[int, dict[str, Any]]:
    result = {}
    for row in read_jsonl(ROOT / filename):
        case_no = row.get("case_no")
        if not isinstance(case_no, int) or isinstance(case_no, bool):
            raise ValueError(f"Invalid case_no in {filename}: {case_no!r}")
        if case_no in result:
            raise ValueError(f"Duplicate case_no {case_no} in {filename}")
        result[case_no] = row
    return result


def require_text(row: dict[str, Any], field: str, case_no: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"case {case_no}: {field} must be a string")
    return value


def verdict(value: Any, label: str) -> str:
    if value not in VERDICTS:
        raise ValueError(f"{label}: invalid verdict {value!r}")
    return value


def identity_for(winner: str, key: dict[str, Any], case_no: int) -> str:
    if winner in {"TIE", "UNRESOLVED"}:
        return winner
    origin = key.get(f"{winner.lower()}_is")
    if origin not in {"qwen30", "qwen38"}:
        raise ValueError(f"case {case_no}: invalid blind key {winner.lower()}_is={origin!r}")
    return origin


def csv_safe(value: Any) -> Any:
    """Protect spreadsheet viewers; JSON outputs keep their original values."""
    if isinstance(value, str) and (value[:1] in ("\t", "\r") or value.lstrip(" \t\r\n")[:1] in ("=", "+", "-", "@")):
        return "'" + value
    return value


def main() -> None:
    # Validate the complete request and generation populations first. Failed
    # generations are accounted for, but are intentionally absent from blind review.
    sample = read_jsonl(ROOT / "sample_200.jsonl")
    predictions = read_jsonl(ROOT / "predictions_qwen38.jsonl")
    if len(sample) != 200:
        raise ValueError(f"Expected 200 requested sample records, got {len(sample)}")
    sample_by_case: dict[int, dict[str, Any]] = {}
    sample_by_id: dict[str, dict[str, Any]] = {}
    for row in sample:
        case_no, rid = row.get("case_no"), row.get("id")
        if not isinstance(case_no, int) or isinstance(case_no, bool) or case_no in sample_by_case:
            raise ValueError(f"Invalid or duplicate sample case_no: {case_no!r}")
        if not isinstance(rid, str) or not rid or rid in sample_by_id:
            raise ValueError(f"Invalid or duplicate sample id: {rid!r}")
        sample_by_case[case_no] = row
        sample_by_id[rid] = row
    expected = set(sample_by_case)
    pred_by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in pred_by_id:
            raise ValueError(f"Missing or duplicate prediction id: {rid!r}")
        if rid not in sample_by_id:
            raise ValueError(f"Unexpected prediction id: {rid}")
        source = sample_by_id[rid]
        if row.get("case_no") != source["case_no"] or row.get("input") != source.get("input"):
            raise ValueError(f"Prediction case_no/input mismatch for id={rid}")
        if row.get("baseline_target") != source.get("baseline_target"):
            raise ValueError(f"Prediction baseline_target mismatch for id={rid}")
        pred_by_id[rid] = row
    if set(pred_by_id) != set(sample_by_id):
        raise ValueError("predictions_qwen38 must contain exactly one record for every sample id")
    valid_ids = {
        rid for rid, row in pred_by_id.items()
        if row.get("error") in (None, "") and isinstance(row.get("clean_text"), str) and row["clean_text"].strip()
    }
    errors = len(predictions) - len(valid_ids)
    expected = {sample_by_id[rid]["case_no"] for rid in valid_ids}

    cases = read_map("blind_cases.jsonl", "case_no")
    keys = read_map("blind_key.jsonl", "case_no")
    if set(cases) != expected:
        missing, extra = sorted(expected - set(cases)), sorted(set(cases) - expected)
        raise ValueError(f"blind_cases coverage differs from valid generation set; missing={missing}, extra={extra}")
    if set(keys) != expected:
        raise ValueError("blind_key coverage differs from valid generation set")

    reviews: dict[int, dict[str, Any]] = {}
    for filename in REVIEW_FILES:
        for row in read_jsonl(ROOT / filename):
            case_no = row.get("case_no")
            if not isinstance(case_no, int) or isinstance(case_no, bool):
                raise ValueError(f"Invalid case_no in {filename}: {case_no!r}")
            if case_no in reviews:
                raise ValueError(f"Duplicate review for case_no {case_no}")
            reviews[case_no] = row
    if set(reviews) != expected:
        missing, extra = sorted(expected - set(reviews)), sorted(set(reviews) - expected)
        raise ValueError(f"Review coverage differs from valid generation set; missing={missing}, extra={extra}")

    lead_by_case: dict[int, dict[str, Any]] = {}
    lead_path = ROOT / "lead_adjudications.jsonl"
    if lead_path.exists():
        for row in read_jsonl(lead_path):
            case_no = row.get("case_no")
            if not isinstance(case_no, int) or isinstance(case_no, bool) or case_no not in expected:
                raise ValueError(f"Invalid lead adjudication case_no: {case_no!r}")
            if case_no in lead_by_case:
                raise ValueError(f"Duplicate lead adjudication for case {case_no}")
            verdict(row.get("qwen30_verdict"), f"lead case {case_no} qwen30_verdict")
            verdict(row.get("qwen38_verdict"), f"lead case {case_no} qwen38_verdict")
            if row.get("winner_identity") not in IDENTITIES:
                raise ValueError(f"case {case_no}: invalid lead winner_identity")
            require_text(row, "reason", case_no)
            lead_by_case[case_no] = row

    output = []
    model_grades = {"qwen30": Counter(), "qwen38": Counter()}
    pairwise, identical, output_change = Counter(), Counter(), Counter()
    valid_comparisons = 0
    for case_no in sorted(expected):
        case, key, review = cases[case_no], keys[case_no], reviews[case_no]
        if key.get("identical") not in (True, False):
            raise ValueError(f"case {case_no}: blind_key identical must be boolean")
        source = require_text(case, "input", case_no)
        candidate_a = require_text(case, "candidate_a", case_no)
        candidate_b = require_text(case, "candidate_b", case_no)
        if (candidate_a == candidate_b) != key["identical"]:
            raise ValueError(f"case {case_no}: blind_key identical disagrees with candidate text")
        for field in ("a_verdict", "b_verdict"):
            verdict(review.get(field), f"case {case_no} {field}")
        for field in ("reason", "source_evidence", "a_evidence", "b_evidence"):
            require_text(review, field, case_no)
        for field, original in (("source_evidence", source), ("a_evidence", candidate_a), ("b_evidence", candidate_b)):
            quote = review[field]
            if quote and quote not in original:
                raise ValueError(f"case {case_no}: {field} is not a contiguous source substring")
        if candidate_a == candidate_b:
            if review["a_verdict"] != review["b_verdict"]:
                raise ValueError(f"case {case_no}: identical candidates have different verdicts")
            if review["winner"] not in {"TIE", "UNRESOLVED"}:
                raise ValueError(f"case {case_no}: identical candidates require TIE or UNRESOLVED")
        winner = review.get("winner")
        if winner not in WINNERS:
            raise ValueError(f"case {case_no}: invalid winner {winner!r}")
        if key["a_is"] not in {"qwen30", "qwen38"} or key["b_is"] not in {"qwen30", "qwen38"} or key["a_is"] == key["b_is"]:
            raise ValueError(f"case {case_no}: invalid A/B model mapping")
        source_row = sample_by_case[case_no]
        if source != source_row['input'] or key.get('id') != source_row['id']:
            raise ValueError(f"case {case_no}: blind source or key id differs from fixed sample")
        prediction_row = pred_by_id[source_row["id"]]
        expected_text = {"qwen30": source_row["baseline_target"], "qwen38": prediction_row["clean_text"]}
        if candidate_a != expected_text[key["a_is"]] or candidate_b != expected_text[key["b_is"]]:
            raise ValueError(f"case {case_no}: blind candidate text does not match key-mapped model outputs")
        grades = {key["a_is"]: review["a_verdict"], key["b_is"]: review["b_verdict"]}
        raw_winner_identity = identity_for(winner, key, case_no)
        lead = lead_by_case.get(case_no)
        final_grades = dict(grades)
        final_winner = raw_winner_identity
        main_note = review["reason"]
        if lead:
            final_grades = {"qwen30": lead["qwen30_verdict"], "qwen38": lead["qwen38_verdict"]}
            final_winner = lead["winner_identity"]
            main_note = lead["reason"]
        if key['identical'] and (final_grades['qwen30'] != final_grades['qwen38'] or final_winner not in {'TIE','UNRESOLVED'}):
            raise ValueError(f"case {case_no}: final adjudication contradicts identical candidates")
        for model in ("qwen30", "qwen38"):
            model_grades[model][final_grades[model]] += 1
        pairwise[final_winner] += 1
        identical["identical" if key["identical"] else "different"] += 1

        sid = sample_by_case[case_no].get("id")
        pred = pred_by_id.get(sid)
        if pred is None:
            raise ValueError(f"case {case_no}: missing generation record id={sid!r}")
        generation_error = bool(pred.get("error"))
        output_change["changed" if pred.get("clean_text") != sample_by_case[case_no].get("baseline_target") else "same"] += 1
        if not generation_error:
            valid_comparisons += 1
        output.append({
            **case,
            **review,
            "id": sid,
            "key_a_is": key["a_is"],
            "key_b_is": key["b_is"],
            "key_identical": key["identical"],
            "generation_error": pred.get("error"),
            "qwen30_review_verdict": grades["qwen30"],
            "qwen38_review_verdict": grades["qwen38"],
            "review_winner_identity": raw_winner_identity,
            "qwen30_final_verdict": final_grades["qwen30"],
            "qwen38_final_verdict": final_grades["qwen38"],
            "final_winner_identity": final_winner,
            "main_note": main_note,
        })

    report = {
        "requested_cases": 200,
        "generation_records": len(predictions),
        "generation_errors": errors,
        "valid_comparisons": valid_comparisons,
        "model_rating_distributions": {m: dict(sorted(c.items())) for m, c in model_grades.items()},
        "paired_wins": dict(pairwise),
        "candidate_outputs": dict(output_change),
        "blind_candidate_identity": dict(identical),
        "lead_adjudications": len(lead_by_case),
        "limitations": [
            "These are diagnostic samples from the training set, not an independent student test set.",
            "The models used different quantized weights and runtime environments, so this is not a fully isolated architecture comparison.",
            "Blind-review judgments are diagnostic and should not be interpreted as a population accuracy estimate.",
        ],
    }
    supplement_path = ROOT / 'report_supplement.json'
    if supplement_path.exists():
        supplement = json.loads(supplement_path.read_text())
        report['limitations'].extend(supplement.pop('additional_limitations', []))
        report.update(supplement)
    write_jsonl(ROOT / "comparison_final.jsonl", output)
    cols = list(output[0].keys()) if output else []
    with (ROOT / "comparison_final.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in output:
            writer.writerow({k: csv_safe(v) for k, v in row.items()})
    (ROOT / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
