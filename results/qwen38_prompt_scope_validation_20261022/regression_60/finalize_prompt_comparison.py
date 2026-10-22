#!/usr/bin/env python3
"""Validate blind prompt reviews and assemble the fixed 60-case comparison."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
VERDICTS = {"PASS", "MINOR", "FAIL", "AMBIGUOUS"}
WINNERS = {"A", "B", "TIE", "UNRESOLVED"}
IDENTITIES = {"old_prompt", "new_prompt", "TIE", "UNRESOLVED"}
REVIEW_FILES = ("blind_review_01.jsonl", "blind_review_02.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line: {path.name}:{line_no}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object: {path.name}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def keyed(rows: list[dict[str, Any]], label: str, field: str = "case_no") -> dict[int, dict[str, Any]]:
    result = {}
    for row in rows:
        n = row.get(field)
        if not isinstance(n, int) or isinstance(n, bool) or n in result:
            raise ValueError(f"Invalid or duplicate {field} in {label}: {n!r}")
        result[n] = row
    return result


def text(row: dict[str, Any], field: str, case_no: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"case {case_no}: {field} must be a string")
    return value


def verdict(value: Any, label: str) -> str:
    if value not in VERDICTS:
        raise ValueError(f"{label}: invalid verdict {value!r}")
    return value


def csv_safe(value: Any) -> Any:
    if isinstance(value, str) and (value[:1] in ("\t", "\r") or value.lstrip(" \t\r\n")[:1] in ("=", "+", "-", "@")):
        return "'" + value
    return value


def main() -> None:
    sample_rows = read_jsonl(ROOT / "sample_60.jsonl")
    sample = keyed(sample_rows, "sample_60.jsonl")
    if len(sample) != 60:
        raise ValueError(f"Expected 60 sample records, got {len(sample)}")
    by_id: dict[str, dict[str, Any]] = {}
    for case_no, row in sample.items():
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in by_id:
            raise ValueError(f"Invalid or duplicate sample id: {rid!r}")
        by_id[rid] = row

    predictions = read_jsonl(ROOT / "predictions_qwen38.jsonl")
    pred_by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in pred_by_id:
            raise ValueError(f"Missing or duplicate prediction id: {rid!r}")
        if rid not in by_id:
            raise ValueError(f"Unexpected prediction id: {rid!r}")
        source = by_id[rid]
        if row.get("case_no") != source["case_no"] or row.get("input") != source.get("input"):
            raise ValueError(f"Prediction case_no/input mismatch for id={rid}")
        if row.get("baseline_target") != source.get("baseline_target"):
            raise ValueError(f"Prediction baseline_target mismatch for id={rid}")
        pred_by_id[rid] = row
    if set(pred_by_id) != set(by_id):
        raise ValueError("predictions_qwen38.jsonl must contain exactly one record for each of 60 sample ids")

    valid = {n for n, source in sample.items() if
             pred_by_id[source["id"]].get("error") in (None, "") and
             isinstance(pred_by_id[source["id"]].get("clean_text"), str) and
             pred_by_id[source["id"]]["clean_text"].strip()}
    cases = keyed(read_jsonl(ROOT / "blind_cases.jsonl"), "blind_cases.jsonl")
    keys = keyed(read_jsonl(ROOT / "blind_key.jsonl"), "blind_key.jsonl")
    if set(cases) != valid:
        raise ValueError(f"blind_cases coverage differs from valid generations; missing={sorted(valid-set(cases))}, extra={sorted(set(cases)-valid)}")
    if set(keys) != valid:
        raise ValueError("blind_key coverage differs from valid generations")

    reviews: dict[int, dict[str, Any]] = {}
    for filename in REVIEW_FILES:
        for row in read_jsonl(ROOT / filename):
            n = row.get("case_no")
            if not isinstance(n, int) or isinstance(n, bool) or n in reviews:
                raise ValueError(f"Invalid or duplicate review case_no in {filename}: {n!r}")
            reviews[n] = row
    if set(reviews) != valid:
        raise ValueError(f"review coverage differs from valid generations; missing={sorted(valid-set(reviews))}, extra={sorted(set(reviews)-valid)}")

    leads: dict[int, dict[str, Any]] = {}
    lead_path = ROOT / "lead_adjudications.jsonl"
    if lead_path.exists():
        for row in read_jsonl(lead_path):
            n = row.get("case_no")
            if not isinstance(n, int) or isinstance(n, bool) or n not in valid or n in leads:
                raise ValueError(f"Invalid or duplicate lead adjudication case_no: {n!r}")
            verdict(row.get("old_prompt_verdict"), f"lead case {n} old_prompt_verdict")
            verdict(row.get("new_prompt_verdict"), f"lead case {n} new_prompt_verdict")
            if row.get("winner_identity") not in IDENTITIES:
                raise ValueError(f"case {n}: invalid lead winner_identity")
            text(row, "reason", n)
            leads[n] = row

    manifest = json.loads((ROOT / "sample_manifest.json").read_text(encoding="utf-8"))
    controls = set(manifest.get("control_case_numbers", []))
    out = []
    ratings = {"old_prompt": Counter(), "new_prompt": Counter()}
    wins, sameness, changes = Counter(), Counter(), Counter()
    control_counts = Counter()
    for n in sorted(valid):
        case, key, review, source = cases[n], keys[n], reviews[n], sample[n]
        pred = pred_by_id[source["id"]]
        if key.get("id") != source["id"] or case.get("input") != source.get("input"):
            raise ValueError(f"case {n}: blind source/id differs from fixed sample")
        old_text, new_text = source.get("baseline_target"), pred["clean_text"]
        if not isinstance(old_text, str):
            raise ValueError(f"case {n}: baseline_target must be a string")
        a, b = text(case, "candidate_a", n), text(case, "candidate_b", n)
        aliases = {"old_prompt": "old_prompt", "new_prompt": "new_prompt", "qwen30": "old_prompt", "qwen38": "new_prompt"}
        if key.get("a_is") not in aliases or key.get("b_is") not in aliases or aliases[key["a_is"]] == aliases[key["b_is"]]:
            raise ValueError(f"case {n}: invalid old/new blind key mapping")
        if key.get("identical") not in (True, False) or ((a == b) != key["identical"]):
            raise ValueError(f"case {n}: blind_key identical disagrees with candidates")
        expected = {"old_prompt": old_text, "new_prompt": new_text}
        a_identity, b_identity = aliases[key["a_is"]], aliases[key["b_is"]]
        if a != expected[a_identity] or b != expected[b_identity]:
            raise ValueError(f"case {n}: candidate text does not match the mapped old/new outputs")
        if case.get("case_no") != n:
            raise ValueError(f"case {n}: blind case number mismatch")
        for field in ("a_verdict", "b_verdict"):
            verdict(review.get(field), f"case {n} {field}")
        for field in ("reason", "source_evidence", "a_evidence", "b_evidence"):
            text(review, field, n)
        for field, original in (("source_evidence", source["input"]), ("a_evidence", a), ("b_evidence", b)):
            if review[field] and review[field] not in original:
                raise ValueError(f"case {n}: {field} is not a contiguous substring")
        winner = review.get("winner")
        if winner not in WINNERS:
            raise ValueError(f"case {n}: invalid winner {winner!r}")
        if a == b and (review["a_verdict"] != review["b_verdict"] or winner not in {"TIE", "UNRESOLVED"}):
            raise ValueError(f"case {n}: identical candidates require equal grades and TIE/UNRESOLVED")
        grades = {a_identity: review["a_verdict"], b_identity: review["b_verdict"]}
        raw_winner = {"A": a_identity, "B": b_identity, "TIE": "TIE", "UNRESOLVED": "UNRESOLVED"}[winner]
        final_grades, final_winner, main_note = dict(grades), raw_winner, review["reason"]
        lead = leads.get(n)
        if lead:
            final_grades = {"old_prompt": lead["old_prompt_verdict"], "new_prompt": lead["new_prompt_verdict"]}
            final_winner, main_note = lead["winner_identity"], lead["reason"]
        if a == b and (final_grades["old_prompt"] != final_grades["new_prompt"] or final_winner not in {"TIE", "UNRESOLVED"}):
            raise ValueError(f"case {n}: final adjudication contradicts identical candidates")
        for identity in ratings:
            ratings[identity][final_grades[identity]] += 1
        wins[final_winner] += 1
        sameness["identical" if key["identical"] else "different"] += 1
        changed = new_text != old_text
        changes["changed" if changed else "same"] += 1
        if n in controls:
            control_counts["reviewed"] += 1
            control_counts["changed" if changed else "same"] += 1
            if final_grades["new_prompt"] != "PASS":
                control_counts["new_prompt_not_pass"] += 1
            if final_grades["old_prompt"] != "PASS":
                control_counts["old_prompt_not_pass"] += 1
        out.append({**case, **review, "id": source["id"], "key_a_is": a_identity, "key_b_is": b_identity,
                    "key_identical": key["identical"], "generation_error": pred.get("error"),
                    "old_prompt_review_verdict": grades["old_prompt"], "new_prompt_review_verdict": grades["new_prompt"],
                    "review_winner_identity": raw_winner, "old_prompt_final_verdict": final_grades["old_prompt"],
                    "new_prompt_final_verdict": final_grades["new_prompt"], "final_winner_identity": final_winner,
                    "main_note": main_note, "new_prompt_changed_output": changed, "is_control": n in controls,
                    "lead_adjudicated": bool(lead)})

    fail_92_162 = {}
    for n in (92, 162):
        row = next((r for r in out if r["case_no"] == n), None)
        fail_92_162[str(n)] = ({"old_prompt": row["old_prompt_final_verdict"], "new_prompt": row["new_prompt_final_verdict"]} if row else None)
    report = {
        "requested_cases": 60, "generation_records": len(predictions),
        "generation_errors": len(predictions) - len(valid), "valid_blind_comparisons": len(valid),
        "same_model": True, "model": "Qwen3.8-27B-AWQ-INT4",
        "comparison": "Same model and source; old_prompt and new_prompt identify prompt variants only.",
        "rating_distributions": {k: dict(sorted(v.items())) for k, v in ratings.items()},
        "paired_wins": dict(wins), "candidate_text_identity": dict(sameness), "new_prompt_output_changes": dict(changes),
        "control_cases": {"manifest_count": len(controls), "valid_reviewed": control_counts["reviewed"],
                          "new_prompt_changed": control_counts["changed"], "new_prompt_unchanged": control_counts["same"],
                          "new_prompt_not_pass": control_counts["new_prompt_not_pass"],
                          "old_prompt_not_pass": control_counts["old_prompt_not_pass"]},
        "known_failure_cases_92_162_final_ratings": fail_92_162,
        "lead_adjudications": len(leads),
        "limitations": ["这是提示词开发诊断子集，不是独立测试集；结果不能解释为总体准确率或总体改进率。",
                        "old_prompt 与 new_prompt 是同一 Qwen3.8-27B-AWQ-INT4 模型的提示词版本标签，不代表不同教师或不同模型。",
                        "盲审评级用于该固定样本的诊断；选择机制包含预设诊断案例和控制案例。"],
    }
    supplement_path = ROOT / 'report_supplement.json'
    if supplement_path.exists():
        supplement = json.loads(supplement_path.read_text())
        report['limitations'].extend(supplement.pop('additional_limitations', []))
        report['control_cases'].update(supplement.pop('controls_changed_from_input', {}))
        report.update(supplement)
    write_jsonl(ROOT / "comparison_final.jsonl", out)
    columns = list(out[0]) if out else []
    with (ROOT / "comparison_final.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in out:
            writer.writerow({k: csv_safe(v) for k, v in row.items()})
    (ROOT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
