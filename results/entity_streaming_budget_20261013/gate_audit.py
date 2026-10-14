#!/usr/bin/env python3
"""Offline score/margin gate audit on the frozen 60-row development plan only."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "results/entity_modes_dataset_20261013"
sys.path.insert(0, str(ROOT))
from system.entity_audio_repair import GenericEntitySelector
from system.entity_store import EntityDefinition

SCORE_THRESHOLDS = (0.63, 0.72, 0.80, 0.90)
MARGIN_THRESHOLDS = (0.00, 0.04, 0.08, 0.12)


def read_json(path):
    return json.loads(Path(path).read_text())


def norm(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(c for c in value if unicodedata.category(c)[0] not in {"P", "Z", "C"})


def definitions_readonly(key):
    """Load selector inputs through a read-only SQLite connection."""
    db = (DATA / "library_snapshots" / f"{key}.db").resolve()
    uri = db.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""SELECT e.id,e.canonical_text,e.entity_type,e.domain,
            e.normalization_policy,e.priority,e.enabled,e.source,e.created_at,e.updated_at,a.alias
            FROM protected_entities e LEFT JOIN entity_aliases a ON a.entity_id=e.id
            ORDER BY e.priority DESC,e.id ASC,a.id ASC""").fetchall()
    grouped = {}
    for row in rows:
        entry = grouped.setdefault(row["id"], dict(
            canonical_text=row["canonical_text"], entity_type=row["entity_type"], domain=row["domain"],
            normalization_policy=row["normalization_policy"], priority=row["priority"],
            enabled=bool(row["enabled"]), source=row["source"], created_at=row["created_at"],
            updated_at=row["updated_at"], aliases=[]))
        if row["alias"] is not None:
            entry["aliases"].append(row["alias"])
    return tuple(EntityDefinition(entity_id=int(i), aliases=tuple(v.pop("aliases")),
                                  spelling_context=None, **v) for i, v in grouped.items())


def changes_for(row):
    result = row.get("result", {})
    repair = result.get("entity_audio_repair") or {}
    return str(result.get("raw_text", "")), list(repair.get("changes") or [])


def reconstruct(original_final, changes):
    """Reverse replacements using untouched source-run lengths; enforce exact forward replay."""
    ordered = sorted(changes, key=lambda c: (int(c["start"]), int(c["end"])))
    pieces, cursor, source_cursor = [], 0, 0
    for change in ordered:
        start, end = int(change["start"]), int(change["end"])
        if start < source_cursor or end < start:
            raise ValueError("overlapping or invalid source spans")
        unchanged = start - source_cursor
        pieces.append(original_final[cursor:cursor + unchanged])
        cursor += unchanged
        canonical = str(change["canonical"])
        observed = str(change["observed"])
        if original_final[cursor:cursor + len(canonical)] != canonical:
            raise ValueError("corrected text does not contain expected canonical at reconstructed span")
        pieces.append(observed)
        cursor += len(canonical)
        source_cursor = end
    pieces.append(original_final[cursor:])
    original = "".join(pieces)
    forward, cursor = [], 0
    for change in ordered:
        start, end = int(change["start"]), int(change["end"])
        if original[start:end] != str(change["observed"]):
            raise ValueError("reconstructed original span does not match observed")
        forward.extend((original[cursor:start], str(change["canonical"])))
        cursor = end
    forward.append(original[cursor:])
    if "".join(forward) != original_final:
        raise ValueError("forward application of changes does not reproduce final.raw_text")
    return original


def plan_ids(plan):
    for key in ("sample_ids", "development_ids", "ids"):
        if isinstance(plan.get(key), list):
            return [str(x.get("id")) if isinstance(x, dict) else str(x) for x in plan[key]]
    for key in ("samples", "development", "recordings"):
        value = plan.get(key)
        if isinstance(value, list):
            return [str(x.get("id", x.get("sample_id"))) if isinstance(x, dict) else str(x) for x in value]
    raise ValueError("plan.json has no recognized development ID list")


def run():
    plan_path = OUT / "frozen_test_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"waiting for prepared development plan: {plan_path}")
    plan = read_json(plan_path)
    if isinstance(plan.get("samples"), list) and any("split" in x for x in plan["samples"] if isinstance(x, dict)):
        development = [x for x in plan["samples"] if isinstance(x, dict) and x.get("split") == "development"]
        if len(development) != 60 or plan.get("development_count") != 60:
            raise ValueError(f"expected frozen 60-row development split, got {len(development)}")
        ids = [str(x["id"]) for x in development]
    else:
        ids = plan_ids(plan)
    if len(ids) != 60 or len(set(ids)) != 60:
        raise ValueError(f"expected exactly 60 unique plan IDs, found {len(ids)}")
    references = read_json(DATA / "references.json")["samples"]
    by_id = {}
    for line in (DATA / "results.jsonl").open():
        row = json.loads(line)
        if row.get("arm") == "audio" and row["id"] in ids:
            if row["id"] in by_id:
                raise ValueError(f"duplicate audio row for {row['id']}")
            by_id[row["id"]] = row
    missing = sorted(set(ids) - by_id.keys())
    if missing:
        raise ValueError(f"plan audio rows missing: {missing[:5]} ({len(missing)} total)")

    selectors = {}
    sample_details, grid = [], {}
    aggregate_preserved = defaultdict(int)
    reference_old = defaultdict(int)
    errors = []
    for sid in ids:
        row = by_id[sid]
        key = row["library_key"]
        if key not in selectors:
            selectors[key] = GenericEntitySelector(definitions_readonly(key))
        final, changes = changes_for(row)
        try:
            original = reconstruct(final, changes)
        except Exception as exc:
            errors.append({"id": sid, "error": str(exc)})
            continue
        candidates, _ = selectors[key].select(original)
        scored = []
        for candidate in candidates:
            overlap_changes = [c for c in changes if int(candidate["start"]) < int(c["end"]) and int(c["start"]) < int(candidate["end"])]
            competitor_scores = [float(other["retrieval_score"]) for other in candidates
                                 if other["canonical"] != candidate["canonical"]
                                 and int(other["start"]) < int(candidate["end"])
                                 and int(candidate["start"]) < int(other["end"])]
            margin = float(candidate["retrieval_score"]) - max(competitor_scores) if competitor_scores else 1.0
            scored.append({"start": int(candidate["start"]), "end": int(candidate["end"]),
                           "observed": candidate["observed"], "canonical": candidate["canonical"],
                           "score": float(candidate["retrieval_score"]), "margin": margin,
                           "overlaps_old_change": bool(overlap_changes)})
        sample_details.append({"id": sid, "library_key": key, "original": original,
                               "final_raw_text": final, "old_changes": changes, "candidates": scored})
        for st in SCORE_THRESHOLDS:
            for mt in MARGIN_THRESHOLDS:
                name = f"score>={st:.2f},margin>={mt:.2f}"
                allowed = [c for c in scored if c["score"] >= st and c["margin"] >= mt]
                cell = grid.setdefault(name, dict(score_threshold=st, margin_threshold=mt,
                    qualifying_samples=0, qualifying_candidates=0, old_confirmations_retained=0,
                    old_confirmations_lost=0, reference_present_old_changes=0,
                    reference_mismatch_old_changes=0))
                if allowed:
                    cell["qualifying_samples"] += 1
                cell["qualifying_candidates"] += len(allowed)
                for change in changes:
                    pos_matches = [c for c in allowed if c["start"] == int(change["start"])
                                   and c["end"] == int(change["end"])
                                   and c["canonical"] == change["canonical"]]
                    cell["old_confirmations_retained" if pos_matches else "old_confirmations_lost"] += 1
                    ref = norm(references[sid]["reference"])
                    present = norm(change["canonical"]) in ref
                    cell["reference_present_old_changes" if present else "reference_mismatch_old_changes"] += 1

    if errors:
        output = {"status": "reconstruction_errors", "errors": errors,
                  "analyzed_samples": len(sample_details), "requested_samples": len(ids),
                  "note": "Offline retention estimate only; prompt changes require actual ASR validation. No speed or accuracy claim."}
    else:
        output = {"status": "complete", "scope": "prepared 60-row development plan only",
                  "requested_samples": len(ids), "analyzed_samples": len(sample_details),
                  "score_thresholds": SCORE_THRESHOLDS, "margin_thresholds": MARGIN_THRESHOLDS,
                  "counts_by_gate": list(grid.values()), "samples": sample_details,
                  "reference_counts_are_scoring_only": True,
                  "interpretation": "Offline retention estimate only. Prompt changes require actual ASR validation; these counts do not establish real speed gains or accuracy."}
    (OUT / "gate_development.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "analyzed_samples": output["analyzed_samples"],
                      "counts_by_gate": output.get("counts_by_gate"), "errors": errors}, ensure_ascii=False))


if __name__ == "__main__":
    run()
