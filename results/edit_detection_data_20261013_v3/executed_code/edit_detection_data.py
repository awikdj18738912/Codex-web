"""Source-coordinate edit supervision; no model response is treated as gold."""
from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json

TASKS = ("disfluency", "repetition", "self_correction", "local_typo",
         "boundary_punctuation", "entity_correction")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def apply_edits(source, edits):
    cursor = 0
    parts = []
    for edit in edits:
        start, end = edit["start"], edit["end"]
        if type(start) is not int or type(end) is not int or not cursor <= start <= end <= len(source):
            raise ValueError("Invalid or overlapping source offsets")
        if source[start:end] != edit["source"]:
            raise ValueError("Edit source differs from input")
        parts.extend((source[cursor:start], edit["target"]))
        cursor = end
    return "".join((*parts, source[cursor:]))


def align_edits(source, target, task=None):
    """Deterministic Unicode code-point offsets, including insertion gaps.

    Repeated strings may have equally valid offsets. We record alternatives
    for single edits rather than pretending the canonical alignment is unique.
    """
    if task is not None and task not in TASKS:
        raise ValueError("Unknown task")
    edits = []
    for operation, start, end, left, right in SequenceMatcher(None, source, target, autojunk=False).get_opcodes():
        if operation == "equal":
            continue
        replacement = target[left:right]
        edits.append(dict(start=start, end=end, source=source[start:end], target=replacement,
                          kind="DELETE_CANDIDATE" if not replacement else "CORRECTION_CANDIDATE",
                          operation=operation, task=task))
    if apply_edits(source, edits) != target:
        raise ValueError("Alignment does not reconstruct target")
    tags = ["KEEP"] * len(source)
    gaps = []
    for edit in edits:
        if edit["start"] == edit["end"]:
            gaps.append(dict(offset=edit["start"], kind="CORRECTION_CANDIDATE"))
        else:
            tags[edit["start"]:edit["end"]] = [edit["kind"]] * (edit["end"] - edit["start"])
    alternatives = []
    if len(edits) == 1:
        edit = edits[0]
        width = edit["end"] - edit["start"]
        alternatives = [dict(start=start, end=start + width)
                        for start in range(len(source) - width + 1)
                        if source[start:start + width] == edit["source"]
                        and source[:start] + edit["target"] + source[start + width:] == target]
    checked = len(edits) <= 1
    return dict(edits=edits, character_tags=tags, insertion_gaps=gaps,
                offset_alternatives=alternatives,
                ambiguous_offsets=(len(alternatives) > 1) if checked else None,
                alignment_ambiguity_checked=checked,
                offset_unit="unicode_code_point", interval="[start,end)")


def review_basis(record):
    return digest({key: record.get(key) for key in
                   ("id", "raw_asr", "baseline_text", "entity_candidates", "entity_hints", "protection",
                    "available_context", "left_context", "right_context", "prepared_recording_text",
                    "recording_group", "split", "split_group", "audio_path", "audio_sha256", "window_pipeline")})


def attach_review(record, review):
    """Explicit reviews cannot promote old data into a fresh test set."""
    if review["id"] != record["id"] or review["review_basis_sha256"] != review_basis(record):
        raise ValueError("Stale review or mismatched record")
    kind = review["review_kind"]
    if kind not in {"assistant_text_only", "human_audio_cleaned"}:
        raise ValueError("Unknown review kind")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("Reviewer required")
    if type(review.get("audio_checked")) is not bool:
        raise ValueError("Explicit audio_checked boolean required")
    if kind == "assistant_text_only" and review["audio_checked"]:
        raise ValueError("Text review cannot claim an audio check")
    if kind == "human_audio_cleaned" and (not review["audio_checked"]
                                            or not record.get("audio_path") or not record.get("audio_sha256")):
        raise ValueError("Audio review requires recording provenance and an audio check")
    target = review["desired_text"]
    if not isinstance(target, str):
        raise ValueError("Reviewed target must be a string")
    from .refinement_guard import restore_rule_numeric_surfaces

    if restore_rule_numeric_surfaces(record["baseline_text"], target) != target:
        raise ValueError("Reviewed edit changes rule-owned numeric values or surfaces")
    aligned = align_edits(record["baseline_text"], target)
    tasks = review["edit_tasks"]
    if len(tasks) != len(aligned["edits"]) or any(task not in TASKS for task in tasks):
        raise ValueError("Every reviewed edit requires a supported task")
    for edit, task in zip(aligned["edits"], tasks):
        edit["task"] = task
    audio_gold = kind == "human_audio_cleaned"
    return {**record, **aligned, "desired_text": target, "review": review,
            "label_status": "verified_audio_cleanup" if audio_gold else "assistant_text_reviewed",
            "human_audio_gold": audio_gold,
            "training_eligible": audio_gold and record["split"] == "train",
            "independent_acceptance_eligible": False}
