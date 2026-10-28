"""Read FCGEC's native character-level edit annotations as auditable candidates.

FCGEC's ``Switch`` is a permutation of source character indices. The remaining
operation positions address the switched sentence, as in the official STG
preprocessor/converters. This adapter deliberately rejects ambiguous edits and
keeps their source record as an isolated candidate with a diagnostic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator


_CATEGORY_BY_NATIVE = {"CR": "C02", "CM": "C03", "IWO": "C04"}
_TAG_RE = re.compile(r"^(?:INS|MOD)_(\d+)(?:\+(?:INS|DEL)_\d+)?$")


def _label_text(label: Any, operation_name: str) -> str:
    # Official STG preprocessing consumes label[0] when label is a list. A
    # multi-item list therefore has ambiguous intent and must not be joined.
    if isinstance(label, list):
        if len(label) != 1 or not isinstance(label[0], str):
            raise ValueError(f"{operation_name} label list must contain exactly one string")
        return label[0]
    if isinstance(label, str):
        return label
    raise ValueError(f"{operation_name} label must be a string or one-item string list")


def _apply_reference(source: str, operation: dict[str, Any]) -> str:
    """Apply one native reference, validating all indices before rendering."""
    allowed = {"Switch", "Delete", "Insert", "Modify"}
    unknown = set(operation) - allowed
    if unknown:
        raise ValueError(f"unsupported operation keys: {sorted(unknown)}")

    switched = source
    if "Switch" in operation:
        order = operation["Switch"]
        if (not isinstance(order, list) or len(order) != len(source)
                or any(type(i) is not int for i in order)
                or sorted(order) != list(range(len(source)))):
            raise ValueError(f"Switch must be a permutation of 0..{len(source)-1}")
        switched = "".join(source[i] for i in order)

    deleted: set[int] = set()
    for pos in operation.get("Delete", []):
        if type(pos) is not int or not 0 <= pos < len(switched):
            raise ValueError(f"Delete index {pos!r} outside switched sentence length {len(switched)}")
        if pos in deleted:
            raise ValueError(f"duplicate Delete index {pos}")
        deleted.add(pos)

    inserts: dict[int, list[str]] = {}
    for edit in operation.get("Insert", []):
        if not isinstance(edit, dict) or type(edit.get("pos")) is not int:
            raise ValueError("Insert requires an integer pos")
        pos = edit["pos"]
        if not -1 <= pos < len(switched):
            raise ValueError(f"Insert pos {pos} outside -1..{len(switched)-1}")
        value = _label_text(edit.get("label"), "Insert")
        tag = edit.get("tag")
        match = _TAG_RE.fullmatch(tag) if isinstance(tag, str) else None
        if not match or not tag.startswith("INS_") or int(match.group(1)) != len(value):
            raise ValueError(f"Insert tag/label length mismatch or invalid tag: {tag!r}")
        inserts.setdefault(pos + 1, []).append(value)

    modifications: dict[int, tuple[int, str]] = {}
    for edit in operation.get("Modify", []):
        if not isinstance(edit, dict) or type(edit.get("pos")) is not int:
            raise ValueError("Modify requires an integer pos")
        pos, tag, label = edit["pos"], edit.get("tag"), edit.get("label")
        match = _TAG_RE.fullmatch(tag) if isinstance(tag, str) else None
        if not match or not tag.startswith("MOD_"):
            raise ValueError(f"invalid Modify tag: {tag!r}")
        span = int(match.group(1))
        if span <= 0:
            raise ValueError(f"Modify span must be positive, got {span}")
        value = _label_text(label, "Modify")
        if pos < 0 or pos + span > len(switched):
            raise ValueError(f"Modify span [{pos}, {pos + span}) outside switched sentence length {len(switched)}")
        if pos in modifications:
            raise ValueError(f"duplicate Modify start index {pos}")
        suffix = tag[len(f"MOD_{span}"):]
        if suffix:
            kind, count = suffix[1:].split("_")
            delta = int(count)
            expected = span + delta if kind == "INS" else span - delta
            if expected != len(value) or expected < 0:
                raise ValueError(f"Modify tag {tag} conflicts with span/label length {span}/{len(value)}")
        elif len(value) != span:
            raise ValueError(f"Modify tag {tag} conflicts with label length {len(value)}")
        modifications[pos] = (span, value)

    occupied: set[int] = set()
    for pos, (span, _) in modifications.items():
        cells = set(range(pos, pos + span))
        if occupied & cells or deleted & cells:
            raise ValueError(f"overlapping Modify/Delete operation at index {pos}")
        occupied.update(cells)
    for boundary in inserts:
        if any(pos < boundary < pos + span for pos, (span, _) in modifications.items()):
            raise ValueError(f"Insert boundary {boundary - 1} falls inside a Modify span")

    out: list[str] = []
    i = 0
    while i <= len(switched):
        out.extend(inserts.get(i, []))
        if i == len(switched):
            break
        if i in modifications:
            span, value = modifications[i]
            out.append(value)
            i += span
        elif i in occupied:
            # A modification began before this position.
            i += 1
        elif i not in deleted:
            out.append(switched[i])
            i += 1
        else:
            i += 1
    return "".join(out)


def _category(error_type: Any, error_flag: Any) -> tuple[str, str | None]:
    flag_is_correct = error_flag in (0, False, "0", "false", "False")
    flag_is_error = error_flag in (1, True, "1", "true", "True")
    type_is_correct = error_type == "*"
    if type_is_correct and flag_is_error:
        return "pending", "conflicting metadata: error_type='*' but error_flag indicates an error"
    if flag_is_correct and error_type is not None and not type_is_correct:
        return "pending", f"conflicting metadata: error_flag indicates correct but error_type={error_type!r}"
    if type_is_correct or flag_is_correct:
        return "E07", None
    if not isinstance(error_type, str):
        return "pending", None
    labels = [part.strip() for part in error_type.split(";") if part.strip()]
    if len(labels) == 1:
        return _CATEGORY_BY_NATIVE.get(labels[0], "pending"), None
    return "pending", None


def iter_fcgec_candidates(path: str | Path, revision: str, split: str) -> Iterator[dict[str, Any]]:
    """Yield one candidate per native reference, preserving groups and labels.

    Each yielded record includes ``valid`` and ``diagnostic``. Invalid or
    unsupported operations remain present with ``reference=None`` and an
    explicit diagnostic, so callers can count/quarantine them without silently
    assigning a corrupt target.
    """
    with Path(path).open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, dict):
        raise ValueError("FCGEC file root must be a JSON object keyed by source id")
    for source_id, row in records.items():
        if not isinstance(row, dict) or not isinstance(row.get("sentence"), str):
            yield {
                "id": str(source_id), "source_id": str(source_id), "group_id": str(source_id),
                "dataset": "FCGEC", "revision": revision, "original_split": split,
                "input": row.get("sentence") if isinstance(row, dict) else None,
                "reference": None, "category": "pending", "native_labels": row.get("error_type") if isinstance(row, dict) else None,
                "operation": None, "reference_index": None, "valid": False,
                "diagnostic": "record missing a string sentence",
            }
            continue
        source = row["sentence"]
        native = row.get("error_type")
        category, metadata_diagnostic = _category(native, row.get("error_flag"))
        raw_operations = row.get("operation", "[]")
        try:
            operations = json.loads(raw_operations) if isinstance(raw_operations, str) else raw_operations
            if not isinstance(operations, list):
                raise ValueError("operation must decode to a list of references")
        except (TypeError, json.JSONDecodeError, ValueError) as exc:
            operations = [None]
            parse_diagnostic = f"invalid operation field: {exc}"
        else:
            parse_diagnostic = None
        # Correct examples can have an empty operation list; retain one row.
        refs = operations if operations else ([{}] if category == "E07" else [None])
        for reference_index, operation in enumerate(refs):
            diagnostic = parse_diagnostic or metadata_diagnostic
            reference = None
            if diagnostic is None:
                if not isinstance(operation, dict):
                    diagnostic = "reference operation is not an object"
                else:
                    try:
                        reference = _apply_reference(source, operation)
                    except (TypeError, ValueError, KeyError) as exc:
                        diagnostic = str(exc)
            yield {
                "id": f"{source_id}::ref{reference_index}",
                "source_id": str(source_id), "group_id": str(source_id),
                "dataset": "FCGEC", "revision": revision, "original_split": split,
                "input": source, "reference": reference, "category": category,
                "native_labels": native, "operation": operation,
                "reference_index": reference_index, "valid": diagnostic is None,
                "diagnostic": diagnostic,
            }
