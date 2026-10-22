"""Admission checks for deterministic synthetic oral construction traces."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from src.utils.numeric_policy import (  # noqa: E402
    lock_training_target,
    normalize_source,
    prepare_training_pair,
)
from src.utils.synthetic_oral import build_oral  # noqa: E402


def plan(intent, values=(), phenomena=("oral:low",), parts=()):
    return {
        "construction_version": "synthetic-oral-v1",
        "raw_intent": intent,
        "construction_phenomena": list(phenomena),
        "correction_values": list(values),
        "spelling_parts": list(parts),
        "expected_clean": normalize_source(intent)[0],
    }


def pair_for(intent, values=(), phenomena=("oral:low",), parts=()):
    metadata = plan(intent, values, phenomena, parts)
    source = build_oral(intent, list(values), list(parts), list(phenomena))
    return source, metadata


@pytest.mark.parametrize("values", [("A", "B", "A"), ("A", "B", "C")])
def test_constructed_multiple_and_rollback_corrections_admit_exact_target(values):
    intent = "采用" + values[-1] + "方案。"
    source, metadata = pair_for(intent, values)
    assert lock_training_target(source, intent, metadata) == intent
    prepared = prepare_training_pair(source, intent, metadata)
    assert prepared is not None
    assert prepared[1] == intent
    assert prepared[2]["english_entity_policy"] == "verbatim-v1"


def test_numeric_construction_does_not_itn_or_change_values():
    intent = "明天3点开会。"
    source, metadata = pair_for(intent, ("2", "3"))
    normalized_source, _ = normalize_source(source)
    assert lock_training_target(normalized_source, intent, metadata) == intent
    assert lock_training_target(normalized_source, "明天4点开会。", metadata) is None


def test_rejects_source_mismatch_target_rewrite_and_bad_version():
    source, metadata = pair_for("用FastAPI训练。")
    assert lock_training_target(source + "额外", "用FastAPI训练。", metadata) is None
    assert lock_training_target(source, "用Flask训练。", metadata) is None
    bad = dict(metadata, construction_version="synthetic-oral-v2")
    assert lock_training_target(source, "用FastAPI训练。", bad) is None


@pytest.mark.parametrize("field,value", [
    ("raw_intent", 3), ("construction_phenomena", "oral:low"),
    ("correction_values", ["A", 2]), ("spelling_parts", [{"word": 3}]),
    ("expected_clean", None),
])
def test_invalid_plan_metadata_is_rejected(field, value):
    source, metadata = pair_for("用FastAPI训练。")
    metadata[field] = value
    assert lock_training_target(source, "用FastAPI训练。", metadata) is None


def test_metadata_must_match_reconstructed_source():
    source, metadata = pair_for("采用A方案。", ("A", "B", "A"))
    metadata["correction_values"] = ["A", "C", "A"]
    assert lock_training_target(source, "采用A方案。", metadata) is None


def test_plan_without_construction_version_keeps_legacy_guard():
    assert lock_training_target("有3个苹果。", "有4个苹果。", {"raw_intent": "有4个苹果。"}) is None
    assert lock_training_target("用FastAPI训练", "用Pytorch训练") is None
