"""Focused regressions for safe cleanup of pronunciation notes and restarts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from src.utils.english_entity_policy import english_entities_preserved  # noqa: E402


def test_explicit_pinyin_explanation_can_be_removed():
    assert english_entities_preserved(
        "名字是林舟，拼音是Lin Zhou。", "名字是林舟。"
    )


def test_removing_pinyin_does_not_remove_an_independent_english_name():
    assert not english_entities_preserved(
        "名字是林舟，英文名是Lin Zhou。", "名字是林舟。"
    )


def test_filler_restart_can_drop_abandoned_entity_phrase():
    assert english_entities_preserved(
        "嗯，用FastAPI，呃，用FastAPI做接口。", "用FastAPI做接口。"
    )


def test_meaningful_entity_emphasis_and_independent_fact_loss_stay_rejected():
    assert not english_entities_preserved(
        "PyTorch，PyTorch，真的很重要。", "PyTorch，真的很重要。"
    )
    assert not english_entities_preserved(
        "我用FastAPI做服务，也用FastAPI做脚本。", "我用FastAPI做服务。"
    )


def test_correction_and_ordered_exact_preservation_remain_supported():
    assert english_entities_preserved(
        "FastAPI不对，改成Flask。", "用Flask。"
    )
    assert not english_entities_preserved(
        "先PyTorch再FastAPI。", "先FastAPI再PyTorch。"
    )
