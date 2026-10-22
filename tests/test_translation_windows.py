"""Pure fake-translator tests for stable incremental translation windows."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

WEB_ROOT = Path(__file__).resolve().parents[1]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

from system.translation_windows import StableTranslationSnapshots, source_windows


def test_three_sentence_groups_are_stable_and_keep_shared_context():
    calls = []
    snapshots = StableTranslationSnapshots(lambda text, lang: calls.append((text, lang)) or f"译:{text}")
    source = "A. B. C. D."

    first, _ = snapshots.update(source, "en")
    assert first == "译:A. B. C.\n译:D."
    assert snapshots.stable_translation == "译:A. B. C."
    assert snapshots.active_translation == "译:D."
    assert snapshots.last_stats["stable_windows"] == 1
    second, _ = snapshots.update(source + " E.", "en")
    assert second == "译:A. B. C.\n译:D. E."
    assert calls == [("A. B. C. ", "en"), ("D.", "en"), ("D. E.", "en")]
    assert snapshots.last_stats["model_calls"] == 1
    assert snapshots.last_stats["cache_hits"] == 1


def test_old_group_revision_uses_exact_new_source_and_replaces_cache():
    calls = []
    snapshots = StableTranslationSnapshots(lambda text, lang: calls.append(text) or f"译:{text}")
    snapshots.update("A. B. C. D.", "en")
    revised, _ = snapshots.update("A. X. C. D.", "en")
    assert revised == "译:A. X. C.\n译:D."
    assert calls == ["A. B. C. ", "D.", "A. X. C. "]
    assert set(snapshots.cache) == {("en", "A. X. C. "), ("en", "D.")}


def test_active_tail_reuses_exact_snapshot_cache_and_language_remains_part_of_key():
    calls = []
    snapshots = StableTranslationSnapshots(lambda text, lang: calls.append(text) or f"译:{text}")
    snapshots.update("A. B. C. D", "en")
    snapshots.update("A. B. C. D", "en")
    snapshots.update("A. B. C. D", "fr")
    assert calls == ["A. B. C. ", "D", "A. B. C. ", "D"]


def test_incomplete_third_sentence_stays_active_until_terminated():
    calls = []
    snapshots = StableTranslationSnapshots(lambda text, lang: calls.append(text) or f"译:{text}")
    first, _ = snapshots.update("A. B. unfinished", "en")
    assert snapshots.stable_translation == ""
    assert snapshots.active_translation == first
    completed, _ = snapshots.update("A. B. unfinished.", "en")
    assert snapshots.stable_translation == completed
    assert snapshots.active_translation == ""
    assert calls == ["A. B. unfinished", "A. B. unfinished."]


@pytest.mark.parametrize("source", [None, "", " \n\t", "x" * 20001])
def test_empty_none_and_oversized_inputs_are_rejected(source):
    snapshots = StableTranslationSnapshots(lambda text, lang: "unused")
    with pytest.raises(ValueError):
        snapshots.update(source, "en")


def test_group_translation_receives_three_sentence_context_and_language_is_keyed():
    calls = []
    snapshots = StableTranslationSnapshots(lambda text, lang: calls.append((text, lang)) or "中文")
    snapshots.update("我想要。你可以。现在开始。", "en")
    snapshots.update("我想要。你可以。现在开始。", "fr")
    assert calls == [("我想要。你可以。现在开始。", "en"), ("我想要。你可以。现在开始。", "fr")]


def test_windows_preserve_all_unicode_characters_respect_limit_and_keep_decimal_abbreviation():
    source = "Value 3.14 is approx. e.g. useful. 中文句子。" + "猫🙂" * 20
    windows = source_windows(source, max_chars=17, group_sentences=3)
    assert "".join(windows) == source
    assert all(len(window) <= 17 for window in windows)
    assert any("3.14" in window for window in windows)
    assert any("e.g." in window for window in windows)


def test_long_unpunctuated_source_is_fully_split_at_the_bound():
    source = "猫🙂" * 27
    windows = source_windows(source, max_chars=8)
    assert "".join(windows) == source
    assert all(len(window) <= 8 for window in windows)


def test_space_at_limit_index_does_not_create_oversized_piece():
    windows = source_windows("12345678 X", max_chars=8)
    assert windows == ["12345678", " X"]
    assert all(len(window) <= 8 for window in windows)


def test_oversized_active_sentence_commits_fixed_chunks_and_keeps_last_chunk_active():
    calls = []
    snapshots = StableTranslationSnapshots(lambda text, lang: calls.append(text) or f"译:{text}", max_chars=8)
    first, _ = snapshots.update("猫🙂" * 10, "en")
    assert first == "\n".join([f"译:{'猫🙂' * 4}", f"译:{'猫🙂' * 4}", f"译:{'猫🙂' * 2}"])
    assert snapshots.stable_translation == "\n".join([f"译:{'猫🙂' * 4}", f"译:{'猫🙂' * 4}"])
    assert snapshots.active_translation == f"译:{'猫🙂' * 2}"
    snapshots.update("猫🙂" * 10 + "鸟", "en")
    assert calls[:3] == ["猫🙂" * 4, "猫🙂" * 4, "猫🙂" * 2]
    assert calls[3:] == ["猫🙂" * 2 + "鸟"]
