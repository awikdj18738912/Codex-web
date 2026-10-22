"""Stable, bounded translation windows for incremental source snapshots."""
from __future__ import annotations

import re
import time


_CJK_END = "。！？"
_END = ".!?" + _CJK_END
_ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "mr.", "mrs.", "ms.", "dr.", "prof.", "vs.", "st."}


def _sentence_data(text: str) -> tuple[list[str], int]:
    """Return punctuation spans and the number terminated by sentence marks."""
    if not text:
        return [], 0
    spans = []
    completed = 0
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char not in _END:
            i += 1
            continue
        if char == ".":
            if i > 0 and i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit():
                i += 1
                continue
            prefix = text[max(start, i - 10):i + 1].lower()
            if any(prefix.endswith(abbr) for abbr in _ABBREVIATIONS):
                i += 1
                continue
            # A dot between letters is usually an abbreviation or domain token.
            if i > 0 and i + 1 < len(text) and text[i - 1].isalpha() and text[i + 1].isalpha():
                i += 1
                continue
        end = i + 1
        while end < len(text) and text[end] in "\"'”’»)]}）】」』":
            end += 1
        # Western punctuation is a boundary only before whitespace or end of input.
        if char in ".!?" and end < len(text) and not text[end].isspace():
            i += 1
            continue
        while end < len(text) and text[end].isspace():
            end += 1
        spans.append(text[start:end])
        completed += 1
        start = end
        i = end
    if start < len(text):
        spans.append(text[start:])
    return spans, completed


def _sentence_spans(text: str) -> list[str]:
    """Split at sentence punctuation while keeping every source character."""
    return _sentence_data(text)[0]


def _split_long(text: str, limit: int) -> list[str]:
    """Split a long sentence into bounded pieces, preferring whitespace."""
    result = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        else:
            # Keep the delimiter with the preceding piece.
            cut += 1
        result.append(text[:cut])
        text = text[cut:]
    if text:
        result.append(text)
    return result


def source_windows(text: str, max_chars: int = 700, group_sentences: int = 3) -> list[str]:
    """Return bounded windows, grouping consecutive sentences with context.

    Windows concatenate to the original text exactly. Each window consists of
    up to ``group_sentences`` sentence spans; oversized spans are subdivided.
    """
    if max_chars < 1 or group_sentences < 1:
        raise ValueError("max_chars and group_sentences must be positive")
    sentences = _sentence_spans(text)
    windows = []
    group = []
    for sentence in sentences:
        pieces = _split_long(sentence, max_chars)
        # Long sentences cannot be merged with neighboring sentences while
        # preserving both the size bound and meaningful sentence grouping.
        if len(pieces) > 1:
            if group:
                windows.append("".join(group))
                group = []
            windows.extend(pieces)
            continue
        if group and (len(group) >= group_sentences or sum(map(len, group)) + len(sentence) > max_chars):
            windows.append("".join(group))
            group = []
        group.append(sentence)
    if group:
        windows.append("".join(group))
    return windows


class StableTranslationSnapshots:
    """Cache fixed sentence groups while re-translating only the active tail."""

    def __init__(self, translate, max_chars: int = 700, group_sentences: int = 3):
        if max_chars < 1 or group_sentences < 1:
            raise ValueError("max_chars and group_sentences must be positive")
        self.translate = translate
        self.max_chars = max_chars
        self.group_sentences = group_sentences
        self.cache = {}
        self.last_stats = {
            "model_calls": 0, "source_characters": 0, "cache_hits": 0,
            "windows": 0, "stable_windows": 0,
            "total_model_calls": 0, "total_source_characters": 0,
        }
        self.stable_translation = ""
        self.active_translation = ""

    def update(self, source, language):
        if not isinstance(source, str) or not source.strip() or len(source) > 20000:
            raise ValueError("原文为空或超过20000字符，请分段处理。")
        started = time.perf_counter()
        sentences, completed_sentences = _sentence_data(source)
        # Only complete groups are stable. Changed partial groups need new
        # translation; identical text/language can reuse the previous result.
        stable_sentence_count = (completed_sentences // self.group_sentences) * self.group_sentences
        full_group_chars = sum(len(sentence) for sentence in sentences[:stable_sentence_count])
        stable_prefix = source[:full_group_chars]
        active_tail = source[full_group_chars:]
        stable_windows = source_windows(stable_prefix, self.max_chars, self.group_sentences)
        active_windows = source_windows(active_tail, self.max_chars, self.group_sentences)
        # A long unfinished sentence has deterministic max_chars boundaries.
        # Earlier chunks cannot change when text is appended, so they can be
        # committed while its final chunk remains active.
        active_sentences = _sentence_spans(active_tail)
        forced_stable_count = 0
        if len(active_sentences) == 1 and len(active_tail) > self.max_chars:
            forced_stable_count = max(0, len(active_windows) - 1)
            stable_windows.extend(active_windows[:forced_stable_count])
            active_windows = active_windows[forced_stable_count:]
        windows = stable_windows + active_windows

        output = []
        fresh = {}
        calls = hits = source_characters = 0
        stable_results = []
        active_results = []
        for index, window in enumerate(windows):
            if not window.strip():
                continue
            key = (language, window)
            stable = index < len(stable_windows)
            value = self.cache.get(key)
            if value is None:
                value = self.translate(window, language)
                calls += 1
                source_characters += len(window)
                if not isinstance(value, str) or not value.strip():
                    raise RuntimeError("翻译模型未返回完整中文，请缩短片段后重试。")
            else:
                hits += 1
            output.append(value.strip())
            (stable_results if stable else active_results).append(value.strip())
            fresh[key] = value
        self.cache = fresh
        self.stable_translation = "\n".join(stable_results)
        self.active_translation = "\n".join(active_results)
        self.last_stats = {
            "model_calls": calls,
            "source_characters": source_characters,
            "cache_hits": hits,
            "windows": len(windows),
            "stable_windows": len(stable_windows),
            "total_model_calls": self.last_stats["total_model_calls"] + calls,
            "total_source_characters": self.last_stats["total_source_characters"] + source_characters,
        }
        return "\n".join(output), (time.perf_counter() - started) * 1000
