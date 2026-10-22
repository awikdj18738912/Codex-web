"""Independent lexical boundary checks for verified and proposed entities.

The tokenizer's dictionary is never populated with matching candidates: doing
that would make a proposed correction its own evidence for a word boundary.
"""

from functools import lru_cache
import logging
import threading
import unicodedata


def _latin_token(character: str) -> bool:
    folded = unicodedata.normalize("NFKC", character)
    return bool(folded) and all(
        c.isascii() and (c.isalnum() or c == "_") for c in folded
    )


def latin_boundaries_valid(text: str, start: int, end: int) -> bool:
    return not (
        start
        and _latin_token(text[start])
        and _latin_token(text[start - 1])
        or end < len(text)
        and _latin_token(text[end - 1])
        and _latin_token(text[end])
    )


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        import jieba
    except ImportError:
        return None
    tokenizer = jieba.Tokenizer()
    jieba.setLogLevel(logging.ERROR)
    tokenizer.initialize()
    return tokenizer


_LOCK = threading.Lock()


@lru_cache(maxsize=256)
def _tokens(text: str):
    tokenizer = _tokenizer()
    if tokenizer is None:
        return ()
    with _LOCK:
        return tuple(tokenizer.tokenize(text, HMM=False))


@lru_cache(maxsize=128)
def inferred_tokens(text: str):
    """Independent unknown-name boundaries; never insert glossary words."""
    tokenizer = _tokenizer()
    if tokenizer is None:
        return ()
    with _LOCK:
        return tuple(tokenizer.tokenize(text, HMM=True))


def candidate_boundary_issue(text: str, start: int, end: int) -> str | None:
    if not latin_boundaries_valid(text, start, end):
        return "latin_word_boundary_conflict"
    observed = text[start:end]
    chinese = any("\u3400" <= c <= "\u9fff" for c in observed)
    if not chinese:
        return None
    if any(c in ",，;；:：" for c in observed):
        return "clause_boundary_conflict"
    tokens = _tokens(text)
    if not tokens:
        return "chinese_boundaries_unavailable"
    if any(a < start < b or a < end < b for _word, a, b in tokens):
        return "chinese_word_boundary_conflict"
    # Two-character homophones can themselves be complete, valid words
    # (务工/误工). Boundary evidence cannot authorize their substitution.
    if sum("\u3400" <= c <= "\u9fff" for c in observed) <= 2 and not any(
        _latin_token(c) for c in observed
    ):
        return "short_chinese_span_requires_alias"
    overlapping = [(word, a, b) for word, a, b in tokens if start <= a and b <= end]
    if overlapping and overlapping[-1][0] in {
        "是",
        "的",
        "了",
        "在",
        "和",
        "与",
        "或",
        "也",
    }:
        return "grammatical_tail_conflict"
    return None
