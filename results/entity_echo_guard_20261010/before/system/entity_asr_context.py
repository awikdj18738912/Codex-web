"""Bounded, session-owned canonical glossary for optional ASR prompting."""

from dataclasses import dataclass
import hashlib
import json

MAX_ASR_CONTEXT_CHARS = 2048


def validate_asr_context(value):
    if not isinstance(value, str):
        raise ValueError("context must be a string")
    if len(value) > MAX_ASR_CONTEXT_CHARS:
        raise ValueError(f"context exceeds {MAX_ASR_CONTEXT_CHARS} characters")
    if "\x00" in value:
        raise ValueError("context must not contain NUL")
    return value.strip()


def context_digest(context):
    return hashlib.sha256(context.encode("utf-8")).hexdigest()[:16] if context else None


@dataclass(frozen=True)
class EntityASRContext:
    text: str
    terms: tuple[str, ...]


def build_entity_asr_context(
    entities, selected_domain="general", max_terms=16, max_chars=1024
):
    if not 1 <= max_terms <= 64 or not 128 <= max_chars <= MAX_ASR_CONTEXT_CHARS:
        raise ValueError("invalid entity ASR context limits")
    prefix = (
        "参考术语仅用于辅助拼写，不代表音频内容；未在音频中听到的词不要补充。术语列表："
    )
    terms = []
    for entity in sorted(
        entities, key=lambda e: (e.domain != selected_domain, -e.priority, e.entity_id)
    ):
        if not entity.enabled or entity.domain != selected_domain:
            continue
        word = " ".join(entity.canonical_text.split())
        # Very short Chinese terms are too ambiguous for an automatic prompt.
        if len(word) < 3 or len(word) > 48 or word in terms or "\x00" in word:
            continue
        proposal = prefix + json.dumps([*terms, word], ensure_ascii=False)
        if len(proposal) > max_chars:
            continue
        terms.append(word)
        if len(terms) == max_terms:
            break
    return EntityASRContext(
        prefix + json.dumps(terms, ensure_ascii=False) if terms else "", tuple(terms)
    )
