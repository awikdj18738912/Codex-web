# -*- coding: utf-8 -*-
"""Character n-gram deduplication and dataset-distribution statistics."""
from collections import Counter, defaultdict
from math import ceil, floor

from tqdm import tqdm


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract a set of character n-grams."""
    text = text.strip()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute set Jaccard similarity."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def is_duplicate(text: str, existing_ngrams: list[set[str]], threshold: float = 0.75) -> bool:
    """Return whether text matches any retained n-gram set."""
    new_ngrams = char_ngrams(text)
    for existing in existing_ngrams:
        if jaccard_similarity(new_ngrams, existing) >= threshold:
            return True
    return False


def deduplicate(items: list[dict], text_key: str = "input", threshold: float = 0.75) -> list[dict]:
    """Retain the first record from each near-duplicate group."""
    unique_items = []
    deduper = StreamingDeduplicator(threshold)
    for item in tqdm(items, desc="Dedup", unit="item"):
        if deduper.add(item, text_key=text_key):
            unique_items.append(item)
    return unique_items


class StreamingDeduplicator:
    """Incremental exact Jaccard deduplicator over retained records."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self._ngrams: list[set[str]] = []
        self._postings: dict[tuple[bool, str], set[int]] = defaultdict(set)

    def add(self, item: dict, text_key: str = "input") -> bool:
        """Return True iff item is retained; empty text keeps legacy behavior."""
        text = item.get(text_key, "")
        if not text:
            return True

        grams = char_ngrams(text)
        fragment = bool((item.get("meta") or {}).get("is_fragment"))
        a = len(grams)
        t = self.threshold

        # Legacy lookup considers only records sharing at least one gram. At
        # t <= 0 every such record qualifies; at t > 1 none can qualify.
        candidate_ids: set[int] = set()
        if t <= 0:
            for gram in grams:
                candidate_ids.update(self._postings.get((fragment, gram), ()))
        elif t <= 1:
            # Round the pruning bounds outward. This leaves a harmless extra
            # candidate at floating point boundaries while avoiding a false
            # negative from ceil/floor rounding in the bound calculation.
            min_common = floor(t * a)
            min_size = max(1, min_common)
            max_size = ceil(a / t)
            # If J(A,B) >= t, intersection >= t*|A|. Using floor(t*a)
            # conservatively, any chosen p=a-floor(t*a)+1 grams therefore
            # intersects B. Probe the rarest p grams to limit fanout.
            p = min(a, max(1, a - min_common + 1))
            rarest = sorted(
                grams,
                key=lambda gram: len(self._postings.get((fragment, gram), ())),
            )[:p]
            for gram in rarest:
                candidate_ids.update(self._postings.get((fragment, gram), ()))
            candidate_ids = {
                idx for idx in candidate_ids
                if min_size <= len(self._ngrams[idx]) <= max_size
            }

        if any(jaccard_similarity(grams, self._ngrams[idx]) >= t for idx in candidate_ids):
            return False

        idx = len(self._ngrams)
        self._ngrams.append(grams)
        for gram in grams:
            self._postings[(fragment, gram)].add(idx)
        return True


def compute_distribution(items: list[dict]) -> dict:
    """Compute scene, language, and fragment distributions."""
    stats = {
        "total": len(items),
        "scene": Counter(),
        "language": Counter(),
        "is_fragment": Counter(),
    }

    for item in items:
        meta = item.get("meta", {})
        stats["scene"][meta.get("scene", "unknown")] += 1
        stats["language"][meta.get("language", "unknown")] += 1
        stats["is_fragment"][str(meta.get("is_fragment", False))] += 1

    return {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
