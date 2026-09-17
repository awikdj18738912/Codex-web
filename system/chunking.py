"""Text chunking for partial streaming ASR hypotheses."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(
    r"(?:[!?\u3002\uff01\uff1f]|(?<!\d)\.(?!\d))\s*"
)
_ANY_PUNCTUATION = re.compile(
    r"(?:[,;:!?\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001]|"
    r"(?<!\d)\.(?!\d))\s*"
)
_SELF_CORRECTION = re.compile(r"(?:不对|不是|而是|应该是)")
_SELF_CORRECTION_PREFIX = re.compile(r"^(?:不对|不是|而是|应该是)")
# A corrected clause may end with either a soft clause punctuation or a
# sentence terminator.  Only searching soft punctuation lets a break jump
# past the real clause boundary to a far-away comma, producing an
# oversized chunk that also pushes the correction value out of the
# K-window refinement range.
_CORRECTION_BOUNDARY_PUNCTUATION = re.compile(r"[,，;；:：、。！？!?]\s*")
_SENTENCE_END_CHARACTERS = frozenset(".!?。！？")
_CLAUSE_BOUNDARY_CHARACTERS = _SENTENCE_END_CHARACTERS | frozenset(
    ",，、;；:："
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One immutable raw-ASR chunk."""

    index: int
    text: str


class ChunkManager:
    """Convert evolving ASR hypotheses into bounded, stable text chunks.

    Hypotheses are cumulative within a VAD segment. Sentence-final punctuation
    closes a chunk. Long clauses containing a self-correction marker
    (``不对``/``不是``/``而是``) may close at the next clause punctuation even
    before the hard character limit. Text longer than ``max_chars`` is cut at
    the nearest preceding punctuation, or exactly at the limit when no
    punctuation exists.
    A VAD boundary flushes the remaining text and starts a new hypothesis.
    """

    def __init__(
        self, max_chars: int = 80, *, one_punctuation_window: bool = False
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        self.max_chars = max_chars
        self.one_punctuation_window = one_punctuation_window
        self._hypothesis = ""
        self._committed_source = ""
        self._next_index = 0

    @property
    def pending_text(self) -> str:
        """Source text accepted since the most recently emitted chunk."""

        return self._pending

    @property
    def _pending(self) -> str:
        return self._hypothesis[len(self._committed_source) :]

    def update(self, hypothesis: str, *, vad_boundary: bool = False) -> list[Chunk]:
        """Accept a cumulative partial hypothesis and emit newly stable chunks."""

        normalized = hypothesis.strip()
        if self._committed_source and not normalized.startswith(self._committed_source):
            raise ValueError(
                "ASR revised text that has already been committed; start a new VAD "
                "segment or delay chunk emission"
            )
        self._hypothesis = normalized
        chunks = self._emit_available(flush=vad_boundary)
        if vad_boundary:
            self._hypothesis = ""
            self._committed_source = ""
        return chunks

    def flush(self) -> list[Chunk]:
        """Close the current segment even when the ASR produced no VAD event."""

        return self.update(self._hypothesis, vad_boundary=True)

    def _emit_available(self, *, flush: bool) -> list[Chunk]:
        emitted: list[Chunk] = []
        while pending := self._pending:
            split = self._split_point(pending, flush=flush)
            if split is None:
                break
            source = pending[:split]
            text = source.strip()
            self._committed_source += source
            if not text:
                continue
            emitted.append(Chunk(index=self._next_index, text=text))
            self._next_index += 1
        if self.one_punctuation_window:
            return emitted
        return merge_self_correction_chunks(emitted, self.max_chars)

    def _split_point(self, text: str, *, flush: bool) -> int | None:
        if self.one_punctuation_window:
            punctuation_end = _ANY_PUNCTUATION.search(text)
            if punctuation_end is not None and punctuation_end.end() <= self.max_chars:
                return punctuation_end.end()
            if len(text) > self.max_chars:
                return self.max_chars
            return len(text) if flush else None

        correction_end = self._self_correction_break(text)
        if correction_end is not None:
            return correction_end

        sentence_end = _SENTENCE_END.search(text)
        if sentence_end is not None and sentence_end.end() <= self.max_chars:
            return sentence_end.end()

        if len(text) > self.max_chars:
            candidates = list(_ANY_PUNCTUATION.finditer(text[: self.max_chars]))
            return candidates[-1].end() if candidates else self.max_chars

        return len(text) if flush else None

    def _self_correction_break(self, text: str) -> int | None:
        """Return the nearest clause boundary after a self-correction."""

        # Keep short examples such as ``苹果，不对，梨`` together so the
        # Refiner sees the full correction. For longer hypotheses, separate
        # the corrected clause before another edit is included in the same
        # generation request.
        if len(text) <= max(1, self.max_chars // 2):
            return None
        for marker in _SELF_CORRECTION.finditer(text):
            search_start = marker.end()
            while (
                search_start < len(text)
                and text[search_start] in " \t,，;；:：、"
            ):
                search_start += 1
            punctuation = _CORRECTION_BOUNDARY_PUNCTUATION.search(
                text,
                search_start,
                min(len(text), self.max_chars),
            )
            if punctuation is not None:
                return punctuation.end()
        return None


def merge_self_correction_chunks(
    chunks: list[Chunk], max_chars: int = 80
) -> list[Chunk]:
    """Keep a self-correction clause in one source-owned chunk.

    ASR punctuation may end the abandoned clause before emitting ``不对``.
    In the one-punctuation streaming mode this commonly produces three
    adjacent chunks: ``错误内容，`` / ``不对，`` / ``修正内容。``.  If those
    chunks are counted independently, the first one can be committed before
    the Refiner ever sees the correction.  Group the antecedent, marker, and
    corrected clause (up to the first sentence boundary) as one logical,
    source-owned chunk.  Ordinary punctuation chunks remain independent.
    """

    merged: list[Chunk] = []
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        marker_text = chunk.text.lstrip()
        if (
            merged
            and _SELF_CORRECTION_PREFIX.match(marker_text)
            and _ends_with_clause_boundary(merged[-1].text)
            and len(merged[-1].text) + len(chunk.text) <= max_chars
        ):
            previous = merged.pop()
            group_text = previous.text + chunk.text
            next_index = index + 1
            # The marker chunk usually ends in a comma. Include the following
            # source chunk(s) until the corrected clause closes, so the model
            # receives the full ``错误，不对，修正。`` relation.
            while (
                next_index < len(chunks)
                and not _ends_with_sentence(group_text)
                and len(group_text) + len(chunks[next_index].text) <= max_chars
            ):
                group_text += chunks[next_index].text
                next_index += 1
            merged.append(Chunk(previous.index, group_text))
            index = next_index
            continue
        merged.append(chunk)
        index += 1
    return merged


def _ends_with_clause_boundary(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] in _CLAUSE_BOUNDARY_CHARACTERS


def _ends_with_sentence(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] in _SENTENCE_END_CHARACTERS
