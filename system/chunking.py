"""Text chunking for partial streaming ASR hypotheses."""

from __future__ import annotations

import re
import os
from dataclasses import dataclass

_SENTENCE_END = re.compile(
    r"(?:[!?\u3002\uff01\uff1f]|(?<!\d)\.(?!\d))\s*"
)
_ANY_PUNCTUATION = re.compile(
    r"(?:[,;:!?\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001]|"
    r"(?<!\d)\.(?!\d))\s*"
)
# Streaming punctuation windows keep enumerated phrases together.  The
# broader set above is still used by the legacy length-bound splitter.
_STREAM_WINDOW_PUNCTUATION = re.compile(
    r"(?:[,;:!?\uff0c\u3002\uff01\uff1f\uff1b\uff1a]|"
    r"(?<!\d)\.(?!\d))\s*"
)
STRONG_CORRECTION_MARKERS = (
    "不对",
    "我说错了",
    "说错了",
    "应该是",
    "应该说",
    "准确地说",
    "我是说",
    "改成",
)
WEAK_CORRECTION_MARKERS = ("不是", "不，是")
_SELF_CORRECTION = re.compile(
    r"(?:不对|我说错了|说错了|应该是|应该说|准确地说|我是说|改成|不是|不，是|而是)"
)
_SELF_CORRECTION_PREFIX = re.compile(
    r"^(?:不对|我说错了|说错了|应该是|应该说|准确地说|我是说|改成|不是|不，是|而是)"
)
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
_HARD_BOUNDARY_CHARACTERS = frozenset(".!?。！？；!?；\n")
_BOUNDARY_DUPLICATE_RE = re.compile(
    r"(?P<left>[的地得了着过])(?P<punctuation>[。！？!?；;])"
    r"(?P<right>[的地得了着过])"
)
_BOUNDARY_DUPLICATE_EXCEPTIONS = (
    "的确",
    "的话",
    "的时候",
    "的一",
    "了不起",
)
FILLER_TOKENS = frozenset(
    {"嗯", "嗯嗯", "呃", "额", "啊", "呃呃", "那个", "这个"}
)
SELF_CORRECTION_MAX_BACKTRACK_CHUNKS = 3
SELF_CORRECTION_MAX_FILLER_CHUNKS = 2
SELF_CORRECTION_MAX_WINDOW_CHARS = 100


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


SELF_CORRECTION_ANTECEDENT_RECOVERY_ENABLED = _env_bool(
    "SELF_CORRECTION_ANTECEDENT_RECOVERY", True
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
            punctuation_end = _STREAM_WINDOW_PUNCTUATION.search(text)
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

    if not chunks:
        return []
    if not SELF_CORRECTION_ANTECEDENT_RECOVERY_ENABLED:
        return _merge_adjacent_self_correction_chunks(chunks, max_chars)

    # Build non-overlapping source intervals first.  This lets a correction
    # reopen an antecedent that was already emitted into ``merged`` without
    # mutating or guessing at refined-text offsets.
    intervals: list[tuple[int, int]] = []
    cursor = 0
    consumed_end = 0
    while cursor < len(chunks):
        antecedent = find_self_correction_antecedent(chunks, cursor)
        if antecedent is None:
            cursor += 1
            continue
        marker_index = cursor
        if antecedent < consumed_end:
            cursor += 1
            continue
        group_end = marker_index + 1
        group_text = "".join(
            chunk.text for chunk in chunks[antecedent:group_end]
        )
        # Include the complete corrected clause, including its closing
        # punctuation.  Self-correction is the only path allowed to use the
        # slightly wider recovery budget; ordinary windows remain unchanged.
        while (
            group_end < len(chunks)
            and not _ends_with_sentence(group_text)
            and len(group_text) + len(chunks[group_end].text)
            <= max(max_chars, SELF_CORRECTION_MAX_WINDOW_CHARS)
        ):
            group_text += chunks[group_end].text
            group_end += 1
        if group_end == marker_index + 1 and marker_index + 1 < len(chunks):
            # Marker-only chunks are still useful context, but do not consume
            # a following clause when doing so would exceed the recovery cap.
            group_end = marker_index + 1
        if antecedent >= consumed_end and len(group_text) <= max(
            max_chars, SELF_CORRECTION_MAX_WINDOW_CHARS
        ):
            intervals.append((antecedent, group_end))
            consumed_end = group_end
            cursor = group_end
        else:
            cursor += 1

    if not intervals:
        return list(chunks)

    merged: list[Chunk] = []
    source_cursor = 0
    for start, end in intervals:
        if start < source_cursor:
            continue
        merged.extend(chunks[source_cursor:start])
        merged.append(
            Chunk(
                chunks[start].index,
                "".join(chunk.text for chunk in chunks[start:end]),
            )
        )
        source_cursor = end
    merged.extend(chunks[source_cursor:])
    return merged


def merge_boundary_anomaly_chunks(
    chunks: list[Chunk], max_chars: int = 80
) -> list[Chunk]:
    """Join adjacent chunks when a suspicious boundary crosses the split.

    One-punctuation mode intentionally creates very small source blocks.  A
    malformed hard stop such as ``三番五次的。`` / ``的提醒`` therefore never
    appears in one gate or Refiner input and can be incorrectly kept.  Join
    only the two blocks containing a reviewed low-risk ``particle + hard mark +
    same particle`` pattern; lexical continuations such as ``的。的确`` remain
    separate.  The slightly wider cap is limited to this recovery path.
    """

    if len(chunks) < 2:
        return list(chunks)
    limit = max(max_chars, 100)
    merged: list[Chunk] = []
    index = 0
    while index < len(chunks):
        if index + 1 < len(chunks):
            left = chunks[index]
            right = chunks[index + 1]
            combined = left.text + right.text
            boundary = len(left.text)
            anomaly = next(
                (
                    match
                    for match in _BOUNDARY_DUPLICATE_RE.finditer(combined)
                    if match.start() < boundary < match.end()
                    and not any(
                        combined.startswith(exception, match.start("right"))
                        for exception in _BOUNDARY_DUPLICATE_EXCEPTIONS
                    )
                ),
                None,
            )
            if anomaly is not None and len(combined) <= limit:
                merged.append(Chunk(left.index, combined))
                index += 2
                continue
        merged.append(chunks[index])
        index += 1
    return merged


def _merge_adjacent_self_correction_chunks(
    chunks: list[Chunk], max_chars: int
) -> list[Chunk]:
    """Baseline merge used when antecedent recovery is disabled by flag."""

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
            while (
                next_index < len(chunks)
                and not _ends_with_sentence(group_text)
                and len(group_text) + len(chunks[next_index].text) <= max_chars
            ):
                group_text += chunks[next_index].text
                next_index += 1
            merged.append(Chunk(previous.index, group_text))
            index = next_index
        else:
            merged.append(chunk)
            index += 1
    return merged


def strip_punctuation_and_spaces(text: str) -> str:
    """Return the lexical content used for filler-only classification."""

    return re.sub(r"[\s,，、。！？!?；;：:]+", "", text)


def is_filler_only(text: str) -> bool:
    """Return whether a chunk consists solely of a spoken filler token."""

    return strip_punctuation_and_spaces(text) in FILLER_TOKENS


def _has_correction_marker(text: str) -> bool:
    value = text.lstrip()
    if any(value.startswith(marker) for marker in STRONG_CORRECTION_MARKERS):
        return True
    return value.startswith("不是") or value.startswith("不，是")


def find_self_correction_antecedent(
    chunks: list[Chunk], correction_idx: int,
    *,
    max_backtrack: int = SELF_CORRECTION_MAX_BACKTRACK_CHUNKS,
    max_fillers: int = SELF_CORRECTION_MAX_FILLER_CHUNKS,
) -> int | None:
    """Find the nearest soft-boundary semantic chunk before a correction.

    Filler-only chunks may be skipped, but hard sentence boundaries stop the
    search. Weak ``不是`` markers require at least one skipped filler; this
    prevents ordinary negations from reopening an unrelated source span.
    """

    if correction_idx <= 0 or correction_idx >= len(chunks):
        return None
    marker_text = chunks[correction_idx].text.lstrip()
    if not _has_correction_marker(marker_text):
        return None
    weak_marker = marker_text.startswith(WEAK_CORRECTION_MARKERS)
    filler_count = 0
    start = max(0, correction_idx - max_backtrack)
    for index in range(correction_idx - 1, start - 1, -1):
        text = chunks[index].text
        if is_filler_only(text):
            filler_count += 1
            if filler_count > max_fillers:
                return None
            continue
        if weak_marker and filler_count == 0:
            return None
        if _ends_with_hard_boundary(text) and filler_count:
            # A filler after a hard boundary is usually a new sentence, not
            # evidence that the earlier sentence is the correction target.
            return None
        if _ends_with_clause_boundary(text):
            return index
        return None
    return None


def _ends_with_clause_boundary(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] in _CLAUSE_BOUNDARY_CHARACTERS


def _ends_with_hard_boundary(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] in _HARD_BOUNDARY_CHARACTERS


def _ends_with_sentence(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] in _SENTENCE_END_CHARACTERS
