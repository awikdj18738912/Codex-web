"""Bounded Refiner input windows and conservative output-quality checks."""

from __future__ import annotations

import re
import os
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from decimal import Decimal, InvalidOperation

from .numeric_normalizer import (
    ContextualNumericNormalizer,
    _COUNT_UNITS,
    _FIXED_EXPRESSIONS,
    _MULTIPLIER_UNITS,
    _RANGE_RE,
)
from .quantifiers import (
    LEXICAL_REDUPLICATIONS,
    is_quantifier_reduplication_deletion,
)


_SENTENCE_ENDINGS = frozenset("。！？!?；;\n")
_SOFT_BREAKS = frozenset("，、：:）)】] ")
_PUNCTUATION_WINDOW_CHARS = frozenset("，,、。！？!?；;：:.")
# A comma or sentence mark may end a streaming window; an enumeration comma
# stays inside the window.  Keep it in the full set above for integrity checks.
_SPLIT_WINDOW_CHARS = _PUNCTUATION_WINDOW_CHARS - frozenset("、")
_NON_DOT_PUNCTUATION = "".join(
    sorted(_PUNCTUATION_WINDOW_CHARS - frozenset("."))
)
_PUNCTUATION_WINDOW_RE = re.compile(
    rf"(?:[{re.escape(_NON_DOT_PUNCTUATION)}]|"
    r"(?<!\d)\.(?!\d))+$"
)
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?")
DEFAULT_REFINEMENT_MAX_CHARS = 80
_SEVERE_LOSS_MIN_SOURCE_CHARS = 16
_SHORT_SEGMENT_MIN_SIMILARITY = 0.35
_SEVERE_LOSS_MIN_LENGTH_RATIO = 0.65
_SHORT_LOSS_MIN_SOURCE_CHARS = 4
_SHORT_LOSS_MIN_SIMILARITY = 0.7
_CHINESE_NUMERAL_RE = re.compile(
    r"[零〇一二三四五六七八九十百千万亿两]+(?:点[零〇一二三四五六七八九]+)?"
)
_ARABIC_NUMERAL_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
_CHINESE_NUMERIC_UNITS = frozenset("十百千万亿年月日号元块点")
_CHINESE_COUNT_UNITS = frozenset(_COUNT_UNITS)
_CHINESE_MULTIPLIER_UNITS = frozenset(_MULTIPLIER_UNITS)
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CHINESE_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}
_PLACEHOLDER_RE = re.compile(r"__ENTITY_\d{3}__")
_LOW_RISK_DELETION_CHARS = frozenset(
    "啊呀吧呢吗嗯呃哈嘿唉哎"
)
_FUNCTION_CHARS = frozenset(
    "的地得了着过是就才又也都还很太吗呢啊呀吧啦哦嗯呃和与及而给把被从向"
)
# Lexical reduplications are protected independently of classifier syntax.
# Productive classifier pairs are checked structurally below and therefore
# require the explicit ``一XX`` prefix.
_PROTECTED_REDUPLICATION_PAIRS = LEXICAL_REDUPLICATIONS - frozenset(
    {"根根", "条条", "座座", "道道", "代代"}
)
_NEGATION_CHARS = frozenset("不没无未别莫勿")
_CORRECTION_MARKERS = (
    "不对",
    "我说错了",
    "说错了",
    "应该是",
    "应该说",
    "准确地说",
    "我是说",
    "改成",
    "不是",
    "而是",
)
_CORRECTION_MARKER_RE = re.compile("|".join(_CORRECTION_MARKERS))
_CORRECTION_SURFACE_NORMALIZER = ContextualNumericNormalizer()
_BOUNDARY_PARTICLES = frozenset("的地得了着过")
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
_CORRECTION_FILLER_TOKENS = frozenset(
    {"嗯", "嗯嗯", "呃", "额", "啊", "那个", "这个"}
)
_SAFE_BOUNDARY_REASONS = frozenset(
    {"boundary", "boundary_repair", "boundary_punctuation", "punctuation", "断句", "标点"}
)
_REPEATED_CJK_RUN_RE = re.compile(r"(?P<unit>[\u3400-\u9fff])(?P=unit)+")
_PUNCTUATED_REPETITION_RE = re.compile(
    r"(?P<unit>[\u3400-\u9fff]+)"
    r"(?P<separator>[^\w\s\u3400-\u9fff])"
    r"(?P=unit)"
)
_REPETITION_REVIEW_SENTENCE_ENDINGS = _SENTENCE_ENDINGS | frozenset("…")
_REPETITION_REVIEW_CLOSING_QUOTES = frozenset("\"'”’」』】）)]〉》〕〗〙〛")


def _repetition_review_context(text: str, start: int, end: int) -> str:
    """Return the candidate sentence with one complete sentence on each side."""
    boundaries: list[int] = []
    index = 0
    while index < len(text):
        if text[index] not in _REPETITION_REVIEW_SENTENCE_ENDINGS:
            index += 1
            continue
        index += 1
        # Consume punctuation runs such as ``？！`` and ellipses as one
        # boundary, then retain any quote/bracket closing that follows it.
        while index < len(text) and (
            text[index] in _REPETITION_REVIEW_SENTENCE_ENDINGS
            or text[index] in _REPETITION_REVIEW_CLOSING_QUOTES
        ):
            index += 1
        boundaries.append(index)

    preceding_boundaries = [boundary for boundary in boundaries if boundary <= start]
    context_start = preceding_boundaries[-2] if len(preceding_boundaries) >= 2 else 0

    current_end = next((boundary for boundary in boundaries if boundary >= end), None)
    if current_end is None:
        return text[context_start:]

    context_end = next(
        (boundary for boundary in boundaries if boundary > current_end),
        len(text),
    )
    return text[context_start:context_end]


@dataclass(frozen=True, slots=True)
class RepetitionReviewCandidate:
    """One structural repeated-span decision presented to the Refiner."""

    index: int
    start: int
    end: int
    source: str
    target: str
    kind: str
    context: str

    def public_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "context": self.context,
        }


def find_repetition_review_candidates(
    text: str, *, max_candidates: int = 24
) -> tuple[RepetitionReviewCandidate, ...]:
    """Find structural AA or adjacent phrase spans for model review.

    This function deliberately has no lexical vocabulary.  It only proposes
    alternatives; the Refiner decides whether a span is lexical or spoken
    repetition, and an unresolved candidate remains unchanged.
    """

    if not text or max_candidates < 1:
        return ()

    spans: list[tuple[int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < other_end and end > other_start for other_start, other_end in occupied)

    for match in _REPEATED_CJK_RUN_RE.finditer(text):
        start, end = match.span()
        unit = match.group("unit")
        # This is the structural ``一XX`` rule selected by the user.  It is
        # checked by shape, never by a list of specific classifier words.
        if start > 0 and text[start - 1] == "一":
            continue
        if end - start < 2:
            continue
        spans.append((start, end, text[start:end], unit))
        occupied.append((start, end))

    # A punctuation mark can be emitted between two copies of the same
    # spoken unit (for example ``喂，喂`` or ``啊！啊``).  Keep this structural:
    # the Refiner decides whether the repetition is natural emphasis or a
    # disfluency.  No word list is used here.
    for match in _PUNCTUATED_REPETITION_RE.finditer(text):
        start, end = match.span()
        unit = match.group("unit")
        if not _is_unicode_punctuation(match.group("separator")):
            continue
        if start > 0 and text[start - 1] == "一":
            continue
        if not unit or (start, end) in occupied or overlaps(start, end):
            continue
        spans.append((start, end, text[start:end], unit))
        occupied.append((start, end))

    # Look for a repeated multi-character span after single-character runs
    # have claimed their positions.  The longest matching unit is considered
    # first, which makes ``我们我们`` a candidate without adding vocabulary.
    max_width = min(12, len(text) // 2)
    for width in range(max_width, 1, -1):
        for start in range(0, len(text) - (width * 2) + 1):
            end = start + (width * 2)
            if overlaps(start, end):
                continue
            unit = text[start : start + width]
            if unit != text[start + width : end]:
                continue
            if not re.fullmatch(r"[\u3400-\u9fff]+", unit):
                continue
            if len(set(unit)) == 1:
                continue
            if start > 0 and text[start - 1] == "一":
                continue
            spans.append((start, end, text[start:end], unit))
            occupied.append((start, end))

    spans.sort(key=lambda item: item[0])
    candidates: list[RepetitionReviewCandidate] = []
    for index, (start, end, source, target) in enumerate(spans[:max_candidates], 1):
        candidates.append(
            RepetitionReviewCandidate(
                index=index,
                start=start,
                end=end,
                source=source,
                target=target,
                kind=(
                    "character_run"
                    if len(set(source)) == 1
                    else "punctuated_run"
                    if any(_is_unicode_punctuation(character) for character in source)
                    else "phrase_run"
                ),
                context=_repetition_review_context(text, start, end),
            )
        )
    return tuple(candidates)


def apply_repetition_review_decisions(
    text: str,
    candidates: tuple[RepetitionReviewCandidate, ...],
    decisions: object,
) -> tuple[str, tuple[dict[str, object], ...]]:
    """Apply only model-selected candidate deletions at source offsets."""

    if not isinstance(decisions, list):
        return text, tuple(
            {**candidate.public_dict(), "decision": "unresolved", "reason": "invalid_response"}
            for candidate in candidates
        )
    by_index = {candidate.index: candidate for candidate in candidates}
    selected: list[RepetitionReviewCandidate] = []
    records: list[dict[str, object]] = []
    for item in decisions:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        candidate = by_index.get(index)
        action = str(item.get("action", "keep")).strip().lower()
        reason = str(item.get("reason", "")).strip()
        if candidate is None or action not in {"keep", "remove"}:
            continue
        record = {**candidate.public_dict(), "decision": action, "reason": reason}
        records.append(record)
        if action == "remove":
            selected.append(candidate)

    selected.sort(key=lambda candidate: candidate.start)
    non_overlapping: list[RepetitionReviewCandidate] = []
    for candidate in selected:
        if non_overlapping and candidate.start < non_overlapping[-1].end:
            continue
        if text[candidate.start : candidate.end] != candidate.source:
            continue
        non_overlapping.append(candidate)

    # Validate each model-selected deletion against the current text instead
    # of validating the whole batch at once.  One ambiguous candidate must not
    # roll back unrelated, independently safe repetitions.
    repaired = text
    applied_spans: set[tuple[int, int]] = set()
    rejected_spans: set[tuple[int, int]] = set()
    for candidate in reversed(non_overlapping):
        start, end = candidate.start, candidate.end
        candidate_repaired = (
            repaired[:start] + candidate.target + repaired[end:]
        )
        reasons = reject_reasons(repaired, candidate_repaired)
        repeated_source = (
            bool(candidate.target)
            and len(candidate.source) % len(candidate.target) == 0
            and candidate.source
            == candidate.target * (len(candidate.source) // len(candidate.target))
        )
        if reasons and not (
            repeated_source and set(reasons) <= {"semantic_content_loss"}
        ):
            rejected_spans.add((start, end))
            continue
        repaired = candidate_repaired
        applied_spans.add((start, end))

    output_records: list[dict[str, object]] = []
    for record in records:
        span = (int(record["start"]), int(record["end"]))
        if span in rejected_spans:
            output_records.append(
                {
                    **record,
                    "applied": False,
                    "decision": "unresolved",
                    "reason": "guard_rejected",
                }
            )
        else:
            output_records.append(
                {**record, "applied": span in applied_spans}
            )
    return repaired, tuple(output_records)


def repetition_decisions_from_text_response(
    text: str,
    candidates: tuple[RepetitionReviewCandidate, ...],
    response: str,
) -> list[dict[str, object]] | None:
    """Read a legacy plain-text review only when every edit is a candidate delete."""

    candidate_ranges: dict[tuple[int, int], int] = {}
    for candidate in candidates:
        # SequenceMatcher is free to represent ``AA -> A`` as deleting the
        # first or the second copy.  Both source spans express the same
        # candidate decision; the actual edit is still applied by the
        # offset-safe candidate path above.
        unit_width = len(candidate.target)
        candidate_ranges[(candidate.start, candidate.start + unit_width)] = (
            candidate.index
        )
        candidate_ranges[(candidate.start + unit_width, candidate.end)] = (
            candidate.index
        )
    matcher = SequenceMatcher(None, text, response.strip(), autojunk=False)
    removed: set[int] = set()
    for tag, source_start, source_end, _, _ in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag != "delete":
            return None
        candidate_index = candidate_ranges.get((source_start, source_end))
        if candidate_index is None:
            return None
        removed.add(candidate_index)
    if not removed:
        return None
    return [
        {
            "index": candidate.index,
            "action": "remove" if candidate.index in removed else "keep",
            "reason": "legacy_text_response",
        }
        for candidate in candidates
    ]


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


SAFE_BOUNDARY_REPAIR_ENABLED = _env_bool("SAFE_BOUNDARY_REPAIR", True)


@dataclass(frozen=True, slots=True)
class BoundaryAnomaly:
    """A suspicious hard punctuation boundary between repeated particles."""

    position: int
    left_particle: str
    right_particle: str
    punctuation: str

    def public_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "left_particle": self.left_particle,
            "right_particle": self.right_particle,
            "punctuation": self.punctuation,
            "pattern": (
                f"{self.left_particle}{self.punctuation}{self.right_particle}"
            ),
        }


def split_for_refinement(
    text: str, *, max_chars: int = DEFAULT_REFINEMENT_MAX_CHARS,
    one_punctuation_window: bool = False,
) -> tuple[str, ...]:
    """Split text into bounded Refiner windows.

    With ``one_punctuation_window=True`` commas and sentence/clause marks
    close the current window, while enumeration commas stay with the following
    phrase. This is used by the live Web path to keep model context short.
    The legacy sentence-preferred policy remains available to other callers.
    """

    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")
    source = text.strip()
    if not source:
        return ()
    if one_punctuation_window:
        return _split_one_punctuation_window(source, max_chars)

    parts: list[str] = []
    start = 0
    while start < len(source):
        end = min(len(source), start + max_chars)
        if end == len(source):
            parts.append(source[start:end])
            break
        minimum_break = start + max_chars // 2
        preferred = _last_break(source, start, end, minimum_break, _SENTENCE_ENDINGS)
        soft = _last_break(source, start, end, minimum_break, _SOFT_BREAKS)
        cut = preferred or soft or end
        parts.append(source[start:cut])
        start = cut
    return tuple(part for part in parts if part)


def _split_one_punctuation_window(source: str, max_chars: int) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    index = 0
    while index < len(source):
        if source[index] in _SPLIT_WINDOW_CHARS and _is_window_punctuation(source, index):
            end = index + 1
            # Keep runs such as ``？！`` together instead of creating a
            # punctuation-only model request.
            while end < len(source) and source[end] in _PUNCTUATION_WINDOW_CHARS:
                end += 1
            if end - start <= max_chars:
                parts.append(source[start:end])
                start = end
                index = end
                continue
        if index - start + 1 >= max_chars:
            parts.append(source[start:index + 1])
            start = index + 1
        index += 1
    if start < len(source):
        parts.append(source[start:])
    return tuple(part for part in parts if part)


def join_refined_segments(parts: Iterable[str]) -> str:
    """Join segment outputs without inserting unwanted spaces in Chinese text."""

    output = ""
    for part in parts:
        value = part.strip()
        if not value:
            continue
        # Keep ASCII words separated across independent chunks, but never
        # insert a space inside a decimal value split at a window boundary
        # (``4.`` + ``5万`` must remain ``4.5万``).
        if (
            output
            and output[-1].isascii()
            and value[0].isascii()
            and not (output[-1].isdigit() and value[0].isdigit())
        ):
            output += " "
        output += value
    return output


def _is_window_punctuation(text: str, index: int) -> bool:
    """Return whether ``text[index]`` is a real boundary punctuation mark.

    An ASCII full stop between two digits is a decimal point, not a sentence
    boundary.  Treating it as a boundary was the source of outputs such as
    ``4. 5万`` and ``45. 22米``.
    """

    character = text[index]
    if character not in _PUNCTUATION_WINDOW_CHARS:
        return False
    if character == ".":
        previous = text[index - 1 : index]
        following = text[index + 1 : index + 2]
        return not (previous.isdigit() and following.isdigit())
    return True


def source_punctuation_lost(source_text: str, refined_text: str) -> bool:
    """Return whether source punctuation is missing from a refined window.

    Extra punctuation can be a legitimate model edit, but removing a source
    mark is unsafe when one model request contains several punctuation-owned
    chunks. Compare the source marks as an ordered subsequence so inserted
    formatting does not cause a false rejection while omissions do.
    """

    return _punctuation_subsequence_lost(source_text, refined_text)


def detect_boundary_anomalies(text: str) -> tuple[BoundaryAnomaly, ...]:
    """Detect punctuation splits that duplicate a low-risk function particle.

    This is deliberately a detector only.  It never edits text and excludes
    common lexical continuations such as ``的确`` and ``了不起``.
    """

    anomalies: list[BoundaryAnomaly] = []
    for match in _BOUNDARY_DUPLICATE_RE.finditer(text):
        right_start = match.start("right")
        if any(text.startswith(exception, right_start) for exception in _BOUNDARY_DUPLICATE_EXCEPTIONS):
            continue
        anomalies.append(
            BoundaryAnomaly(
                position=match.start(),
                left_particle=match.group("left"),
                right_particle=match.group("right"),
                punctuation=match.group("punctuation"),
            )
        )
    return tuple(anomalies)


def validate_boundary_repair(
    source: str,
    target: str,
    reason: str,
    *,
    protected_spans: Iterable[str] = (),
) -> bool:
    """Validate one structured, local repair of an anomalous boundary.

    Only deletion-only patches over a detected ``particle + punctuation +
    particle`` anomaly are accepted.  Numeric values, protected placeholders,
    and negation characters must remain unchanged; all other semantic guards
    still run after this validator returns.
    """

    if not SAFE_BOUNDARY_REPAIR_ENABLED:
        return False
    normalized_reason = str(reason).strip().lower()
    if normalized_reason not in _SAFE_BOUNDARY_REASONS:
        return False
    source = str(source)
    target = str(target)
    if not source or not target or len(source) > 30:
        return False
    if not detect_boundary_anomalies(source):
        return False
    if _numeric_values(source) != _numeric_values(target):
        return False
    if tuple(_PLACEHOLDER_RE.findall(source)) != tuple(_PLACEHOLDER_RE.findall(target)):
        return False
    if tuple(char for char in source if char in _NEGATION_CHARS) != tuple(
        char for char in target if char in _NEGATION_CHARS
    ):
        return False
    protected = tuple(item for item in protected_spans if item)
    if any(source.count(item) != target.count(item) for item in protected):
        return False

    punctuation_chars = _PUNCTUATION_WINDOW_CHARS
    deleted_punctuation = 0
    deleted_content: list[str] = []
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag != "delete":
            return False
        deleted = source[source_start:source_end]
        deleted_punctuation += sum(char in punctuation_chars for char in deleted)
        deleted_content.extend(
            char for char in deleted
            if char not in punctuation_chars and not char.isspace()
        )
    if not deleted_punctuation or deleted_punctuation > 2:
        return False
    if len(deleted_content) > 1:
        return False
    if deleted_content:
        anomalies = detect_boundary_anomalies(source)
        if not any(char == anomaly.right_particle for char in deleted_content for anomaly in anomalies):
            return False
    return True


def _punctuation_subsequence_lost(
    source_text: str,
    refined_text: str,
    *,
    excluded_span: tuple[int, int] | None = None,
) -> bool:
    """Compare source punctuation with the candidate as an ordered subsequence."""

    source_marks = tuple(
        char
        for index, char in enumerate(source_text)
        if _is_window_punctuation(source_text, index)
        and (
            excluded_span is None
            or not (excluded_span[0] <= index < excluded_span[1])
        )
    )
    if not source_marks:
        return False
    target_marks = tuple(
        char
        for index, char in enumerate(refined_text)
        if _is_window_punctuation(refined_text, index)
    )
    target_index = 0
    for source_mark in source_marks:
        while (
            target_index < len(target_marks)
            and target_marks[target_index] != source_mark
        ):
            target_index += 1
        if target_index >= len(target_marks):
            return True
        target_index += 1
    return False


def permits_self_correction_punctuation_repair(
    source_text: str, refined_text: str
) -> bool:
    """Return whether punctuation loss is confined to a superseded prefix.

    In a punctuation-sized window, ASR can emit ``错误句。`` and then a
    correction marker in the next chunk: ``错误句。不对，正确句。``.  A
    Refiner that resolves the correction is expected to return only
    ``正确句。``.  The punctuation attached to the abandoned clause is then
    intentionally absent, rather than an accidental formatting loss.

    Only the boundary from the previous clause mark through the correction
    marker is exempted.  Punctuation in the retained replacement clause and
    anything after it must still remain an ordered subsequence, so this does
    not turn a general punctuation deletion into an accepted edit.
    """

    source = _correction_surface(source_text)
    refined = _correction_surface(refined_text)
    span = _matched_correction_tail_span(source, refined)
    if span is None:
        return False
    allowed_start, allowed_end, _ = span
    return not _punctuation_subsequence_lost(
        source, refined, excluded_span=(allowed_start, allowed_end)
    )


def preserve_terminal_punctuation(source_text: str, refined_text: str) -> str:
    """Keep punctuation that closed a source-owned refinement window.

    The Refiner is allowed to clean spoken content, but it must not make a
    comma or sentence mark disappear merely because a short window omitted it
    in its response. In punctuation-window mode each source segment ends in
    at most one punctuation run, so restoring that exact suffix is
    deterministic and does not invent punctuation for an unfinished segment.
    """

    source = source_text.rstrip()
    refined = refined_text.rstrip()
    if not source or not refined:
        return refined_text
    source_match = _PUNCTUATION_WINDOW_RE.search(source)
    if source_match is None:
        return refined_text
    source_suffix = source_match.group(0)
    refined_match = _PUNCTUATION_WINDOW_RE.search(refined)
    if refined_match is not None:
        refined = refined[: refined_match.start()]
    return refined + source_suffix


def reject_reasons(raw_text: str, refined_text: str) -> tuple[str, ...]:
    """Return deterministic reasons to fall back to a raw source segment."""

    raw = raw_text.strip()
    refined = refined_text.strip()
    if not refined:
        return ("empty_refiner_output",)

    reasons: list[str] = []
    raw_sentences = _sentence_counts(raw)
    for sentence, count in _sentence_counts(refined).items():
        if len(sentence) >= 8 and count >= 3 and count > raw_sentences[sentence]:
            reasons.append("repeated_sentence")
            break

    # Catch a loop without sentence-ending punctuation, such as a generated
    # phrase repeated several times before max_new_tokens is exhausted.
    repeated_phrase = re.search(r"(.{8,80}?)(?:\1){2,}", refined)
    if repeated_phrase and repeated_phrase.group(1) not in raw:
        reasons.append("repeated_phrase")

    # A fluent, punctuated hallucination can still replace most of a source
    # window with one short sentence. Do not let final punctuation bypass the
    # completeness guard. This conservative floor still permits substantial
    # filler and repetition cleanup.
    if len(raw) >= _SHORT_LOSS_MIN_SOURCE_CHARS:
        length_ratio = len(refined) / len(raw)
        similarity = SequenceMatcher(None, raw, refined, autojunk=False).ratio()
        numeric_surface_only = _numeric_surface_only(raw, refined)
        if len(raw) >= _SEVERE_LOSS_MIN_SOURCE_CHARS and (
            length_ratio < _SEVERE_LOSS_MIN_LENGTH_RATIO
            and (len(raw) >= 24 or similarity < _SHORT_SEGMENT_MIN_SIMILARITY)
            and not numeric_surface_only
            # A self-correction intentionally removes the false start.  If
            # the candidate retains the corrected tail, do not classify that
            # deliberate compression as whole-window content loss.
            and not _is_intentional_correction_compression(
                raw, refined, length_ratio
            )
        ):
            reasons.append("severe_content_loss")
        # Sentence-sized windows can be short enough to bypass the long-source
        # floor above. Reject an unrelated rewrite as well, while still allowing
        # a legitimate self-correction whose final clause is retained verbatim
        # in the source (for example, ``...不对，有一个梨``).
        if (
            similarity < _SHORT_LOSS_MIN_SIMILARITY
            and refined not in raw
            and not _retains_correction_tail(raw, refined)
            and not numeric_surface_only
        ):
            reasons.append("severe_content_loss")

    # LLM cleanup can preserve the surrounding sentence while silently
    # changing a value (for example, ``两千一百三十五`` -> ``两百三十五``).
    # Length and edit-distance checks cannot detect that semantic loss, so
    # compare numeric mentions before accepting the candidate.  This also
    # permits safe Chinese-to-Arabic forms such as ``百分之五`` -> ``5%`` and
    # ``二零一五年五月五日`` -> ``2015年5月5日``.
    if (
        _numeric_values(raw) != _numeric_values(refined)
        and not _retains_correction_tail(raw, refined)
    ):
        reasons.append("numeric_value_mismatch")

    if _partially_converted_numeric_range(raw, refined):
        reasons.append("partial_numeric_range_conversion")

    # Length and global similarity are useful for catching a collapsed window,
    # but they miss small deletions that change who did what (``他给``), or a
    # one-character substitution that turns a meaningful word into a particle
    # (``傻`` -> ``的``).  Inspect the aligned edits locally as a second line of
    # defense.  Numeric-only surface changes are handled by the numeric guard
    # above and are deliberately excluded here.
    reasons.extend(_semantic_edit_reasons(raw, refined))

    source_complete = bool(raw) and raw[-1] in _SENTENCE_ENDINGS
    target_complete = bool(refined) and refined[-1] in _SENTENCE_ENDINGS
    if (
        source_complete
        and not target_complete
        and len(raw) >= 100
        and len(refined) < len(raw) * 0.8
    ):
        reasons.append("truncated_refiner_output")
    return tuple(dict.fromkeys(reasons))


def _partially_converted_numeric_range(raw: str, refined: str) -> bool:
    """Reject a model edit that changes only one side of a supported range.

    Reuse the source normalizer's accepted range spans so fixed expressions
    and unsupported numeric contexts keep their existing behavior.
    """

    for change in _CORRECTION_SURFACE_NORMALIZER.normalize(raw).changes:
        if change.kind not in {"measurement_range", "count_range"}:
            continue
        source_match = _RANGE_RE.fullmatch(change.original)
        if source_match is None:
            continue
        separator = source_match.group("separator")
        unit = source_match.group("unit")
        converted_left, converted_tail = change.replacement.split(separator, 1)
        converted_right = converted_tail[: -len(unit)]
        left = source_match.group("left")
        right = source_match.group("right")
        if (
            f"{left}{separator}{converted_right}{unit}" in refined
            or f"{converted_left}{separator}{right}{unit}" in refined
        ):
            return True
    return False


def _last_break(
    text: str, start: int, end: int, minimum: int, characters: frozenset[str]
) -> int | None:
    for index in range(end - 1, minimum - 1, -1):
        if text[index] in characters:
            return index + 1
    return None


def _sentence_counts(text: str) -> Counter[str]:
    return Counter(
        value.strip()
        for value in _SENTENCE_RE.findall(text)
        if value.strip()
    )


def _semantic_edit_reasons(raw: str, refined: str) -> tuple[str, ...]:
    """Reject local edits that can remove or invent sentence meaning.

    The Refiner is allowed to remove disfluencies, repeated runs, and the
    false start before a self-correction.  Other deleted CJK spans are treated
    conservatively: a short deletion can be semantically riskier than a large
    rewrite because the surrounding text still looks very similar.
    """

    if _numeric_surface_only(raw, refined):
        return ()

    reasons: list[str] = []
    matcher = SequenceMatcher(None, raw, refined, autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        source = raw[source_start:source_end]
        target = refined[target_start:target_end]
        if tag == "delete":
            if _allowed_deletion(raw, refined, source_start, source_end, source):
                continue
            if _meaningful_edit_text(source):
                reasons.append("semantic_content_loss")
        elif tag == "replace":
            # Compare the text surrounding an equivalent numeric core as
            # well as the value itself.  This accepts open-ended approximate
            # forms such as ``五十多座 -> 50多座`` without maintaining a
            # suffix allowlist, while rejecting a candidate that silently
            # drops the qualifier (``五十多 -> 50``).
            if _numeric_context_is_lost(source, target):
                reasons.append("numeric_value_mismatch")
                continue
            # A fluent-looking synonym or discourse marker can reverse the
            # speaker's meaning.  An unequal-length, character-disjoint
            # Chinese rewrite has no source-owned evidence.  Equal-length
            # typo corrections, equivalent numbers, and explicit
            # self-corrections keep their existing acceptance paths.
            source_cjk = set(re.findall(r"[\u3400-\u9fff]", source))
            target_cjk = set(re.findall(r"[\u3400-\u9fff]", target))
            if (
                len(source_cjk) >= 2
                and len(target_cjk) >= 2
                and len(source) != len(target)
                and source_cjk.isdisjoint(target_cjk)
                and not _numeric_edit_is_equivalent(source, target)
                and not (
                    _retains_correction_tail(raw, refined)
                    and _deletion_touches_correction(raw, source_start, source_end)
                )
            ):
                reasons.append("semantic_substitution")
                continue
            if _allowed_replacement(
                raw,
                refined,
                source_start,
                source_end,
                target_start,
                target_end,
                source,
                target,
            ):
                continue
            if (
                len(source) == 1
                and len(_strip_edit_punctuation(target)) >= 2
                and _meaningful_edit_text(source)
            ):
                reasons.append("unsupported_insertion")
                continue
            if (
                len(_strip_edit_punctuation(source)) >= 2
                and len(_strip_edit_punctuation(target)) == 1
                and _meaningful_edit_text(source)
            ):
                reasons.append("semantic_content_loss")
                continue
            if _meaningful_edit_text(source) and len(source) - len(target) >= 2:
                reasons.append("semantic_content_loss")
            elif _is_protected_semantic_substitution(
                raw, source_start, source, target
            ) or _suspicious_function_substitution(
                refined, target_start, target_end, source, target
            ):
                reasons.append("semantic_substitution")
        elif tag == "insert" and _suspicious_insertion(refined, target_start, target_end, target):
            reasons.append("unsupported_insertion")
    return tuple(dict.fromkeys(reasons))


def _allowed_deletion(
    raw: str, refined: str, start: int, end: int, source: str
) -> bool:
    if not _meaningful_edit_text(source):
        return True
    if _retains_correction_tail(raw, refined) and _deletion_touches_correction(raw, start, end):
        return True
    compact = _strip_edit_punctuation(source)
    if not compact:
        return True
    if all(char in _LOW_RISK_DELETION_CHARS for char in compact):
        return True
    if _is_protected_reduplication_deletion(raw, start, end, compact):
        return False
    if _is_repeated_deletion(raw, start, end, compact):
        return True
    if len(compact) == 1 and _is_boundary_echo_deletion(raw, start, end, compact):
        return True
    # A single deleted character can change the assertion just as much as a
    # long omission (for example 只不过 -> 不过).  Numeric formatting is
    # checked separately; unsupported character deletion is never inferred
    # safe merely from its length.
    return False


def _allowed_replacement(
    raw: str,
    refined: str,
    source_start: int,
    source_end: int,
    target_start: int,
    target_end: int,
    source: str,
    target: str,
) -> bool:
    if _numeric_surface_only(raw, refined):
        return True
    # A wider rewrite (for example a self-correction that also drops a
    # number-bearing false start) disables ``_numeric_surface_only`` for the
    # whole pair.  Legal ITN such as ``百分之三十七`` -> ``37%`` must still be
    # accepted, but only when the numeric value itself is unchanged.
    if _numeric_edit_is_equivalent(source, target):
        return True
    # Resolving a numeric self-correction replaces the superseded value with
    # the corrected one (``...获得12345431元。不对，应该是100块。`` ->
    # ``...获得100元。``).  Allow that only when the new value is the one the
    # kept replacement clause states, so an invented value is still rejected.
    tail_values = _correction_tail_values(raw, refined)
    if tail_values:
        # The sequence-alignment span usually stops right after the digits,
        # leaving the unit (``元``/``块``/``%``) in an ``equal`` block, so
        # test the span both with and without one trailing character.
        for width in (0, 1):
            candidate_values = _numeric_values(refined[target_start : target_end + width])
            if candidate_values and all(
                value in tail_values for value in candidate_values
            ):
                return True
    if _retains_correction_tail(raw, refined) and _deletion_touches_correction(
        raw, source_start, source_end
    ):
        return True
    # Pure punctuation and whitespace edits are editorial surface changes.
    if not _meaningful_edit_text(source) and not _meaningful_edit_text(target):
        return True
    if _is_protected_semantic_substitution(raw, source_start, source, target):
        return False
    # Rewriting two meaningful characters into two other meaningful characters
    # is the normal typo-correction case.  Keep it available to the Refiner;
    # the stricter checks target omission and particle substitutions.
    if len(source) == len(target) and len(source) >= 2:
        return True
    return False


def _suspicious_function_substitution(
    refined: str, target_start: int, target_end: int, source: str, target: str
) -> bool:
    if len(source) != 1 or len(target) != 1:
        return False
    if source in _FUNCTION_CHARS or target not in _FUNCTION_CHARS:
        return False
    # A content word becoming a function particle immediately before sentence
    # punctuation is especially likely to be an accidental semantic rewrite:
    # ``你当我傻？`` -> ``你当我的？``.
    return target_end >= len(refined) or refined[target_end : target_end + 1] in _SENTENCE_ENDINGS


def _suspicious_insertion(text: str, start: int, end: int, target: str) -> bool:
    compact = _strip_edit_punctuation(target)
    if len(compact) < 2:
        return False
    if all(char in _LOW_RISK_DELETION_CHARS for char in compact):
        return False
    # A multi-character content insertion has no source evidence.  Treat it
    # as unsafe even when it is not a repeated phrase (for example the model
    # inventing “的东西” in “互相连通、互相补给”).  Missing words require an
    # audio-backed re-decode or an explicit reviewed patch.
    if len(compact) >= 2:
        return True
    # Do not allow a repeated content run to be copied into a second location
    # (``...打打杀杀...不会跟你打`` -> ``...不会跟你打杀杀``).
    if len(set(compact)) == 1 and text[:start].count(compact) + text[end:].count(compact):
        return True
    return False


def _is_protected_reduplication_deletion(
    raw: str, start: int, end: int, compact: str
) -> bool:
    """Protect lexical forms and explicit ``一XX`` classifier forms."""

    if not compact or len(set(compact)) != 1:
        return False
    character = compact
    pair = character * 2
    if pair not in _PROTECTED_REDUPLICATION_PAIRS:
        # A classifier is protected only when the repeated pair is directly
        # preceded by the required ``一``.  This also handles an unseen
        # classifier without adding a phrase-specific exception.
        if len(compact) == 1:
            return is_quantifier_reduplication_deletion(raw, start, end, character)
        pair_start = max(0, start - 1)
        pair_end = min(len(raw) - 1, end + 1)
        return any(
            raw[index : index + 2] == pair
            and index > 0
            and raw[index - 1] == "一"
            for index in range(pair_start, pair_end)
        )
    return True


def _is_protected_semantic_substitution(
    raw: str, source_start: int, source: str, target: str
) -> bool:
    """Reject small edits that alter negation or a fixed contrast pattern."""

    if source == "只" and target == "是" and raw[source_start - 1 : source_start] == "不":
        return True
    if len(source) == len(target) == 1:
        if source in _NEGATION_CHARS and target not in _NEGATION_CHARS:
            return True
        if target in _NEGATION_CHARS and source not in _NEGATION_CHARS:
            return True
    return False


def _deletion_touches_correction(raw: str, start: int, end: int) -> bool:
    for match in _CORRECTION_MARKER_RE.finditer(raw):
        marker_start = match.start()
        marker_end = match.end()
        if start <= marker_end and end >= marker_start:
            return True
    return False


def _is_repeated_deletion(raw: str, start: int, end: int, compact: str) -> bool:
    if _is_adjacent_repetition_deletion(raw, start, end, compact):
        return True
    if _is_punctuated_repetition_deletion(raw, start, end, compact):
        return True
    if len(compact) >= 2 and raw[:start].endswith(compact):
        return True
    if len(compact) >= 2 and raw[end:].startswith(compact):
        return True
    # Punctuation can sit between two spoken repetitions (可恶！可恶！).
    pieces = [piece for piece in re.split(r"[，,、。！？!?；;\s]+", compact) if piece]
    return len(pieces) >= 2 and len(set(pieces)) == 1


def _is_punctuated_repetition_deletion(
    raw: str, start: int, end: int, compact: str
) -> bool:
    """Recognize deleting one copy from ``unit，unit``-shaped text."""

    if not compact:
        return False
    repeated_re = re.compile(
        re.escape(compact)
        + r"([^\w\s\u3400-\u9fff])"
        + re.escape(compact)
    )
    for match in repeated_re.finditer(raw):
        separator = match.group(1)
        if not _is_unicode_punctuation(separator):
            continue
        possible_ranges = {
            # SequenceMatcher may align the retained copy with either side
            # of the repetition, so permit removing the first copy+separator
            # or separator+second copy.
            (match.start(), match.start() + len(compact) + 1),
            (match.start() + len(compact), match.end()),
        }
        # It may also delete the second copy together with the separator that
        # follows it when the retained copy is aligned to the first.
        if match.end() < len(raw) and raw[match.end()] == separator:
            possible_ranges.add(
                (match.start() + len(compact) + 1, match.end() + 1)
            )
        if (start, end) in possible_ranges:
            return True
    return False


def _is_unicode_punctuation(value: str) -> bool:
    """Return whether a single character belongs to Unicode punctuation."""

    return len(value) == 1 and unicodedata.category(value).startswith("P")


def _is_boundary_echo_deletion(raw: str, start: int, end: int, echo: str) -> bool:
    """Allow removing one exact echoed glyph together with its boundary."""

    source = raw[start:end]
    boundary = "。！？!?；;"
    return (
        len(source) == 2
        and (
            (source[0] == echo and source[1] in boundary and raw[end:end + 1] == echo)
            or (source[1] == echo and source[0] in boundary and raw[start - 1:start] == echo)
        )
    )


def _is_adjacent_repetition_deletion(
    raw: str, start: int, end: int, compact: str
) -> bool:
    """Recognize deletion of one or more copies from an adjacent character run."""

    if not compact or len(set(compact)) != 1:
        return False
    if raw[start:end] != compact:
        return False
    unit = compact[0]
    if len(compact) == 1:
        return (
            raw[start - 1 : start] == unit
            or raw[end : end + 1] == unit
        )
    return (
        raw[start - 1 : start] == unit
        or raw[end : end + 1] == unit
    )


def _strip_edit_punctuation(value: str) -> str:
    return re.sub(r"[，,、。！？!?；;：:（）()【】\[\]《》\s]", "", value)


def _meaningful_edit_text(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fffA-Za-z0-9]", value))


def _correction_surface(text: str) -> str:
    """Return ``text`` with explicit numbers in one canonical surface form.

    The Refiner input has already passed through contextual ITN (``一个`` ->
    ``1个``), so the reference text and the candidate can legitimately differ
    only in number form.  Normalization is value preserving, which keeps a real
    value change from being mistaken for a formatting difference.
    """

    return _CORRECTION_SURFACE_NORMALIZER.normalize(text).text


def _matched_correction_tail_span(
    raw: str, refined: str
) -> tuple[int, int, str] | None:
    """Locate a retained replacement clause in the normalized source.

    The returned span is ``(allowed_start, tail_start, tail_text)``.  The
    interval before ``tail_start`` begins at the punctuation immediately
    preceding the correction marker; punctuation in that interval belongs to
    the superseded clause/marker boundary and may be removed when the marker
    is resolved.
    """

    raw_surface = _correction_surface(raw)
    refined_surface = _correction_surface(refined)
    refined_values = set(_numeric_values(refined_surface))
    for marker in _CORRECTION_MARKERS:
        # A chunk boundary can leave a dangling marker such as ``……不对，``
        # at the very end of a window.  That occurrence has no replacement
        # clause, so scan backwards for the most recent marker that does.
        offsets = [m.start() for m in re.finditer(re.escape(marker), raw_surface)]
        for marker_start in reversed(offsets):
            marker_end = marker_start + len(marker)
            remainder = raw_surface[marker_end:].lstrip(" ，,、")
            # Only the clause immediately following the correction marker is
            # the replacement target. Do not require later independent
            # sentences to survive verbatim, otherwise a valid correction is
            # mistaken for whole-window loss because another clause changed.
            tail = re.split(
                r"[,，、.。！？!?;；\n]", remainder, maxsplit=1
            )[0].strip(" ，,、")
            if not tail:
                continue
            tail_start = marker_end
            while (
                tail_start < len(raw_surface)
                and raw_surface[tail_start] in " ，,、"
            ):
                tail_start += 1

            # The refiner may copy the replacement clause or restate it while
            # changing only its numeric surface form.
            tail_values = _numeric_values(tail)
            if tail not in refined_surface and not (
                tail_values
                and all(value in refined_values for value in tail_values)
            ):
                continue

            allowed_start = marker_start
            for index in range(marker_start - 1, -1, -1):
                if _is_window_punctuation(raw_surface, index):
                    allowed_start = index
                    break
            # A filler can sit between the abandoned clause and the explicit
            # correction (``苹果，嗯，不对，我有梨``).  Its punctuation is
            # part of the superseded prefix too, so widen the exempted span
            # across filler-only material.  Do not cross ordinary lexical
            # content or an unrelated sentence boundary.
            while allowed_start > 0:
                previous_mark = None
                for index in range(allowed_start - 1, -1, -1):
                    if _is_window_punctuation(raw_surface, index):
                        previous_mark = index
                        break
                if previous_mark is None:
                    break
                filler_surface = raw_surface[previous_mark + 1 : allowed_start]
                tokens = [
                    token
                    for token in re.split(r"[，,、；;：:\s]+", filler_surface)
                    if token
                ]
                if not tokens or not all(
                    token in _CORRECTION_FILLER_TOKENS for token in tokens
                ):
                    break
                allowed_start = previous_mark
            return allowed_start, tail_start, tail
    return None


def _matched_correction_tail(raw: str, refined: str) -> str | None:
    """Return the replacement clause that ``refined`` still keeps, if any."""

    span = _matched_correction_tail_span(raw, refined)
    return span[2] if span is not None else None


def _retains_correction_tail(raw: str, refined: str) -> bool:
    """Allow a compact rewrite when it keeps the corrected clause."""

    return _matched_correction_tail(raw, refined) is not None


def _correction_tail_values(
    raw: str, refined: str
) -> tuple[tuple[Decimal, str], ...]:
    """Return the numeric values stated by the kept replacement clause."""

    tail = _matched_correction_tail(raw, refined)
    return _numeric_values(tail) if tail else ()


def _self_correction_count(text: str) -> int:
    """Count clause-initial self-correction markers such as ``不对``.

    A marker only counts when it starts a clause and is followed by the
    replacement content.  This keeps ordinary negations (``我不是学生``) and the
    ``是`` in ``不是 A 而是 B`` from looking like an extra correction.
    """

    break_chars = "，,、。！？!?；;\n"
    count = 0
    for match in _CORRECTION_MARKER_RE.finditer(text):
        before = text[match.start() - 1 : match.start()]
        after = text[match.end():].lstrip(" ，,、")
        if after and (not before or before in break_chars):
            count += 1
    return count


def _is_intentional_correction_compression(
    raw: str, refined: str, length_ratio: float
) -> bool:
    """Exempt a deliberate self-correction collapse from content-loss checks.

    A single correction must still leave enough of the window intact.  A
    multi-stage chain legitimately collapses further, because every clause
    before the last marker is a superseded false start.
    """

    if not _retains_correction_tail(raw, refined):
        return False
    return length_ratio >= 0.45 or _self_correction_count(raw) >= 2


def _numeric_values(text: str) -> tuple[tuple[Decimal, str], ...]:
    """Extract ordered numeric values with a small amount of unit context.

    Standalone Chinese ``一``/``两`` in ordinary classifier phrases is not
    treated as a numeric assertion unless it is followed by a number unit.
    This keeps normal wording cleanup from being rejected while protecting
    prices, percentages, dates, and multi-digit Chinese numerals.
    """

    values: list[tuple[int, int, Decimal, str]] = []
    occupied: list[tuple[int, int]] = []

    # Protected-span placeholders such as ``__ENTITY_000__`` must not be
    # parsed as Arabic numerals.  Masking a verified entity turns it into a
    # numbered placeholder; treating that id (``000``/``001``) as a numeric
    # value makes the masked baseline and the restored text compare unequal
    # and spuriously rejects an otherwise valid refinement with
    # ``numeric_value_mismatch``.
    for match in _PLACEHOLDER_RE.finditer(text):
        occupied.append((match.start(), match.end()))

    # Fixed idioms contain numerals that are not numbers (一五一十, 三番五次,
    # 乱七八糟).  Exclude their spans so they are never parsed as numeric
    # assertions: otherwise ``一五一十`` can be misread as 10 and a valid
    # unchanged idiom can look like a numeric change.
    for expression in _FIXED_EXPRESSIONS:
        for match in re.finditer(re.escape(expression), text):
            occupied.append((match.start(), match.end()))

    # Chinese percentages may carry a decimal part (百分之十二点五 -> 12.5%).
    for match in re.finditer(
        r"百分之([零〇一二三四五六七八九十百千万亿两]+(?:点[零〇一二三四五六七八九]+)?)",
        text,
    ):
        value = _parse_chinese_number(match.group(1))
        if value is None:
            continue
        start, end = match.span()
        values.append((start, end, value, "percent"))
        occupied.append((start, end))

    # Chinese clock times ``七点十分`` split into (hour, minute) components so
    # they align with the Arabic form ``7点10分``.
    for match in re.finditer(
        r"([零〇一二三四五六七八九十百千万亿两]+)点([零〇一二三四五六七八九十百千万亿两]+)分(?:钟)?",
        text,
    ):
        start, end = match.span()
        if _overlaps(start, end, occupied):
            continue
        hour = _parse_chinese_number(match.group(1))
        minute = _parse_chinese_number(match.group(2))
        if hour is None or minute is None:
            continue
        hour_start = match.start(1)
        minute_start = match.start(2)
        values.append((hour_start, hour_start + 1, hour, ""))
        values.append((minute_start, match.end(2), minute, ""))
        occupied.append((start, end))

    # Hybrid large-unit forms are the normal written result of ITN:
    # ``六万 -> 6万`` and ``一点七亿 -> 1.7亿``.  Parse the Arabic
    # coefficient together with the Chinese multiplier so both surfaces have
    # one comparable semantic value.  The Chinese-decimal form needs the same
    # treatment because the general numeral regex intentionally ends its
    # fractional part before ``万``/``亿``.
    for match in re.finditer(
        r"([零〇一二三四五六七八九十百千两]+点[零〇一二三四五六七八九]+)([万亿])",
        text,
    ):
        start, end = match.span()
        if _overlaps(start, end, occupied):
            continue
        coefficient = _parse_chinese_number(match.group(1))
        if coefficient is None:
            continue
        multiplier = Decimal(_CHINESE_LARGE_UNITS[match.group(2)])
        values.append(
            (start, end, coefficient * multiplier, _unit_for(text[end : end + 1]))
        )
        occupied.append((start, end))

    for match in re.finditer(r"([+-]?\d+(?:\.\d+)?)([万亿])", text):
        start, end = match.span()
        if _overlaps(start, end, occupied):
            continue
        try:
            coefficient = Decimal(match.group(1))
        except InvalidOperation:
            continue
        multiplier = Decimal(_CHINESE_LARGE_UNITS[match.group(2)])
        values.append(
            (start, end, coefficient * multiplier, _unit_for(text[end : end + 1]))
        )
        occupied.append((start, end))

    for match in _ARABIC_NUMERAL_RE.finditer(text):
        start, end = match.span()
        if _overlaps(start, end, occupied):
            continue
        token = match.group(0)
        suffix = text[end : end + 1]
        unit = "percent" if suffix in {"%", "％"} else _unit_for(suffix)
        # Mirror the Chinese-numeral guard: a single bare Arabic digit that is
        # not part of an explicit numeric context (unit, percent, decimal,
        # sign, or a multi-digit run) is not treated as a numeric assertion.
        # Otherwise idioms such as ``三番五次 -> 3番5次`` would look like a
        # numeric change and spuriously reject a valid cleanup with
        # ``numeric_value_mismatch``.
        if (
            len(token) == 1
            and not unit
            and "." not in token
            and suffix not in "点"
            and not (token[0] in "+-")
        ):
            continue
        try:
            value = Decimal(token)
        except InvalidOperation:
            continue
        values.append((start, end, value, unit))

    for match in _CHINESE_NUMERAL_RE.finditer(text):
        start, end = match.span()
        if _overlaps(start, end, occupied):
            continue
        token = match.group(0)
        suffix = text[end : end + 1]
        if _ambiguous_chinese_number(token, text, end):
            continue
        # Skip ordinary classifier phrases with an ambiguous digit run
        # (``二三个人``), but keep dates and values with an explicit unit.  A
        # single glyph such as ``十`` is different from a bare digit: it is a
        # positional Chinese numeral whose value is 10.  It must be retained
        # here so ``十个`` and the equivalent ``10个`` compare equally.  The
        # same applies to ``百``/``千``/``万``/``亿``; fixed idioms and
        # ambiguous adjacent-digit runs are excluded above.
        if (
            len(token) == 1
            and token in _CHINESE_DIGITS
            and suffix not in _CHINESE_NUMERIC_UNITS
            and suffix not in _CHINESE_COUNT_UNITS
            and suffix not in _CHINESE_MULTIPLIER_UNITS
        ):
            continue
        value = _parse_chinese_number(token)
        if value is None:
            continue
        values.append((start, end, value, _unit_for(suffix)))

    values.sort(key=lambda item: item[0])
    return tuple((value, unit) for _, _, value, unit in values)


def _numeric_edit_is_equivalent(source: str, target: str) -> bool:
    """Return whether an edit only restates the same number differently."""

    source_values = _numeric_values(source)
    if not source_values or source_values != _numeric_values(target):
        return False
    return _numeric_skeleton(source) == _numeric_skeleton(target)


def preserve_safe_numeric_edits(raw: str, refined: str) -> str | None:
    """Keep equivalent numeric surface edits from an otherwise rejected output.

    Integrity validation is intentionally conservative for semantic deletions,
    but rejecting a whole window can hide an independently safe conversion such
    as a Chinese count to Arabic digits.  This helper projects only aligned
    replacements whose surrounding numeric values and shells are equivalent
    back onto the original text. All other model edits are discarded.
    """

    source = raw.strip()
    candidate = refined.strip()
    if not source or not candidate or source == candidate:
        return None

    edits: list[tuple[int, int, str]] = []
    matcher = SequenceMatcher(None, source, candidate, autojunk=False)
    # Keep the comparison local. A whole sentence may contain an unrelated
    # rejected deletion; including it would hide an otherwise equivalent
    # numeric edit from this salvage pass.
    context_radius = 8
    boundary_chars = frozenset("\uff0c,\u3001\u3002\uff01\uff1f!?\uff1b;\uff1a:\n")

    def local_context(text: str, start: int, end: int) -> str:
        left_boundary = max(
            (index for index, char in enumerate(text[:start]) if char in boundary_chars),
            default=-1,
        )
        right_candidates = [
            index for index in range(end, len(text)) if text[index] in boundary_chars
        ]
        right_boundary = (right_candidates[0] + 1) if right_candidates else len(text)
        if right_boundary - left_boundary <= 40:
            return text[left_boundary + 1 : right_boundary]
        return text[max(0, start - context_radius) : min(len(text), end + context_radius)]
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag != "replace" or source_start == source_end or target_start == target_end:
            continue
        source_fragment = source[source_start:source_end]
        target_fragment = candidate[target_start:target_end]
        # SequenceMatcher normally isolates a numeric surface replacement
        # from nearby punctuation.  Prefer that exact replacement when it is
        # independently equivalent: another rejected edit may have removed a
        # comma immediately after the number, which would otherwise make the
        # punctuation-bounded contexts differ and hide the safe conversion.
        if _numeric_edit_is_equivalent(source_fragment, target_fragment):
            edits.append((source_start, source_end, target_fragment))
            continue
        source_context = local_context(source, source_start, source_end)
        target_context = local_context(candidate, target_start, target_end)
        if not _numeric_edit_is_equivalent(source_context, target_context):
            continue
        edits.append((source_start, source_end, candidate[target_start:target_end]))

    if not edits:
        return None
    repaired = source
    for start, end, replacement in reversed(edits):
        repaired = repaired[:start] + replacement + repaired[end:]
    if (
        repaired == source
        or not _numeric_surface_only(source, repaired)
        or _partially_converted_numeric_range(source, repaired)
    ):
        return None
    return repaired


def preserve_safe_repetition_edits(raw: str, refined: str) -> str | None:
    """Project independently provable adjacent repetition deletions.

    A rejected refinement can contain a valid stutter cleanup next to an
    unsafe rewrite.  This helper starts from the source and accepts only
    delete opcodes whose entire source span is an adjacent repeated-character
    run or an exact adjacent repeated phrase.  Insertions, substitutions,
    punctuation edits, numbers, entities, and unsupported semantic deletions
    are never copied from the rejected candidate.
    The result is checked by the same guard before it is returned.
    """

    source = raw.strip()
    candidate = refined.strip()
    if not source or not candidate or source == candidate:
        return None

    edits: list[tuple[int, int, str]] = []
    matcher = SequenceMatcher(None, source, candidate, autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag != "delete" or source_start == source_end:
            continue
        source_fragment = source[source_start:source_end]
        # The model may delete one repeated phrase together with an unrelated
        # particle: 我们我们就假设 -> 我们假设.  Recover only the exact adjacent
        # repeated phrase, leaving the particle and all other source text in
        # place.  This uses the model's deletion as evidence but never copies
        # its wider rewrite into the fallback.
        phrase_width = next(
            (
                width
                for width in range(min(6, len(source_fragment)), 1, -1)
                if len(set(source_fragment[:width])) > 1
                and source[:source_start].endswith(source_fragment[:width])
            ),
            0,
        )
        if phrase_width:
            edits.append((source_start, source_start + phrase_width, ""))
            continue
        compact = _strip_edit_punctuation(source_fragment)
        if compact != source_fragment or not compact:
            continue
        single_char_run = len(set(compact)) == 1
        if single_char_run:
            if _is_protected_reduplication_deletion(
                source, source_start, source_end, compact
            ):
                continue
            if not _is_adjacent_repetition_deletion(
                source, source_start, source_end, compact
            ):
                continue
        elif not (
            len(compact) >= 2
            and (
                source[:source_start].endswith(compact)
                or source[source_end:].startswith(compact)
            )
        ):
            continue
        edits.append((source_start, source_end, ""))

    if not edits:
        return None
    repaired = source
    for start, end, replacement in reversed(edits):
        repaired = repaired[:start] + replacement + repaired[end:]
    if repaired == source or reject_reasons(source, repaired):
        return None
    return repaired


def _numeric_context_is_lost(source: str, target: str) -> bool:
    """Return whether an equivalent local value lost its semantic shell.

    Approximation, range, and boundary wording remains ordinary text in the
    numeric skeleton.  The check is therefore vocabulary-independent: ``多``,
    ``来``, ``左右``, ``上下``, ``不到`` and future forms all follow the same
    rule instead of needing to be enumerated as numeric suffixes.
    """

    source_values = _numeric_values(source)
    if not source_values or source_values != _numeric_values(target):
        return False
    return _numeric_skeleton(source) != _numeric_skeleton(target)


def _numeric_surface_only(raw: str, refined: str) -> bool:
    """Return whether the textual difference is only numeric formatting."""

    if not _numeric_values(raw) or _numeric_values(raw) != _numeric_values(refined):
        return False
    return _numeric_skeleton(raw) == _numeric_skeleton(refined)


def _numeric_skeleton(text: str) -> str:
    # Keep protected placeholders intact so their id digits are not treated as
    # numeric tokens and do not leak into the numeric-surface comparison.
    text = _PLACEHOLDER_RE.sub("<ENTITY>", text)
    text = re.sub(
        r"百分之[零〇一二三四五六七八九十百千万亿两]+(?:点[零〇一二三四五六七八九]+)?",
        "<NUM>",
        text,
    )
    text = _ARABIC_NUMERAL_RE.sub("<NUM>", text)
    text = _CHINESE_NUMERAL_RE.sub("<NUM>", text)
    return text.replace("%", "").replace("％", "")


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def _unit_for(value: str) -> str:
    if value in {"%", "％"}:
        return "percent"
    if value in {"年", "月", "日", "号"}:
        return "date"
    if value in {"元", "块"}:
        return "currency"
    if value in _CHINESE_COUNT_UNITS:
        return "count"
    if value in _CHINESE_MULTIPLIER_UNITS:
        return "multiplier"
    return ""


def _ambiguous_chinese_number(token: str, text: str, end: int) -> bool:
    """Keep list-like Chinese digit runs out of numeric checks.

    Approximation wording after a parseable value is deliberately not handled
    here.  The numeric core is compared by value and its surrounding wording
    is preserved by ``_numeric_context_is_lost``.
    """

    # A date year such as ``二零一五年`` is an explicit numeric assertion;
    # other bare adjacent digits such as ``二三个人`` commonly mean an
    # approximation and must not be equated with ``23个人``.
    if len(token) >= 2 and all(char in _CHINESE_DIGITS for char in token):
        suffix = text[end : end + 1]
        if not suffix or suffix not in "年月日号":
            return True
    first_unit = next(
        (index for index, char in enumerate(token) if char in "十百千万亿"),
        None,
    )
    if first_unit is not None and first_unit >= 2:
        prefix = token[:first_unit]
        if all(char in _CHINESE_DIGITS for char in prefix):
            return True
    return False


def _parse_chinese_number(token: str) -> Decimal | None:
    if not token:
        return None
    # Chinese decimals: ``十二点五`` -> 12.5, ``零点五`` -> 0.5.
    if "点" in token:
        integer_part, _, fractional_part = token.partition("点")
        integer = _parse_chinese_number(integer_part) if integer_part else Decimal(0)
        if integer is None:
            return None
        if not fractional_part or any(
            char not in _CHINESE_DIGITS for char in fractional_part
        ):
            return None
        fraction = "".join(str(_CHINESE_DIGITS[char]) for char in fractional_part)
        return Decimal(f"{int(integer)}.{fraction}")
    if not any(char in _CHINESE_SMALL_UNITS or char in _CHINESE_LARGE_UNITS for char in token):
        digits = [_CHINESE_DIGITS.get(char) for char in token]
        if any(value is None for value in digits):
            return None
        return Decimal("".join(str(value) for value in digits))

    total = 0
    section = 0
    number = 0
    for char in token:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
        elif char in _CHINESE_SMALL_UNITS:
            number = number or 1
            section += number * _CHINESE_SMALL_UNITS[char]
            number = 0
        elif char in _CHINESE_LARGE_UNITS:
            section += number
            section = section or 1
            total += section * _CHINESE_LARGE_UNITS[char]
            section = 0
            number = 0
        else:
            return None
    return Decimal(total + section + number)
