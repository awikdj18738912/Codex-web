"""Decoupled, deterministic routing gate for ASR text refinement.

The gate only decides whether the neural Refiner should run.  Entity
normalization and output-integrity validation remain separate stages, so the
gate can be disabled for an exact A/B baseline without changing either one.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Iterable

from .numeric_normalizer import _COUNT_UNITS, _MEASURE_UNITS
from .refinement_guard import detect_boundary_anomalies


_SENTENCE_ENDINGS = frozenset("。！？!?；;")
_VISIBLE_RE = re.compile(r"[\w\u3400-\u9fff]", re.UNICODE)
_REPEATED_CJK_RE = re.compile(r"([\u3400-\u9fff])\1{2,}")
_REPEATED_SINGLE_RE = re.compile(
    r"(?:^|[，,、。！？!?；;\s])"
    r"([\u3400-\u9fff])(?:\1|[，,、。！？!?；;\s]+\1)+"
)
_REPEATED_PHRASE_RE = re.compile(r"([\u3400-\u9fff]{2,8})(?:[，,、 ]?\1){1,}")
_DISFLUENCY_RE = re.compile(
    r"(?:^|[，,。！？!?；;、\s])(?:嗯+|呃+|额+|啊+|哎+|呀+|嘿+|呼+|喂|这个|那个|就是)(?=$|[，,。！？!?；;、\s])"
)
_EMBEDDED_DISFLUENCY_RE = re.compile(r"(?:嗯+|呃+)")
_SELF_CORRECTION_RE = re.compile(
    r"(?:不对|我说错了|说错了|应该是|应该说|准确地说|我是说|改成|更正一下)"
)
_CROSS_CLAUSE_PUNCTUATION_RE = re.compile(
    r"[\u3400-\u9fff][。！？!?][\u3400-\u9fff]"
)
_NUMERIC_CLASSIFIER_PATTERN = "|".join(
    sorted(
        map(re.escape, dict.fromkeys((*_COUNT_UNITS, *_MEASURE_UNITS, "座"))),
        key=len,
        reverse=True,
    )
)
# This is a routing hint only. Multi-character Chinese numbers are allowed
# without a following unit, while a single digit needs numeric grammar (a
# classifier/measure, percentage, ordinal, or decimal context). This avoids
# waking the model for lexical uses such as ``不值一提`` without maintaining
# sentence- or phrase-specific exceptions.
_NUMERIC_NORMALIZATION_RE = re.compile(
    r"百分之[零〇一二三四五六七八九十百千万亿两0-9]+"
    r"|第[零〇一二三四五六七八九十百千万亿两]+"
    r"|[零〇一二三四五六七八九十百千万亿两]{2,}"
    r"(?:点[零〇一二三四五六七八九]+)?"
    rf"|[零〇一二三四五六七八九两](?:{_NUMERIC_CLASSIFIER_PATTERN})"
)


class RefinementGateMode(str, Enum):
    """Supported experiment modes."""
    TRI_STATE = "tri_state"

    OFF = "off"
    CONSERVATIVE = "conservative"

    @classmethod
    def parse(cls, value: str | "RefinementGateMode") -> "RefinementGateMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(value.strip().lower())
        except (AttributeError, ValueError) as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"refinement gate mode must be one of: {choices}"
            ) from error


@dataclass(frozen=True, slots=True)
class RefinementGateDecision:
    """One auditable routing decision for one source segment."""

    mode: str
    should_refine: bool
    reasons: tuple[str, ...]
    visible_chars: int
    asr_confidence: float | None
    cleanup_signals: tuple[str, ...]
    calibrated: bool = False
    covers_segment: bool = False
    action: str | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "action": self.action or ("refine" if self.should_refine else "skip"),
            "state": (self.action or ("refine" if self.should_refine else "keep")).upper(),
            "reasons": list(self.reasons),
            "visible_chars": self.visible_chars,
            "asr_confidence": self.asr_confidence,
            "calibrated": self.calibrated,
            "covers_segment": self.covers_segment,
            "cleanup_signals": list(self.cleanup_signals),
        }

@dataclass(frozen=True, slots=True)
class HypothesisStability:
    """Backend-neutral stability snapshot for one cumulative ASR update."""

    text: str
    unchanged_updates: int
    revision_ratio: float
    tail_age_ms: float
    sentence_complete: bool
    stable: bool


class HypothesisTracker:
    """Track whether a non-final ASR tail is ready for processing."""

    def __init__(self, *, stable_updates: int = 2, stable_ms: float = 600.0) -> None:
        if stable_updates < 1:
            raise ValueError("stable_updates must be positive")
        if stable_ms < 0:
            raise ValueError("stable_ms must be non-negative")
        self.stable_updates = stable_updates
        self.stable_ms = float(stable_ms)
        self._text = ""
        self._unchanged_updates = 0
        self._changed_at = time.monotonic()

    def observe(self, text: str, *, now: float | None = None) -> HypothesisStability:
        source = text.strip()
        timestamp = time.monotonic() if now is None else float(now)
        previous = self._text
        if source == previous and source:
            self._unchanged_updates += 1
        else:
            self._unchanged_updates = 0
            self._changed_at = timestamp
        self._text = source
        revision_ratio = (
            1.0 - SequenceMatcher(None, previous, source, autojunk=False).ratio()
            if previous else 0.0
        )
        tail_age_ms = max(0.0, (timestamp - self._changed_at) * 1000.0)
        sentence_complete = bool(source) and source[-1] in _SENTENCE_ENDINGS
        # Streaming ASR usually emits a cumulative prefix that grows by a
        # few characters on every update. Treat a low-edit append as stable
        # enough to enter the gate; reserve DEFER for real rewrites or a
        # brand-new tail. This keeps asynchronous refinement alive without
        # refining every transient typo.
        low_edit_growth = bool(previous) and source != previous and revision_ratio <= 0.20
        stable = bool(source) and (
            self._unchanged_updates >= self.stable_updates
            or low_edit_growth
            or (
                sentence_complete
                and self._unchanged_updates >= 1
                and tail_age_ms >= self.stable_ms
            )
        )
        return HypothesisStability(
            source, self._unchanged_updates, revision_ratio, tail_age_ms,
            sentence_complete, stable
        )


class RefinementGate:
    """Conservative pre-Refiner gate with no model or database dependency.

    In ``off`` mode every non-empty segment follows the original Refiner path.
    In ``conservative`` mode the gate skips complete high-confidence text with
    no visible cleanup signal. Missing ASR confidence, calibration evidence,
    or complete segment coverage is treated as uncertainty and therefore does
    not skip useful refinement work.
    """

    def __init__(
        self,
        mode: str | RefinementGateMode = RefinementGateMode.OFF,
        *,
        high_confidence: float = 0.92,
        refine_on_unverified_confidence: bool = True,
    ) -> None:
        self.mode = RefinementGateMode.parse(mode)
        if not 0 <= high_confidence <= 1:
            raise ValueError("high_confidence must be between 0 and 1")
        self.high_confidence = high_confidence
        # Kept configurable for A/B compatibility.  The Web service enables
        # the fail-closed value because Qwen currently reports only a short,
        # uncalibrated generated suffix rather than confidence for the full
        # segment.
        self.refine_on_unverified_confidence = bool(refine_on_unverified_confidence)

    def config_dict(self) -> dict[str, object]:
        """Return the effective settings for experiment manifests and logs."""

        return {
            "mode": self.mode.value,
            "high_confidence": self.high_confidence,
            "refine_on_unverified_confidence": self.refine_on_unverified_confidence,
        }

    def decide(
        self,
        text: str,
        *,
        asr_confidence: float | None = None,
        entity_hints: Iterable[str] = (),
        calibrated: bool = False,
        covers_segment: bool = False,
        is_final: bool = False,
        stable: bool = True,
        tail_age_ms: float = 0.0,
        revision_ratio: float = 0.0,
        unchanged_updates: int = 0,
        cooldown_active: bool = False,
        same_as_last_refined: bool = False,
        numeric_refinement: bool = False,
    ) -> RefinementGateDecision:
        source = text.strip()
        visible_chars = len(_VISIBLE_RE.findall(source))
        signals = _cleanup_signals(source)
        hints = tuple(item.strip() for item in entity_hints if item.strip())

        if self.mode is RefinementGateMode.TRI_STATE:
            return self._decide_tri_state(
                source,
                visible_chars=visible_chars,
                asr_confidence=asr_confidence,
                calibrated=calibrated,
                covers_segment=covers_segment,
                signals=signals,
                hints=hints,
                is_final=is_final,
                stable=stable,
                tail_age_ms=tail_age_ms,
                revision_ratio=revision_ratio,
                unchanged_updates=unchanged_updates,
                cooldown_active=cooldown_active,
                same_as_last_refined=same_as_last_refined,
                numeric_refinement=numeric_refinement,
            )

        if self.mode is RefinementGateMode.OFF:
            return self._decision(
                True, "gate_disabled", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        if not source:
            return self._decision(
                False, "empty_segment", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        # An entity hint means the segment contains an unresolved candidate.
        # Never suppress the only context-aware correction opportunity.
        if hints:
            return self._decision(
                True, "entity_hint_present", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        if signals:
            return self._decision(
                True, "cleanup_signal_present", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        if asr_confidence is None:
            return self._decision(
                True, "confidence_unavailable", visible_chars, None,
                calibrated, covers_segment, signals
            )
        if not calibrated:
            return self._decision(
                True, "confidence_uncalibrated", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        if not covers_segment:
            return self._decision(
                True, "confidence_coverage_incomplete", visible_chars,
                asr_confidence, calibrated, covers_segment, signals
            )
        if asr_confidence < self.high_confidence:
            return self._decision(
                True, "confidence_below_skip_threshold", visible_chars,
                asr_confidence, calibrated, covers_segment, signals
            )
        if source[-1] not in _SENTENCE_ENDINGS:
            return self._decision(
                True, "sentence_incomplete", visible_chars, asr_confidence,
                calibrated, covers_segment, signals
            )
        return self._decision(
            False, "high_confidence_clean_segment", visible_chars,
            asr_confidence, calibrated, covers_segment, signals
        )

    def _decide_tri_state(
        self,
        source: str,
        *,
        visible_chars: int,
        asr_confidence: float | None,
        calibrated: bool,
        covers_segment: bool,
        signals: tuple[str, ...],
        hints: tuple[str, ...],
        is_final: bool,
        stable: bool,
        tail_age_ms: float,
        revision_ratio: float,
        unchanged_updates: int,
        cooldown_active: bool,
        same_as_last_refined: bool,
        numeric_refinement: bool,
    ) -> RefinementGateDecision:
        """Route a cumulative hypothesis to keep, defer, or refine.

        ``defer`` is limited to non-final unstable hypotheses; finalization
        always resolves to either ``keep`` or ``refine``. Numeric signals can
        wake the neural Refiner when model-driven numeric normalization is
        enabled by the caller.
        """
        del unchanged_updates  # retained for future policy tuning and auditability
        if not source:
            return self._decision(
                False, "empty_segment", visible_chars, asr_confidence,
                calibrated, covers_segment, signals, action="keep"
            )
        if same_as_last_refined:
            return self._decision(
                False, "already_refined_hypothesis", visible_chars,
                asr_confidence, calibrated, covers_segment, signals,
                action="keep"
            )
        if not is_final and (cooldown_active or not stable):
            if cooldown_active:
                reason = "refinement_cooldown"
            elif tail_age_ms < 600.0:
                reason = "tail_too_young"
            elif revision_ratio > 0.12:
                reason = "revision_ratio_high"
            else:
                reason = "unstable_hypothesis"
            return self._decision(
                False, reason, visible_chars, asr_confidence,
                calibrated, covers_segment, signals, action="defer"
            )

        # Entity hints and human disfluency signals are high-value reasons to
        # spend a model call. In model-driven numeric mode, a numeric signal
        # is also a reason to invoke the Refiner; in deterministic mode it is
        # handled by the numeric normalizer instead.
        model_signals = tuple(
            signal
            for signal in signals
            if numeric_refinement or signal != "numeric_normalization"
        )
        if hints:
            return self._decision(
                True, "entity_hint_present", visible_chars, asr_confidence,
                calibrated, covers_segment, signals, action="refine"
            )
        if model_signals:
            return self._decision(
                True, "cleanup_signal_present", visible_chars, asr_confidence,
                calibrated, covers_segment, signals, action="refine"
            )
        confidence_usable = (
            asr_confidence is not None and calibrated and covers_segment
        )
        if confidence_usable and asr_confidence < 0.80:
            return self._decision(
                True, "confidence_below_refine_threshold", visible_chars,
                asr_confidence, calibrated, covers_segment, signals,
                action="refine"
            )
        if confidence_usable and asr_confidence >= self.high_confidence:
            return self._decision(
                False, "high_confidence_clean_segment", visible_chars,
                asr_confidence, calibrated, covers_segment, signals,
                action="keep"
            )
        if is_final:
            if self.refine_on_unverified_confidence:
                return self._decision(
                    True, "final_uncertain", visible_chars, asr_confidence,
                    calibrated, covers_segment, signals, action="refine"
                )
            if source[-1] not in _SENTENCE_ENDINGS:
                return self._decision(
                    True, "sentence_incomplete", visible_chars, asr_confidence,
                    calibrated, covers_segment, signals, action="refine"
                )
            return self._decision(
                False, "confidence_unverified_safe_keep", visible_chars,
                asr_confidence, calibrated, covers_segment, signals, action="keep"
            )
        if asr_confidence is None and self.refine_on_unverified_confidence:
            return self._decision(
                True, "confidence_unavailable", visible_chars, None,
                calibrated, covers_segment, signals, action="refine"
            )
        if not calibrated and self.refine_on_unverified_confidence:
            return self._decision(
                True, "confidence_uncalibrated", visible_chars,
                asr_confidence, calibrated, covers_segment, signals,
                action="refine"
            )
        if not covers_segment and self.refine_on_unverified_confidence:
            return self._decision(
                True, "confidence_coverage_incomplete", visible_chars,
                asr_confidence, calibrated, covers_segment, signals,
                action="refine"
            )
        if not is_final and revision_ratio > 0.20:
            return self._decision(
                True, "revision_ratio_high", visible_chars, asr_confidence,
                calibrated, covers_segment, signals, action="refine"
            )
        return self._decision(
            False, "stable_clean_segment", visible_chars, asr_confidence,
            calibrated, covers_segment, signals, action="keep"
        )

    def _decision(
        self,
        should_refine: bool,
        reason: str,
        visible_chars: int,
        asr_confidence: float | None,
        calibrated: bool,
        covers_segment: bool,
        signals: tuple[str, ...],
        *,
        action: str | None = None,
    ) -> RefinementGateDecision:
        return RefinementGateDecision(
            mode=self.mode.value,
            should_refine=should_refine,
            reasons=(reason,),
            visible_chars=visible_chars,
            asr_confidence=asr_confidence,
            cleanup_signals=signals,
            calibrated=calibrated,
            covers_segment=covers_segment,
            action=action,
        )


def _cleanup_signals(text: str) -> tuple[str, ...]:
    signals: list[str] = []
    if _REPEATED_CJK_RE.search(text) or _REPEATED_SINGLE_RE.search(text):
        signals.append("repeated_character")
    if _REPEATED_PHRASE_RE.search(text):
        signals.append("repeated_phrase")
    if _DISFLUENCY_RE.search(text):
        signals.append("disfluency")
    if _has_embedded_disfluency(text):
        signals.append("embedded_disfluency")
    if _SELF_CORRECTION_RE.search(text):
        signals.append("self_correction")
    if _NUMERIC_NORMALIZATION_RE.search(text):
        signals.append("numeric_normalization")
    if "  " in text or "，，" in text or "。。" in text:
        signals.append("malformed_spacing_or_punctuation")
    # In tri-state mode a request may contain the three adjacent punctuation
    # chunks used as context.  Give the Refiner one chance to repair an ASR
    # boundary error spanning those chunks, while clean single chunks remain
    # KEEP when confidence is unverified.
    if _CROSS_CLAUSE_PUNCTUATION_RE.search(text):
        signals.append("cross_clause_punctuation")
    if detect_boundary_anomalies(text):
        signals.append("boundary_anomaly")
    return tuple(signals)


def _has_embedded_disfluency(text: str) -> bool:
    """Detect low-ambiguity fillers attached to surrounding Chinese text."""

    for match in _EMBEDDED_DISFLUENCY_RE.finditer(text):
        suffix = text[match.end() : match.end() + 1]
        # Preserve the meaningful response 嗯哼 and the medical term 呃逆.
        if (match.group().startswith("嗯") and suffix == "哼") or (
            match.group().startswith("呃") and suffix == "逆"
        ):
            continue
        remainder = text[: match.start()] + text[match.end() :]
        if _VISIBLE_RE.search(remainder):
            return True
    return False
