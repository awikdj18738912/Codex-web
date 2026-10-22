"""Opt-in ASR risk review. Scores can add review, never cancel native review."""
from __future__ import annotations

from dataclasses import replace
from math import isfinite

from .asr_window_confidence import locate_chunks, score_source_range
from .refinement_guard import split_for_refinement

MODES = ("off", "shadow", "on")


def compose_projections(parts):
    """Join VAD states in the same coordinate space as the ASR transcript.

    Unsupported/missing states retain their text but contribute no evidence.
    Evidence identities are namespaced because generation revisions restart.
    """
    text, spans = "", []
    for index, part in enumerate(parts):
        value = part.get("text", "").strip()
        if not value:
            continue
        separator = " " if text and text[-1].isascii() and value[0].isascii() else ""
        offset = len((text + separator).encode("utf-8"))
        if part.get("status") == "mapped" and part.get("text") == value:
            for span in part.get("spans", []):
                spans.append({**span, "start": offset + span["start"],
                              "end": offset + span["end"],
                              "evidence_id": f"{index}:{span['evidence_id']}"})
        text += separator + value
    return dict(status="mapped", text=text, spans=spans,
                coordinate_space="raw_ASR_UTF8_bytes", calibrated=False,
                score_source="chosen_generation_token_logprob_with_retained_prefix_history",
                production_skip_allowed=False)


def window_evidence(source, projection):
    """Locate exact window text; whitespace removal is the only tolerated edit.

    Repeated windows are intentionally unknown instead of borrowing another
    occurrence's score. Any stale or incomplete evidence remains unknown.
    """
    unknown = dict(status="unknown", reason="source_evidence_unavailable", fragments=[],
                   calibrated=False)
    if not isinstance(projection, dict) or projection.get("status") != "mapped":
        return unknown
    text = projection.get("text")
    if not isinstance(text, str) or projection.get("coordinate_space") != "raw_ASR_UTF8_bytes":
        return unknown
    spans = projection.get("spans")
    if not isinstance(spans, list):
        return unknown | dict(reason="invalid_source_evidence")
    byte_length = len(text.encode("utf-8"))
    previous_end = 0
    for span in spans:
        if not isinstance(span, dict):
            return unknown | dict(reason="invalid_source_evidence")
        a, b, width = (span.get(key) for key in ("start", "end", "original_width"))
        value = span.get("logprob")
        if (any(not isinstance(item, int) or isinstance(item, bool) for item in (a, b, width))
                or not previous_end <= a < b <= byte_length or width < b-a
                or (value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                          or not isfinite(value) or value > 0))):
            return unknown | dict(reason="invalid_source_evidence")
        previous_end = b
    positions = [index for index, char in enumerate(text) if not char.isspace()]
    compact = "".join(text[index] for index in positions)
    target = "".join(source.split())
    start = compact.find(target) if target else -1
    if start < 0 or compact.find(target, start + 1) >= 0:
        return unknown | dict(reason="source_missing_or_ambiguous")
    a, b = positions[start], positions[start + len(target) - 1] + 1
    try:
        chunks = split_for_refinement(text[a:b], one_punctuation_window=True)
        offsets = locate_chunks(text[a:b], chunks)
        fragments = [dict(text=chunk, start=a + left, end=a + right,
                          **score_source_range(projection, a + left, a + right))
                     for chunk, (left, right) in zip(chunks, offsets)]
    except (KeyError, TypeError, ValueError, OverflowError):
        return unknown | dict(reason="invalid_source_evidence")
    full = bool(fragments) and all(item["coverage"] == 1. and item["score"] is not None
                                   and isfinite(item["score"]) and 0 <= item["score"] <= 1
                                   for item in fragments)
    return dict(status="complete" if full else "unknown", start=a, end=b,
                reason="source_aligned" if full else "source_evidence_incomplete",
                fragments=fragments, calibrated=False)


class ConfidenceSupplement:
    def __init__(self, mode="off", *, low_threshold=.8):
        if mode not in MODES:
            raise ValueError("confidence supplement mode must be off, shadow or on")
        if (isinstance(low_threshold, bool) or not isfinite(low_threshold)
                or not 0 < low_threshold < 1):
            raise ValueError("confidence supplement threshold must be finite and in (0, 1)")
        self.mode, self.low_threshold = mode, float(low_threshold)

    def config_dict(self):
        return dict(mode=self.mode, low_fragment_threshold=self.low_threshold,
                    missing_evidence="review", high_score_veto=False, calibrated=False,
                    purpose="supplement_native_review_not_certify_cleanup_need")

    def decide(self, native, source, projection):
        audit = dict(**self.config_dict(), native_action=native.public_dict()["action"],
                     suggested_action=native.public_dict()["action"], applied=False)
        if self.mode == "off":
            return native, audit | dict(reason="disabled")
        # Preserve scheduling, already-processed source and mandatory reviews.
        if native.should_refine or native.reasons[0] not in {
            "stable_clean_segment", "confidence_unverified_safe_keep",
            "high_confidence_clean_segment", "numeric_handled_by_rule_pack",
        }:
            return native, audit | dict(reason="native_action_preserved")
        evidence = window_evidence(source, projection)
        audit["evidence"] = evidence
        low = [item for item in evidence["fragments"]
               if isinstance(item.get("score"), (int, float))
               and isfinite(item["score"]) and item["score"] < self.low_threshold]
        review = bool(low) or evidence["status"] != "complete"
        reason = ("low_confidence_fragment_review" if low else
                  "confidence_evidence_unknown_review" if review else "no_extra_risk_detected")
        audit.update(reason=reason, suggested_action="refine" if review else "keep",
                     applied=review and self.mode == "on")
        if audit["applied"]:
            return replace(native, should_refine=True, action="refine", reasons=(reason,)), audit
        return native, audit
