"""Shared entity preparation and restoration used by every inference entry point."""

from __future__ import annotations

from dataclasses import dataclass

from .entity_matcher import EntityCandidateMatcher, EntityMatchReport, EntityDecision
from .protection import EntityProtector, ProtectionResult
from .entity_prose_projection import project_prose_entity
from .refinement_guard import (
    permits_self_correction_punctuation_repair,
    preserve_terminal_punctuation,
    reject_reasons,
    source_punctuation_lost,
    window_context_reject_reasons,
)

_TRUSTED_ENTITY_MATCH_TYPES = frozenset({'EXACT_ALIAS','EXACT_CANONICAL','EXACT_ACRONYM_PRESENTATION','CONTEXT_PINYIN','CONTEXT_OMISSION','CONTEXT_NEAR_PINYIN','CONTEXT_ALIAS_PINYIN','CONTEXT_ALIAS_NEAR_PINYIN'})


@dataclass(frozen=True, slots=True)
class PreparedEntitySegment:
    source_text: str
    protection: ProtectionResult
    baseline_text: str
    hints: tuple[str, ...]
    report: EntityMatchReport
    trusted_baseline_text: str = ""

    @property
    def refiner_candidates(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {"observed": m.observed, "canonical": m.canonical, "type": m.entity_type}
            for m in self.report.matches
            if m.decision is EntityDecision.HINT_ONLY
        )

    @property
    def normalizations(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "observed": span.original,
                "canonical": span.replacement,
                "entity_type": span.entity_type,
                "entity_id": span.entity_id,
                "match_type": span.match_type,
                "match_score": span.match_score,
            }
            for span in self.protection.spans
            if span.original != span.replacement
        )


@dataclass(frozen=True, slots=True)
class FinalizedEntitySegment:
    text: str
    accepted: bool
    reject_reasons: tuple[str, ...]
    normalizations: tuple[dict[str, object], ...] = ()


def fallback_entity_segment(prepared, reasons):
    """Retain verified aliases while discarding proposed fuzzy changes."""
    changes = tuple(
        c
        for c in prepared.normalizations
        if c["match_type"] in _TRUSTED_ENTITY_MATCH_TYPES
    )
    return FinalizedEntitySegment(
        prepared.trusted_baseline_text or prepared.baseline_text,
        False,
        tuple(reasons),
        changes,
    )


_PLACEHOLDER_REJECT_PREFIXES = (
    "placeholder_count:",
    "unknown_placeholders:",
    "placeholder_order_changed",
    "malformed_placeholder",
)


def has_placeholder_failure(reasons: tuple[str, ...]) -> bool:
    """Return whether rejection was caused by damaged protected markers."""

    return any(reason.startswith(_PLACEHOLDER_REJECT_PREFIXES) for reason in reasons)


def has_retryable_integrity_failure(
    reasons: tuple[str, ...], *, numeric_rule_mode: bool = False
) -> bool:
    """Return whether a stricter second refinement attempt can recover output."""

    # Placeholder damage has a targeted retry prompt. Numeric failures use
    # the validated source fallback instead of spending a second generation
    # on a request whose numbers already belong to the rule backend.
    if numeric_rule_mode and not has_placeholder_failure(reasons) and any(
        reason in {
            "numeric_value_mismatch", "numeric_rule_surface_mismatch",
            "partial_numeric_range_conversion",
        }
        for reason in reasons
    ):
        return False
    return has_placeholder_failure(reasons) or any(
        reason
        in {
            "severe_content_loss",
            "truncated_refiner_output",
            "entity_sentence_boundary_lost",
            "source_punctuation_lost",
            "numeric_value_mismatch",
            "semantic_substitution",
            "unsupported_insertion",
        }
        for reason in reasons
    )


def prepare_entity_segment(
    text: str,
    protector: EntityProtector,
    matcher: EntityCandidateMatcher | None,
    *,
    allow_auto: bool,
    confidence: float | None = None,
) -> PreparedEntitySegment:
    exact = protector.protect(text, confidence=confidence)
    trusted = protector.restore(exact.masked_text, exact)
    if not trusted.accepted:
        raise RuntimeError("internal trusted entity restoration failed")
    if matcher is None:
        report = EntityMatchReport((), 0.0)
    else:
        report = matcher.match(
            text,
            allow_auto=allow_auto,
            blocked_spans=((span.start, span.end) for span in exact.spans),
        )
    # Context spellings and explicitly enabled internal completions have passed
    # the entity verifier. Preserve them if later refinement fails.
    contextual_matches = tuple(m for m in report.auto_matches if m.match_type in {'CONTEXT_PINYIN','CONTEXT_OMISSION','CONTEXT_NEAR_PINYIN','CONTEXT_ALIAS_PINYIN','CONTEXT_ALIAS_NEAR_PINYIN'})
    if contextual_matches:
        contextual = protector.protect(text,confidence=confidence,fuzzy_matches=contextual_matches)
        contextual_trusted = protector.restore(contextual.masked_text,contextual)
        if not contextual_trusted.accepted:
            raise RuntimeError('internal contextual entity restoration failed')
        trusted = contextual_trusted
    protection = protector.protect(
        text,
        confidence=confidence,
        fuzzy_matches=report.auto_matches,
    )
    baseline = protector.restore(protection.masked_text, protection)
    if not baseline.accepted:
        raise RuntimeError(
            "internal entity baseline restoration failed: "
            + ",".join(baseline.reject_reasons)
        )
    max_hints = matcher.config.max_refiner_hints if matcher is not None else 16
    hints = tuple(
        dict.fromkeys(
            (*protector.refinement_hints(protection), *report.hint_canonicals)
        )
    )[:max_hints]
    return PreparedEntitySegment(
        text, protection, baseline.text, hints, report, trusted.text
    )


def finalize_entity_segment(
    refined_masked_text: str,
    prepared: PreparedEntitySegment,
    protector: EntityProtector,
    *,
    preserve_source_punctuation: bool = False,
    preserve_terminal_only: bool = False,
    allow_boundary_punctuation_repair: bool = False,
    numeric_rule_mode: bool = False,
    entity_patch: dict[str, object] | None = None,
    window_context: dict[str, str] | None = None,
) -> FinalizedEntitySegment:
    trusted_baseline = prepared.trusted_baseline_text or prepared.baseline_text
    trusted_changes = tuple(
        change
        for change in prepared.normalizations
        if change["match_type"] in _TRUSTED_ENTITY_MATCH_TYPES
    )
    restored = protector.restore(refined_masked_text, prepared.protection)
    if not restored.accepted:
        return FinalizedEntitySegment(
            trusted_baseline,
            False,
            restored.reject_reasons,
            trusted_changes,
        )
    restored_text = (
        preserve_terminal_punctuation(prepared.baseline_text, restored.text)
        if preserve_source_punctuation or preserve_terminal_only
        else restored.text
    )
    punctuation_reasons = (
        ("source_punctuation_lost",)
        if preserve_source_punctuation
        and not allow_boundary_punctuation_repair
        and source_punctuation_lost(prepared.baseline_text, restored_text)
        and not permits_self_correction_punctuation_repair(
            prepared.baseline_text, restored_text
        )
        else ()
    )
    reasons = (
        *punctuation_reasons,
        *_introduced_hint_reasons(prepared, restored_text),
        *_unconfirmed_entity_edit_reasons(prepared, restored_text, entity_patch),
        *reject_reasons(
            prepared.baseline_text,
            restored_text,
            numeric_rule_mode=numeric_rule_mode,
        ),
        *window_context_reject_reasons(
            prepared.baseline_text, restored_text, **(window_context or {}),
        ),
    )
    if reasons:
        # Legacy checkpoints can propose the right entity spelling in prose
        # while deleting a source character. Keep only a bounded entity edit;
        # never accept their non-entity rewrite or copied hint annotation.
        projection = (
            project_prose_entity(prepared, restored_text)
            if entity_patch is None else None
        )
        if projection is not None:
            projected_text, match = projection
            patch = {
                "action": "replace", "source": match.observed,
                "target": match.canonical, "reason": "entity_correction",
            }
            projection_reasons = (
                *_introduced_hint_reasons(prepared, projected_text),
                *_unconfirmed_entity_edit_reasons(prepared, projected_text, patch),
                *reject_reasons(prepared.baseline_text, projected_text,
                                numeric_rule_mode=numeric_rule_mode),
                *window_context_reject_reasons(
                    prepared.baseline_text, projected_text, **(window_context or {}),
                ),
            )
            if not projection_reasons:
                change = {
                    "observed": match.observed, "canonical": match.canonical,
                    "entity_type": match.entity_type, "entity_id": match.entity_id,
                    "match_type": match.match_type, "match_score": match.final_score,
                    "decision": "CONTEXT_CONFIRMED",
                    "confirmation_method": "bounded_prose_entity_projection_v1",
                    "discarded_response_reasons": reasons,
                }
                return FinalizedEntitySegment(
                    projected_text, True, (), (*trusted_changes, change)
                )
        return FinalizedEntitySegment(trusted_baseline, False, reasons, trusted_changes)
    confirmed_changes = tuple(
        {
            "observed": m.observed,
            "canonical": m.canonical,
            "entity_type": m.entity_type,
            "entity_id": m.entity_id,
            "match_type": m.match_type,
            "match_score": m.final_score,
            "decision": "CONTEXT_CONFIRMED",
        }
        for m in prepared.report.matches
        if m.decision is EntityDecision.HINT_ONLY
        and restored_text.count(m.canonical) > prepared.baseline_text.count(m.canonical)
        and restored_text.count(m.observed) < prepared.baseline_text.count(m.observed)
    )
    return FinalizedEntitySegment(
        restored_text, True, (), (*prepared.normalizations, *confirmed_changes)
    )


def _unconfirmed_entity_edit_reasons(prepared, output, patch):
    reasons = []
    baseline = prepared.baseline_text
    for match in prepared.report.matches:
        if match.observed == match.canonical:
            continue
        introduced = output.count(match.canonical) > baseline.count(match.canonical)
        removed = output.count(match.observed) < baseline.count(match.observed)
        if not introduced and not removed:
            continue
        if match.decision is EntityDecision.AUTO_NORMALIZE:
            continue  # harmless width/spacing normalization already verified
        if match.decision is EntityDecision.KEEP_RAW:
            if introduced:
                reasons.append("entity_boundary_or_shadow_edit:" + match.canonical)
            continue
        if (
            "preserve_policy" in match.reasons
            or "high_risk_entity_type" in match.reasons
        ):
            if introduced:
                reasons.append("entity_requires_verified_alias:" + match.canonical)
            continue
        confirmed = (
            isinstance(patch, dict)
            and patch.get("action") == "replace"
            and patch.get("source") == match.observed
            and patch.get("target") == match.canonical
            and patch.get("reason")
            in {"entity_correction", "local_typo", "term_normalization"}
            and baseline.count(match.observed) == 1
            and introduced
            and removed
        )
        if not confirmed:
            reasons.append("entity_confirmation_required:" + match.canonical)
    return tuple(dict.fromkeys(reasons))


def _introduced_hint_reasons(
    prepared: PreparedEntitySegment, restored_text: str
) -> tuple[str, ...]:
    """Reject glossary entities added without consuming a matching ASR span."""

    reasons: list[str] = []
    for canonical in prepared.hints:
        added = max(
            0,
            restored_text.count(canonical) - prepared.baseline_text.count(canonical),
        )
        if not added:
            continue
        allowed = 0
        for match in prepared.report.matches:
            if match.canonical != canonical:
                continue
            removed = max(
                0,
                prepared.baseline_text.count(match.observed)
                - restored_text.count(match.observed),
            )
            allowed += min(1, removed)
        if added > allowed:
            reasons.append(f"introduced_entity_from_hint:{canonical}")
    return tuple(reasons)
