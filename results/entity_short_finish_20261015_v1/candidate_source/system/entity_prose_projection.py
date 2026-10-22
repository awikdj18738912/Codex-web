"""Project a narrowly aligned prose entity correction onto the source text.

This is confirmation from an actual model edit, never automatic homophone
normalization. The full prose response is not adopted.
"""

from __future__ import annotations

import unicodedata

from .entity_matcher import EntityDecision


def _lexical(text: str) -> str:
    return "".join(
        char for char in text
        if not char.isspace() and unicodedata.category(char)[0] not in {"P", "Z"}
    )


def project_prose_entity(prepared, output: str):
    """Return one uniquely supported edit, or None when evidence is weak.

    Require a three-or-more-character registered candidate with precisely one
    changed character and identical pronunciation, eight outside lexical
    characters, and an otherwise identical response (at most one deletion).
    Preserve policies, risky types, competing candidates and short utterances
    retain the explicit-patch confirmation requirement.
    """
    baseline = prepared.trusted_baseline_text or prepared.baseline_text
    if baseline != prepared.baseline_text or any(c in output for c in "<>{}"):
        return None
    proposals = []
    for match in prepared.report.matches:
        observed, canonical = match.observed, match.canonical
        if (
            match.decision is not EntityDecision.HINT_ONLY
            or any(reason in match.reasons for reason in (
                "preserve_policy", "high_risk_entity_type", "candidate_margin_too_small"
            ))
            or (match.margin is not None and match.margin < 0.12)
            or not 3 <= len(observed) == len(canonical) <= 12
            or match.pinyin_score < 1.0 - 1e-9
            or sum(a != b for a, b in zip(observed, canonical)) != 1
            or baseline.count(observed) != 1
            or baseline.count(canonical) != 0
            or output.count(canonical) != 1
            or observed in output
        ):
            continue
        start = baseline.index(observed)
        if len(_lexical(baseline[:start] + baseline[start + len(observed):])) < 8:
            continue
        projected = baseline[:start] + canonical + baseline[start + len(observed):]
        expected, actual = _lexical(projected), _lexical(output)
        if expected != actual:
            # Only one omitted lexical character outside the entity is tolerated.
            # Enumerating the omission also avoids ambiguous sequence alignment.
            if len(expected) != len(actual) + 1:
                continue
            entity_start = len(_lexical(baseline[:start]))
            if not any(
                expected[:index] + expected[index + 1:] == actual
                for index in range(len(expected))
                if not entity_start <= index < entity_start + len(canonical)
            ):
                continue
        proposals.append((projected, match))
    return proposals[0] if len(proposals) == 1 else None
