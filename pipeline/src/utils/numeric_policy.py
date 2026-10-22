"""Keep numeric formatting owned by the same native rules used at runtime."""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from system.refinement_guard import (
    _numeric_rule_allowed_drop_spans,
    reject_reasons,
    restore_rule_numeric_surfaces,
)
from system.zh_itn_bridge import DEFAULT_LIBRARY, ZhITNNormalizer
from .english_entity_policy import english_entities_preserved

NUMERIC_POLICY = "rule-owned-numeric-v1"
_NUMERIC_CANDIDATE = re.compile(r"[\d零〇一二两三四五六七八九十百千万亿]")
_NUMERIC_LITERAL = re.compile(
    r"[+-]?\d+(?:[.:/]\d+)*|"
    r"[零〇一二两三四五六七八九十百千万亿兆]+(?:点[零〇一二两三四五六七八九]+)?"
)
_NUMERIC_REJECTIONS = frozenset({
    "numeric_value_mismatch", "numeric_rule_surface_mismatch",
    "partial_numeric_range_conversion",
})
_CORRECTION_MARKER = re.compile(
    r"不对|不是|应该是|改成|改为|回退|重新说|准确地说|最后定|最后还是|还是"
)
_EXPLICIT_REVERSAL = re.compile(r"不对|不是|应该是|改成|改为|回退|重新说|准确地说|还是")
_WEEKDAY = re.compile(r"周[一二三四五六日天]")


def _training_only_numeric_spans(text: str) -> list[re.Match[str]]:
    """Match numeric spellings while excluding the lexical filler ``一下``."""
    return [
        match for match in _NUMERIC_LITERAL.finditer(text)
        if not (match.group() == "一" and text[match.end():].startswith("下"))
    ]


def _weekday_correction_drops(source: str, target: str) -> tuple[tuple[int, int], ...]:
    """Allow weekday numerals in an explicitly abandoned clause to be dropped.

    Runtime numeric-value extraction does not model weekdays. Keep this narrow:
    the marker must introduce retained text verbatim, and only weekday tokens
    before that marker are eligible.
    """
    weekdays = tuple(_WEEKDAY.finditer(source))
    if len(weekdays) < 2:
        return ()
    # The final weekday's clause must survive verbatim. The marker itself may
    # be discarded as filler, e.g. ``最后改成，周三开会`` -> ``周三开会``.
    final = weekdays[-1]
    final_end = next(
        (boundary.start() for boundary in re.finditer(r"[，,。！？!?；;\n]", source[final.start():])),
        len(source) - final.start(),
    ) + final.start()
    final_clause = source[final.start():final_end].strip(" ，,、。！？!?；;:")
    if not final_clause or final_clause not in target:
        return ()
    for marker in _CORRECTION_MARKER.finditer(source):
        if marker.start() >= final.start():
            continue
        # Every adjacent weekday candidate in the chain needs an explicit
        # correction operator between them; a separate schedule is not a chain.
        chain = True
        for left, right in zip(weekdays, weekdays[1:]):
            if not _CORRECTION_MARKER.search(source[left.end():right.start()]):
                chain = False
                break
        if chain and _EXPLICIT_REVERSAL.search(source[weekdays[0].end():final.start()]):
            return tuple((match.start(), match.end()) for match in weekdays[:-1])
    return ()

TRAINING_SYSTEM_PROMPT = (
    "你是中文 ASR 文本精修助手。保留最终原意，最小修改：去无意义口癖、口吃和重复，"
    "修复明确的错字和自我修正，结合句意删除或替换造成意外断句的误插标点，保留正常句界与语气。"
    "保留有意义重复、否定和条件，不总结、扩写或解释。"
    "数字格式由外部规则库负责；英文名称和型号沿用原文。"
    "删除明确放弃的旧候选和仅识别用的拼字说明，保留最终有效内容。"
    "原样保留 __ENTITY_NNN__ 标记。只输出精修后的正文，不输出解释或 Markdown。"
)


@lru_cache(maxsize=1)
def _normalizer() -> ZhITNNormalizer:
    library = Path(os.environ.get("REFINER_TRAIN_ZH_ITN_LIBRARY", str(DEFAULT_LIBRARY)))
    return ZhITNNormalizer(library)


def normalize_source(text: str) -> tuple[str, dict]:
    """Normalize only through native rules; never ask a teacher to fill rule gaps."""
    metadata = {"numeric_policy": NUMERIC_POLICY, "itn_owner": "rules"}
    if not _NUMERIC_CANDIDATE.search(text):
        return text, metadata
    result = _normalizer().normalize(text)
    metadata.update({
        "numeric_rule_pack": result.rule_pack,
        "numeric_rule_changes": [change.public_dict() for change in result.changes],
    })
    return result.text, metadata


def lock_numeric_target(source: str, target: str) -> str | None:
    """Restore equivalent spellings; reject invented values or unsupported drops.

    The existing runtime guard permits explicit self-corrections and repeated
    clauses. Other content quality remains the semantic judge's responsibility.
    """
    restored = restore_rule_numeric_surfaces(source, target)
    if restored is None:
        return None
    if _NUMERIC_REJECTIONS.intersection(
        reject_reasons(source, restored, numeric_rule_mode=True)
    ):
        return None
    # The runtime guard intentionally ignores some ambiguous standalone Chinese
    # numerals. For training, they must still never become conversion labels.
    # Match literal spellings, permitting omissions only in the runtime's
    # explicitly validated correction/repeated-clause spans.
    drops = (*_numeric_rule_allowed_drop_spans(source, restored),
             *_weekday_correction_drops(source, restored))
    desired = [match.group() for match in _training_only_numeric_spans(restored)]
    states = {0}
    for match in _training_only_numeric_spans(source):
        droppable = any(start <= match.start() and match.end() <= end for start, end in drops)
        next_states = set(states) if droppable else set()
        next_states.update(
            index + 1 for index in states
            if index < len(desired) and match.group() == desired[index]
        )
        states = next_states
        if not states:
            return None
    if len(desired) not in states:
        return None
    return restored


def _construction_plan_matches(source: str, target: str, plan: object) -> bool:
    """Check an oral pair against its deterministic construction trace."""
    if not isinstance(plan, dict):
        return False
    if plan.get("construction_version") != "synthetic-oral-v1":
        return False
    raw_intent = plan.get("raw_intent")
    phenomena = plan.get("construction_phenomena")
    values = plan.get("correction_values")
    parts = plan.get("spelling_parts")
    expected = plan.get("expected_clean")
    if (not isinstance(raw_intent, str) or not raw_intent.strip() or not isinstance(expected, str)
            or not isinstance(phenomena, list)
            or any(not isinstance(item, str) for item in phenomena)
            or not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or not isinstance(parts, list)
            or any(not isinstance(part, dict)
                   or not isinstance(part.get("word"), str)
                   or not isinstance(part.get("character"), str)
                   for part in parts)):
        return False
    if plan.get("slot_policy") == "single-slot-v1":
        from .synthetic_oral import validate_correction_slots
        try:
            validate_correction_slots(raw_intent, values)
        except (TypeError, ValueError):
            return False

    # Import lazily so legacy callers do not depend on the synthetic generator.
    from .synthetic_oral import build_oral

    try:
        rebuilt = build_oral(raw_intent, values, parts, phenomena)
    except (ValueError, TypeError, IndexError, KeyError):
        return False
    rebuilt_normalized, _ = normalize_source(rebuilt)
    source_normalized, _ = normalize_source(source)
    intent_normalized, _ = normalize_source(raw_intent)
    return (rebuilt_normalized == source_normalized
            and intent_normalized == expected == target)


def natural_pair_matches(source: str, target: str, plan: object) -> bool:
    """Check exact provenance and independent teacher agreement, not a construction proof.

    Semantic validity is checked separately by the final pair audit. No correction
    keyword list determines which spoken values are allowed to disappear.
    """
    if not isinstance(plan, dict) or plan.get("generation_mode") != "teacher-natural-v1":
        return False
    raw_oral, raw_intent = plan.get("raw_oral"), plan.get("raw_intent")
    if not isinstance(raw_oral, str) or not isinstance(raw_intent, str):
        return False
    oral, _ = normalize_source(raw_oral)
    intent, _ = normalize_source(raw_intent)
    normalized, _ = normalize_source(source)
    verification = plan.get("label_verification")
    verified = verification == "independent-teacher-agreement-v1"
    if verification == "exact-passthrough-v1":
        verified = (oral == intent and plan.get("requested_phenomena") == ["clean"]
                    and plan.get("correction_values") == [] and plan.get("spelling_parts") == [])
    return (verified and normalized == oral == plan.get("independent_source")
            and target == intent == plan.get("expected_clean") == plan.get("independent_clean"))


def _lock_training_target(source: str, target: str, generation_plan: object | None) -> str | None:
    if isinstance(generation_plan, dict) and generation_plan.get("generation_mode") == "upstream-teacher-v1":
        # Upstream Oral generation has no mechanically constructed target.
        # The teacher-produced Clean label is checked later by semantic QC;
        # deterministic numeric/entity locks here incorrectly reject valid
        # abandoned values and ASR corrections before that judgment.
        expected = generation_plan.get("expected_clean")
        return target if isinstance(expected, str) and target == expected else None
    if isinstance(generation_plan, dict) and generation_plan.get("generation_mode") == "teacher-natural-v1":
        return target if natural_pair_matches(source, target, generation_plan) else None
    if generation_plan is not None and isinstance(generation_plan, dict) and "construction_version" in generation_plan:
        if _construction_plan_matches(source, target, generation_plan):
            return target
        return None
    locked = lock_numeric_target(source, target)
    if locked is None or not english_entities_preserved(source, locked):
        return None
    return locked


def lock_training_target(source: str, target: str, generation_plan: dict | None = None) -> str | None:
    """Validate constraints, admitting only exact source/target construction proofs."""
    return _lock_training_target(source, target, generation_plan)


def prepare_training_pair(source: str, target: str, generation_plan: dict | None = None) -> tuple[str, str, dict] | None:
    normalized, metadata = normalize_source(source)
    locked = lock_training_target(normalized, target, generation_plan)
    if locked is None:
        return None
    metadata["english_entity_policy"] = "verbatim-v1"
    return normalized, locked, metadata


def prepare_fragment_training_pair(
    source: str,
    target: str,
    full_oral: str,
    full_clean: str,
    generation_plan: dict | None = None,
    *,
    observed_prefix: str | None = None,
) -> tuple[str, str, dict] | None:
    """Validate a truncated pair, using construction provenance only for exact no-noise slices.

    ASR edits still go through the ordinary numeric and English entity guards.
    """
    from ..generators.asr_simulator import _align_observed_prefix, valid_partial_pair

    if not all(isinstance(value, str) and value for value in (source, target, full_oral, full_clean)):
        return None
    if not valid_partial_pair(full_oral, full_clean, source, target):
        return None

    has_plan = generation_plan is not None
    plan_matches = _construction_plan_matches(full_oral, full_clean, generation_plan)
    if has_plan and not plan_matches:
        return None

    if plan_matches:
        aligned = _align_observed_prefix(full_oral, target, generation_plan, full_clean)
        if generation_plan.get("slot_policy") == "single-slot-v1":
            if not isinstance(observed_prefix, str) or observed_prefix != aligned:
                return None
        if aligned is not None:
            aligned_normalized, _ = normalize_source(aligned)
            source_normalized, metadata = normalize_source(source)
            if (source_normalized == aligned_normalized
                    or _same_terminal_chinese_separator(source_normalized, aligned_normalized)):
                metadata.update({
                    "english_entity_policy": "verbatim-v1",
                    "fragment_training_proof": {
                        "version": "construction-prefix-v1",
                        "construction_version": generation_plan.get("construction_version"),
                        "slot_policy": generation_plan.get("slot_policy"),
                        "source_alignment": "exact_normalized_observed_prefix",
                    },
                })
                return source_normalized, target, metadata

    # Any recognition substitution/omission outside exact source alignment
    # retains the regular numeric and entity protections.
    return prepare_training_pair(source, target, None)


def _same_terminal_chinese_separator(left: str, right: str) -> bool:
    separators = "，。；、："
    return bool(left and right and (
        (left[:-1] == right and left[-1] in separators)
        or (right[:-1] == left and right[-1] in separators)
        or (left[:-1] == right[:-1] and left[-1] in separators and right[-1] in separators)
    ))
