"""Conservative preservation of Latin names and model identifiers in Chinese."""

from __future__ import annotations

import re

_TOKEN = r"[-/]*[A-Za-z0-9_]+(?:[.+#/-]+[A-Za-z0-9_]+)*[+#]*"
_ENTITY = re.compile(rf"(?<![A-Za-z0-9_]){_TOKEN}(?:[ \t]+{_TOKEN})*")
_CLAUSE = re.compile(r"[^，,。；;！？!?\n]+")
_CORRECTION = re.compile(r"不对|哦不|噢不|改成|换成|改用|换用|更正为|准确地说|应该用|还是用")
_SPELLING_CUE = re.compile(r"(?:拼音|英文拼写|拼写)\s*(?:是|为|写作|叫做)?\s*$")
_FILLER = re.compile(r"^(?:嗯+|呃+|啊+|那个|就是|呃嗯|嗯呃)[，,、 ]*$")


def english_entity_spans(text: str) -> list[tuple[int, int, str]]:
    """Keep exact case, identifier punctuation and multiword-name spacing."""
    return [
        (match.start(), match.end(), match.group())
        for match in _ENTITY.finditer(text)
        if re.search(r"[A-Za-z]", match.group())
    ]


def _allowed_drops(source: str, target: str) -> list[tuple[int, int]]:
    clauses = list(_CLAUSE.finditer(source))
    wanted = {value for _, _, value in english_entity_spans(target)}
    spans = []
    for marker in _CORRECTION.finditer(source):
        index = next((i for i, c in enumerate(clauses)
                      if c.start() <= marker.start() < c.end()), None)
        if index is None:
            continue
        clause = clauses[index]
        prefix = source[clause.start():marker.start()]
        old_span = (clause.start(), marker.start())
        if not english_entity_spans(prefix):
            if index == 0:
                continue
            old_span = clauses[index - 1].span()
        tail = source[marker.end():clause.end()].strip()
        if not tail and index + 1 < len(clauses):
            tail = clauses[index + 1].group().strip()
        final_entities = [value for _, _, value in english_entity_spans(tail)]
        if final_entities and all(value in wanted for value in final_entities):
            spans.append(old_span)

    # A Latin pronunciation transcription can be removed only when its own
    # clause explicitly labels it as pinyin/spelling. This does not cover an
    # English name introduced as a separate fact (for example, "英文名是 Lin").
    for clause in clauses:
        value = clause.group()
        for start, end, _ in english_entity_spans(value):
            if _SPELLING_CUE.search(value[:start].strip()):
                spans.append((clause.start() + start, clause.start() + end))

    # Allow restart cleanup where a filler follows a short abandoned phrase,
    # and the next phrase carries that same entity into the target. The entity
    # must occur again in the target; this is not general duplicate removal.
    compact_target = re.sub(r"\s+", "", target)
    for i, clause in enumerate(clauses[:-2]):
        next_clause = clauses[i + 1]
        final_clause = clauses[i + 2]
        first_entities = english_entity_spans(clause.group())
        filler = next_clause.group().strip()
        if not first_entities or not _FILLER.fullmatch(filler):
            continue
        following_entities = [v for _, _, v in english_entity_spans(final_clause.group())]
        if (all(v in following_entities for _, _, v in first_entities)
                and re.sub(r"\s+", "", final_clause.group()) in compact_target):
            spans.extend((clause.start() + s, clause.start() + e)
                         for s, e, _ in first_entities)

    # Only a repeated complete clause with an unchanged surviving occurrence
    # gets this exception; isolated English-word emphasis is not deduplicated.
    compact_target = re.sub(r"\s+", "", target)
    for left, right in zip(clauses, clauses[1:]):
        compact = re.sub(r"\s+", "", left.group())
        separator = source[left.end():right.start()]
        if (re.fullmatch(r"[。！？!?]+", separator)
                and compact == re.sub(r"\s+", "", right.group())
                and compact in compact_target and english_entity_spans(left.group())):
            spans.extend((left.span(), right.span()))
    return spans


def english_entities_preserved(source: str, target: str) -> bool:
    """Reject translation, invention, spelling/case drift and unexplained loss.

    Explicit corrections and exact repeated clauses can discard their old
    entities. All retained names must remain in source order and verbatim.
    This lexical check complements, rather than replaces, semantic QC.
    """
    source_entities = english_entity_spans(source)
    desired = [value for _, _, value in english_entity_spans(target)]
    drops = _allowed_drops(source, target)
    states = {0}
    for start, end, value in source_entities:
        droppable = any(left <= start and end <= right for left, right in drops)
        next_states = set(states) if droppable else set()
        next_states.update(
            index + 1 for index in states
            if index < len(desired) and value == desired[index]
        )
        states = next_states
        if not states:
            return False
    return len(desired) in states
