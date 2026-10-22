"""Insert known speech artifacts into a teacher-authored final intent.

No new facts are authored here. The generated source has an explicit clean
reference, while the separate Clean generator must still recover it.
"""
from __future__ import annotations

import re


_FORBIDDEN_SLOT_PUNCTUATION = re.compile(r"[，。！？；?!;\r\n]")
_NUMBER_GROUP = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{3})+(?![\d,])")
_MEANINGFUL_CONTEXT = re.compile(r"[\w\u3400-\u9fff]", re.UNICODE)
_CLAUSE_LEAD = re.compile(r"^(?:请|我(?:要|想|需要)|帮我|在.{1,12}(?:中|里))")
_CLAUSE_USE_LEAD = re.compile(r"^(用|使用)")
_CLAUSE_PREDICATES = re.compile(r"替代|替换|实现|部署|清理|管理|启用|关闭|更新|修复|处理|检查|运行|选择")


def validate_correction_slots(
    intent: str, values: list[str], max_length: int = 40
) -> None:
    """Validate new correction candidates as short, unambiguous slot values.

    This is for generated-plan admission and old-plan audits. ``build_oral``
    deliberately remains permissive so historical construction-v1 rows can
    still be reconstructed exactly.
    """
    if not isinstance(intent, str) or not intent.strip() or not isinstance(values, list):
        raise ValueError("Intent and correction values must be strings and a list")
    if not values:
        return None
    for value in values:
        if (not isinstance(value, str) or not value or value != value.strip()
                or len(value) > max_length or _FORBIDDEN_SLOT_PUNCTUATION.search(value)):
            raise ValueError("Correction candidate is not a short single slot")
        # ASCII commas are accepted only as conventional digit grouping, such
        # as 348,700 or 1,234,567. Periods remain available for code/version IDs.
        if "," in _NUMBER_GROUP.sub("", value):
            raise ValueError("ASCII commas are allowed only in grouped numbers")
        # A slot may be an entity/attribute or one short action. A request-like
        # lead or several predicates signals a whole clause, even without commas.
        use_lead = _CLAUSE_USE_LEAD.match(value)
        use_clause = bool(use_lead and _CLAUSE_PREDICATES.search(value[use_lead.end():]))
        if (_CLAUSE_LEAD.search(value) or use_clause
                or len(_CLAUSE_PREDICATES.findall(value)) >= 2):
            raise ValueError("Correction candidate contains a whole clause, not one slot")

    final = values[-1]
    body = intent.strip()
    while body and body[-1] in "。！？!?；;":
        body = body[:-1].rstrip()
    # A single ASCII full stop may be sentence-final, but never erase a dot
    # that belongs to a code, URL, version, or a final slot ending in a dot.
    if body.endswith(".") and not final.endswith("."):
        body = body[:-1]
    if body.count(final) != 1:
        raise ValueError("Final correction slot must occur exactly once in the intent")
    start = body.find(final)
    if start < 0 or body == final:
        raise ValueError("Final correction slot must be a proper substring of the intent")
    context = body[:start] + body[start + len(final):]
    if not _MEANINGFUL_CONTEXT.search(context):
        raise ValueError("Final correction slot must retain meaningful intent context")


def build_oral(intent: str, values: list[str], parts: list[dict], phenomena: list[str]) -> str:
    if phenomena == ["clean"]:
        return intent
    terminal = intent[-1] if intent[-1] in "。！？!?." else "。"
    body = intent[:-1] if intent[-1] in "。！？!?." else intent
    if values:
        final = values[-1]
        position = body.find(final)
        if position < 0:
            raise ValueError("Final correction value is missing from the clean intent")
        chain = values[0]
        for index, value in enumerate(values[1:], start=1):
            marker = "，不对，" if index == 1 else "，最后改成，"
            chain += marker + value
        body = body[:position] + chain + body[position + len(final):]
    if parts:
        body += "，" + "，".join(f"{part['word']}的{part['character']}" for part in parts)
    if "boundary_punctuation" in phenomena:
        match = re.search(r"我想|请把|让我|希望|需要|使用|确认|查询|安排|明天|因为|如果|不能|不要", body)
        if match is None:
            match = re.search(r"[\u4e00-\u9fff]{2}", body)
        if match is None:
            raise ValueError("No Chinese phrase is available for a boundary error")
        index = match.start() + 1
        body = body[:index] + "。" + body[index:]
    level = next((x.split(":", 1)[1] for x in phenomena if x.startswith("oral:")), "low")
    if level == "high" and re.match(r"[\u4e00-\u9fff]", body):
        body = body[0] + "，" + body
    prefix = {"low": "嗯，", "moderate": "呃，那个，", "high": "嗯，那个，就是说，"}.get(level, "嗯，")
    result = prefix + body + terminal
    if len(result) > 200:
        raise ValueError("Constructed Oral exceeds 200 characters")
    return result
