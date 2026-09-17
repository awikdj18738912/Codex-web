"""Shared protocol text for training and runtime Refiner requests.

The runtime prompt is intentionally kept in one place.  Changing this value
is a model-behaviour change, so callers should update the protocol version and
run the refinement regression set before changing it.
"""

import json

RUNTIME_PROTOCOL_VERSION = "v2-rule-numeric"

REFINER_SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
    "处理自我修正。不要总结、扩写或解释。数字、日期、术语和代码符号已由系统规则"
    "处理，不得自行转换数字，成语中的汉字数字（如三番五次）必须保持原样。"
    "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。"
    "输入中形如 __ENTITY_000__ 的受保护标记必须在输出中原样保留一次，"
    "不得删除、改写、重复或调整顺序。"
)

STRUCTURED_PROTOCOL_VERSION = "v3-structured-patch"

STRUCTURED_REFINER_SYSTEM_PROMPT = (
    f"{REFINER_SYSTEM_PROMPT}"
    "只返回一个 JSON 对象，不要解释或使用 Markdown。无需修改时返回"
    '{"action":"keep","source":"","target":"","reason":"no_change"}。'
    "需要修改时返回 {\"action\":\"replace\",\"source\":\"原文中的连续片段\","
    "\"target\":\"修改后的片段\",\"reason\":\"具体原因\"}。source 必须逐字来自输入，"
    "target 不得凭空增加没有依据的实词；允许 source 跨相邻标点块以处理自我修正，"
    "但只能修改一个局部片段。"
)

STRICT_PLACEHOLDER_PROMPT = (
    "最高优先级：完整保留输入中的每个句子和信息，不得总结、缩写、"
    "合并或删除内容。先逐字复制所有 __ENTITY_NNN__ 标记到对应位置，"
    "再仅修正其他文字的错字和标点。任何标记都不能省略或改变。"
)


def apply_structured_patch(
    source: str, response: str
) -> tuple[str, dict[str, object] | None, str | None]:
    """Apply a Refiner JSON patch without allowing free-form window rewrites.

    The legacy plain-text response remains supported for compatibility with
    older checkpoints.  A response that clearly attempts JSON but is malformed
    is rejected and falls back to the source segment; this prevents a partial
    JSON generation from being treated as a transcript.
    """

    candidate = response.strip()
    if not candidate:
        return source, None, "structured_patch_empty"
    payload = _extract_json_object(candidate)
    if payload is None:
        if candidate.startswith("{") or candidate.startswith("```"):
            return source, None, "structured_patch_invalid_json"
        return candidate, None, None
    if not isinstance(payload, dict):
        return source, payload, "structured_patch_not_object"
    action = payload.get("action")
    if action == "keep":
        return source, payload, None
    if action != "replace":
        return source, payload, "structured_patch_invalid_action"
    patch_source = payload.get("source")
    patch_target = payload.get("target")
    if not isinstance(patch_source, str) or not isinstance(patch_target, str):
        return source, payload, "structured_patch_missing_fields"
    if not patch_source or source.count(patch_source) != 1:
        return source, payload, "structured_patch_source_not_unique"
    start = source.index(patch_source)
    return (
        source[:start] + patch_target + source[start + len(patch_source) :],
        payload,
        None,
    )


def permits_boundary_punctuation_repair(payload: dict[str, object] | None) -> bool:
    """Return whether a patch explicitly requests a bounded boundary repair.

    Source punctuation is protected by default.  Removing one or two marks is
    allowed only when the model labels the local patch as punctuation/boundary
    repair or self-correction; all other edits still require exact punctuation
    preservation and fall back to the source on failure.
    """

    if not isinstance(payload, dict) or payload.get("action") != "replace":
        return False
    source = payload.get("source")
    target = payload.get("target")
    reason = str(payload.get("reason", "")).strip().lower()
    if not isinstance(source, str) or not isinstance(target, str):
        return False
    if not any(
        marker in reason
        for marker in (
            "boundary",
            "punctuation",
            "断句",
            "标点",
            "自我修正",
            "self_correction",
        )
    ):
        return False
    punctuation_chars = set("，,、。！？!?；;：:.")
    deleted = sum(char in punctuation_chars for char in source) - sum(
        char in punctuation_chars for char in target
    )
    return 1 <= deleted <= 2


def _extract_json_object(value: str) -> dict[str, object] | None:
    """Decode a single JSON object, accepting one fenced object for tolerance."""

    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    if not text.startswith("{") or not text.endswith("}"):
        return None
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None
