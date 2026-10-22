"""Shared protocol text for training and runtime Refiner requests.

The runtime prompt is intentionally kept in one place.  Changing this value
is a model-behaviour change, so callers should update the protocol version and
run the refinement regression set before changing it.
"""

import json
from difflib import SequenceMatcher
import unicodedata

WINDOW_EDIT_POLICIES = ("compat", "strict_local")
STRICT_LOCAL_PROTOCOL_VERSION = "v1-strict-local-window-edit"
STRICT_LOCAL_MAX_SPAN_CHARS = 32
STRICT_LOCAL_REASONS = frozenset({
    "disfluency", "repetition", "self_correction", "boundary_punctuation",
    "local_typo", "entity_correction",
})
_TERMINAL_BOUNDARY_CHARS = frozenset("，,、。！？!?；;：:…“”‘’\"'")

WINDOW_CONTEXT_PROTOCOL_VERSION = "v1-read-only-adjacent-context"


def window_context_messages(
    source: str, *, left_context: str = "", right_context: str = "",
) -> tuple[str, str]:
    """Separate source-owned text from adjacent, read-only ASR context."""

    instructions = (
        "用户消息是当前窗口（current_window），是唯一允许修改和输出的正文。"
        "系统提示中的read_only_left_context和read_only_right_context是只读的相邻ASR原文，"
        "仅帮助判断语义、重复和断句，不得修改、复制到输出或作为补丁source。"
        "相邻上下文中的标点可能是ASR误断句，不要把标点块当成完整句子。"
        "窗口末尾的词可能是下文的主语或句子开头，必须结合右侧上下文判断，"
        "不得仅因窗口末尾缺少谓语或与前文出现过相同词语就删除它。"
        "优先只返回一个严格JSON对象，且只能包含action、source、target、reason四个字符串字段。"
        '无需修改返回{"action":"keep","source":"","target":"","reason":"no_change"}。'
        "需要修改返回action为replace的局部补丁，source必须逐字且唯一地"
        "出现在current_window中，target不得包含相邻上下文。"
        "如返回转录文本，只能返回用户消息的精修结果，不能输出任何只读上下文。"
        "只读上下文缺失时，保留存疑的窗口边缘词，不得猜测或补写下文。"
    )
    context = json.dumps(
        {
            "read_only_left_context": left_context,
            "read_only_right_context": right_context,
        },
        ensure_ascii=False,
    )
    return instructions + "\n只读上下文JSON：" + context, source


def apply_window_context_patch(source: str, response: str, *, policy: str = "compat"):
    """Validate scoped patches; legacy prose is checked after entity restoration."""

    if policy == "compat" and not response.lstrip().startswith(("{", "```")):
        return apply_structured_patch(source, response, policy=policy)

    payload, issue = _strict_local_payload(response.strip())
    if issue:
        return source, payload, "window_context_invalid_patch"
    if payload["action"] == "keep" and (
        payload["source"] or payload["target"] or payload["reason"] != "no_change"
    ):
        return source, payload, "window_context_invalid_keep"
    return apply_structured_patch(source, response, policy=policy)


def parse_window_edit_policy(value: str) -> str:
    policy = str(value).strip().lower()
    if policy not in WINDOW_EDIT_POLICIES:
        raise ValueError(f"window_edit_policy must be one of {WINDOW_EDIT_POLICIES}")
    return policy


def strict_local_system_prompt(*, numeric_rule_mode: bool) -> str:
    """Protocol for window editing; candidate KEEP/REMOVE review is independent."""
    numeric = (
        "数字格式已由上游规则确定，数字、单位、范围和写法都必须原样保留。"
        if numeric_rule_mode else
        "保留原有数字、日期、单位和范围；本轮不做数字格式改写。"
    )
    return (
        "你是忠实的ASR局部纠错助手。优先保留正确内容，只修改有明确依据的异常口吃、"
        "非必要填充词、局部错字或误断句。信息存疑时保留原文。"
        "正常叠词、逐次表达、程度强调、对话应答不可按口吃删除，"
        "例如‘一级一级’‘一件一件’‘很多很多’‘特别特别’应保留。"
        "保留否定、人物指代、实体和条件关系；不得总结、润色、改写事实或补充内容。"
        f"{numeric}"
        "原样保留所有__ENTITY_NNN__标记。必须保留输入末尾的标点和引号，"
        "不得把窗口末尾的逗号改成句号。"
        "只返回一个严格JSON对象，不加Markdown、解释或<KEY>后缀。"
        "对象必须且只能包含action、source、target、reason四个字符串字段。"
        '无需修改时返回{"action":"keep","source":"","target":"","reason":"no_change"}。'
        '需修改时返回{"action":"replace","source":"唯一原文片段","target":"修复片段","reason":"原因"}。'
        f"source必须逐字且唯一地出现在输入中，source和target各不超过{STRICT_LOCAL_MAX_SPAN_CHARS}个字符。"
        "一次只允许一个连续修改位置，不得把多处修改包进一个长补丁。"
        "replace的reason只能是disfluency、repetition、self_correction、boundary_punctuation、"
        "local_typo或entity_correction。无法满足要求时返回keep。"
    )

RUNTIME_PROTOCOL_VERSION = "v4-model-numeric-range-guard"

REFINER_SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
    "处理自我修正。规范数字、日期、术语和代码符号，但必须保持数值和原意不变。"
    "数字范围必须作为一个整体规范化：凡由‘到’、‘至’、连字符或波浪线连接的数量或测量表达式，"
    "两端必须同时转换，禁止只转换一端，也禁止生成中文数字和阿拉伯数字混合的范围。"
    "不要总结、扩写或解释。成语中的汉字数字（如三番五次）必须保持原样。"
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
    "target 不得凭空增加没有依据的实词；reason 优先使用 no_change、disfluency、"
    "repetition、self_correction、boundary_punctuation、local_typo 之一；允许 source"
    "跨相邻标点块以处理自我修正，但只能修改一个局部片段。"
)

NUMERIC_RULES_REFINER_SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖和重复，修错字，补必要标点，处理明确的自我纠正。"
    "数字格式已经由上游规则库确定。所有保留的数字、单位、范围端点及其写法必须与输入一致；"
    "不得重新换写、扩展、缩写或改变数值。"
    "只有在相邻完整分句确属重复时，才可删除其中一份；只有在说话者明确改口时，才可删除被纠正的旧分句。"
    "删除时保留剩余分句中的数字原样，其他位置的数字必须保留。"
    "如果重复或改口关系不确定，保留原文。不要总结、扩写或解释。"
    "成语中的汉字数字（如三番五次）必须保持原样。"
    "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。"
    "输入中形如 __ENTITY_000__ 的受保护标记必须在输出中原样保留一次，"
    "不得删除、改写、重复或调整顺序。"
    "只返回一个 JSON 对象，不要解释或使用 Markdown。无需修改时返回"
    '{"action":"keep","source":"","target":"","reason":"no_change"}。'
    "需要修改时返回 {\"action\":\"replace\",\"source\":\"原文中的连续片段\","
    "\"target\":\"修改后的片段\",\"reason\":\"具体原因\"}。source 必须逐字来自输入，"
    "target 不得凭空增加没有依据的实词；reason 优先使用 no_change、disfluency、"
    "repetition、self_correction、boundary_punctuation、local_typo 之一；允许 source"
    "跨相邻标点块以处理自我纠正，但只能修改一个局部片段。"
)

NUMERIC_RULES_PROTOCOL_VERSION = "v1-rule-owned-number-surface"

NUMERIC_RULES_STRICT_PLACEHOLDER_PROMPT = (
    "严格保留所有 __ENTITY_NNN__ 标记：每个标记原样出现一次并留在对应语义位置。"
    "继续遵守数字格式锁定。仍可删除经判断为重复的完整分句或明确被改口取代的分句；"
    "删除时不得丢失被保留分句中的标记、数字或单位。"
)

REPETITION_REVIEW_SYSTEM_PROMPT = (
    f"{REFINER_SYSTEM_PROMPT}"
    "前文和后文只用于理解用户输入的转写。只精修用户消息中的句子。"
)

STRICT_PLACEHOLDER_PROMPT = (
    "最高优先级：完整保留输入中的每个句子和信息，不得总结、缩写、"
    "合并或删除内容。先逐字复制所有 __ENTITY_NNN__ 标记到对应位置，"
    "再仅修正其他文字的错字和标点。任何标记都不能省略或改变。"
)


def apply_structured_patch(
    source: str, response: str, *, policy: str = "compat"
) -> tuple[str, dict[str, object] | None, str | None]:
    """Apply a Refiner JSON patch without allowing free-form window rewrites.

    The legacy plain-text response remains supported for compatibility with
    older checkpoints.  A response that clearly attempts JSON but is malformed
    is rejected and falls back to the source segment; this prevents a partial
    JSON generation from being treated as a transcript.
    """

    policy = parse_window_edit_policy(policy)
    candidate = response.strip()
    if not candidate:
        return source, None, "structured_patch_empty"
    if policy == "strict_local":
        payload, issue = _strict_local_payload(candidate)
        if issue:
            return source, payload, issue
        if payload["action"] == "keep":
            if payload["source"] or payload["target"] or payload["reason"] != "no_change":
                return source, payload, "strict_local_invalid_keep"
            return source, payload, None
        if payload["action"] != "replace" or payload["reason"] not in STRICT_LOCAL_REASONS:
            return source, payload, "strict_local_invalid_action_or_reason"
        patch_source, patch_target = payload["source"], payload["target"]
        if max(len(patch_source), len(patch_target)) > STRICT_LOCAL_MAX_SPAN_CHARS:
            return source, payload, "strict_local_span_too_long"
        if patch_source == patch_target:
            return source, payload, "strict_local_noop_replace"
        # str.count misses overlapping occurrences, e.g. 哈哈 in 哈哈哈.
        starts = [i for i in range(len(source)) if patch_source and source.startswith(patch_source, i)]
        if len(starts) != 1:
            return source, payload, "structured_patch_source_not_unique"
        # A syntactically local patch can still replace a short utterance
        # wholesale (e.g. deleting all words and leaving only its final mark).
        # Preserve a lexical anchor outside the edited span so the operation
        # remains source-owned and the existing fallback keeps the full window.
        meaningful_source = "".join(
            char for char in source
            if not char.isspace() and unicodedata.category(char)[0] not in {"P", "Z"}
        )
        meaningful_patch = "".join(
            char for char in patch_source
            if not char.isspace() and unicodedata.category(char)[0] not in {"P", "Z"}
        )
        if meaningful_source and meaningful_patch == meaningful_source:
            return source, payload, "strict_local_edit_consumes_entire_window"
        edits = [tag for tag, *_ in SequenceMatcher(None, patch_source, patch_target, autojunk=False).get_opcodes()
                 if tag != "equal"]
        if len(edits) != 1:
            return source, payload, "strict_local_multiple_edit_regions"
        start = starts[0]
        refined = source[:start] + patch_target + source[start + len(patch_source):]
        if _terminal_boundary(source) != _terminal_boundary(refined):
            return source, payload, "strict_local_terminal_boundary_changed"
        return refined, payload, None
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


def _terminal_boundary(text: str) -> str:
    text = text.rstrip()
    start = len(text)
    while start and text[start - 1] in _TERMINAL_BOUNDARY_CHARS:
        start -= 1
    return text[start:]


def _strict_local_payload(candidate: str) -> tuple[dict | None, str | None]:
    if not candidate.startswith("{"):
        return None, "strict_local_json_required"

    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(candidate, object_pairs_hook=unique_fields)
    except (ValueError, TypeError):
        return None, "strict_local_invalid_json"
    fields = {"action", "source", "target", "reason"}
    if not isinstance(payload, dict) or set(payload) != fields or any(type(payload[key]) is not str for key in fields):
        return payload if isinstance(payload, dict) else None, "strict_local_invalid_fields"
    return payload, None


def permits_boundary_punctuation_repair(
    payload: dict[str, object] | None,
    *,
    source_text: str | None = None,
    protected_spans: tuple[str, ...] = (),
) -> bool:
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
    if not 1 <= deleted <= 2:
        return False
    if source_text is None:
        # Keep the protocol helper backwards-compatible for callers that only
        # perform a syntactic patch check.  Runtime acceptance passes the
        # source text and uses the full anomaly validator below.
        return True
    from .refinement_guard import validate_boundary_repair

    return validate_boundary_repair(
        source,
        target,
        reason,
        protected_spans=protected_spans,
    )


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
