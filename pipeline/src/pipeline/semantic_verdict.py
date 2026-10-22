"""Parse teacher verdicts without coercing malformed responses into approval."""
from __future__ import annotations

import json
from dataclasses import dataclass

PARSER_VERSION = "strict-json-bool-v2"


@dataclass(frozen=True)
class SemanticVerdict:
    passed: bool
    reason: str
    valid: bool


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def parse_semantic_verdict(response: object) -> SemanticVerdict:
    """Require an object with an actual JSON boolean and a string reason."""
    if not isinstance(response, str):
        return SemanticVerdict(False, "qc_response_not_text", False)
    try:
        result = json.loads(
            response.strip(), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError):
        return SemanticVerdict(False, "qc_response_invalid_json", False)
    if not isinstance(result, dict):
        return SemanticVerdict(False, "qc_response_not_object", False)
    if type(result.get("pass")) is not bool:
        return SemanticVerdict(False, "qc_response_missing_or_invalid_boolean", False)
    reason = result.get("reason")
    if not isinstance(reason, str):
        return SemanticVerdict(False, "qc_response_missing_or_invalid_reason", False)
    return SemanticVerdict(result["pass"], reason.strip(), True)
