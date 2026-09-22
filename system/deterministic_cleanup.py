"""Low-risk deterministic transcript cleanup independent of the Refiner."""

from __future__ import annotations

import re

from .transcript_repairs import apply_known_transcript_repairs


_UTTERANCE_RE = re.compile(r"[^。！？!?\uff1b;\n]+(?:[。！？!?\uff1b;]+|\n+|$)")
_TRAILING_BOUNDARY_RE = re.compile(r"[。！？!?\uff1b;\s]+$")
_TERMINAL_BOUNDARY_RE = re.compile(r"([。！？!?\uff1b;\n]+\s*)$")
_COMMA_RE = re.compile(r"([，,])")
_VISIBLE_RE = re.compile(r"[\w\u3400-\u9fff]", re.UNICODE)
_PRONOUN_STUTTER_RE = re.compile(r"([我你您他她它这那])\1+")
_STANDALONE_FILLER_RE = re.compile(
    r"(?P<left>^|[，,、；;])\s*"
    r"(?P<filler>呃+|额+)\s*"
    r"(?P<right>[，,、；;。！？!?]|$)"
)
_POST_SENTENCE_FILLER_RE = re.compile(
    r"(?P<boundary>[。！？!?；;\n])\s*嗯(?P<separator>[，,、：:])"
)
_BOUNDARY_ECHO_RE = re.compile(
    r"(?<=[\u3400-\u9fff])"
    r"(?<![，,、；;。！？!?\n\s])"
    r"(?P<echo>[\u3400-\u9fff])"
    r"(?P<separator>[，,])"
    r"(?:(?P=echo)(?P=separator))+"
)
_SENTENCE_BOUNDARY_ECHO_RE = re.compile(
    r"(?P<echo>[\u3400-\u9fff])"
    r"(?P<separator>[。！？!?；;])"
    r"(?P=echo)(?=[\u3400-\u9fff])"
)

def clean_transcript_deterministically(text: str) -> str:
    """Apply only exact, low-ambiguity cleanup rules to refined text."""

    return collapse_repeated_short_utterances(
        collapse_sentence_boundary_echoes(
            collapse_boundary_echo_fields(
                collapse_repeated_comma_items(
                    collapse_repeated_pronoun_stutters(
                        collapse_sentence_boundary_fillers(
                            collapse_standalone_fillers(
                                apply_known_transcript_repairs(text)
                            )
                        )
                    )
                )
            )
        )
    )


def collapse_boundary_echo_fields(text: str) -> str:
    """Remove repeated one-character comma fields at a clause boundary.

    This handles a source-owned boundary echo such as ``的话，话，你``.
    The first occurrence remains part of the preceding clause; only the
    subsequent standalone ``话，`` field is removed. The preceding field must
    contain more than that one character, which prevents ``慢，慢，`` at the
    start of a sentence from being treated as this boundary shape. No lexical
    list is used, and the operation is idempotent.
    """

    if not text:
        return text

    return _BOUNDARY_ECHO_RE.sub(
        lambda match: match.group("echo") + match.group("separator"),
        text,
    )


def detect_boundary_echo_repairs(text: str) -> tuple[dict[str, object], ...]:
    """Describe deterministic comma and sentence-boundary echo repairs."""

    repairs: list[dict[str, object]] = []
    for match in _BOUNDARY_ECHO_RE.finditer(text):
        repairs.append(
            {
                "source": match.group(0),
                "target": match.group("echo") + match.group("separator"),
                "echo": match.group("echo"),
                "start": match.start(),
                "end": match.end(),
            }
        )
    for match in _SENTENCE_BOUNDARY_ECHO_RE.finditer(text):
        repairs.append(
            {
                "source": match.group(0),
                "target": match.group("echo"),
                "echo": match.group("echo"),
                "start": match.start(),
                "end": match.end(),
                "boundary_type": "sentence",
            }
        )
    return tuple(repairs)


def collapse_sentence_boundary_echoes(text: str) -> str:
    """Merge an exact repeated character split by a sentence boundary."""

    if not text:
        return text
    return _SENTENCE_BOUNDARY_ECHO_RE.sub(
        lambda match: match.group("echo"),
        text,
    )


def collapse_standalone_fillers(text: str) -> str:
    """Remove unambiguous standalone ``呃``/``额`` fillers.

    Only punctuation-delimited fillers are touched.  This deliberately keeps
    attached words such as ``呃逆`` and meaningful phrases such as ``嗯哼``.
    When a filler sits between two commas, one comma is retained; before a
    sentence terminator the preceding comma is removed to avoid ``，。``.
    """

    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        left = match.group("left")
        right = match.group("right")
        if right in "，,、；;":
            return "" if left == "" else left
        if right in "。！？!?":
            return right
        return ""

    return _STANDALONE_FILLER_RE.sub(replace, text)


def collapse_sentence_boundary_fillers(text: str) -> str:
    """Remove an isolated ``嗯`` that opens a new clause after a sentence.

    A sentence-boundary position followed by a clause separator is a
    structural cue for a discourse filler, as in ``调查一下。嗯，那我``.
    Restricting this rule to that shape avoids changing lexical forms such as
    ``嗯哼``/``嗯嗯`` and leaves an initial or comma-delimited ``嗯`` for the
    Refiner, where it may be a meaningful acknowledgement.
    """

    if not text:
        return text
    return _POST_SENTENCE_FILLER_RE.sub(
        lambda match: match.group("boundary"),
        text,
    )


def collapse_repeated_pronoun_stutters(text: str) -> str:
    """Collapse adjacent repeated pronouns/demonstratives used as stutters.

    The character set is deliberately narrow. Normal lexical reduplication
    such as ``人人``/``天天`` and verb reduplication such as ``看看`` remain intact.
    """

    return _PRONOUN_STUTTER_RE.sub(lambda match: match.group(1), text)


def collapse_repeated_character_stutters(text: str) -> str:
    """Retain ambiguous AA forms until the Refiner proposes a local edit.

    Text alone cannot distinguish a stutter from a lexical reduplication or a
    name.  The previous global AA collapse damaged words even when the model
    had preserved them.  Keep this public helper for existing callers, but
    leave ambiguous text untouched.
    """

    return text


def collapse_repeated_comma_items(
    text: str,
    *,
    min_repetitions: int = 2,
    max_visible_chars: int = 8,
) -> str:
    """Collapse exact adjacent short items separated only by commas.

    Neural cleanup can be rejected when it removes a repeated filler together
    with unrelated content.  This post-merge rule handles the low-ambiguity
    remainder, for example ``不，不，不`` and ``少主，少主，少主！``.  Items
    must occupy complete comma-delimited fields in the same sentence; a word
    repeated inside a longer field therefore does not match.
    """

    if not text or min_repetitions < 2 or max_visible_chars < 1:
        return text

    return "".join(
        _collapse_comma_items_in_utterance(
            match.group(0),
            min_repetitions=min_repetitions,
            max_visible_chars=max_visible_chars,
        )
        for match in _UTTERANCE_RE.finditer(text)
    )


def _collapse_comma_items_in_utterance(
    utterance: str,
    *,
    min_repetitions: int,
    max_visible_chars: int,
) -> str:
    boundary_match = _TERMINAL_BOUNDARY_RE.search(utterance)
    if boundary_match is None:
        body, boundary = utterance, ""
    else:
        body = utterance[: boundary_match.start()]
        boundary = boundary_match.group(1)

    parts = _COMMA_RE.split(body)
    items = parts[0::2]
    separators = parts[1::2]
    if len(items) < min_repetitions:
        return utterance

    output: list[str] = []
    index = 0
    while index < len(items):
        key = _comma_item_key(items[index])
        end = index + 1
        while end < len(items) and _comma_item_key(items[end]) == key:
            end += 1
        visible_chars = len(_VISIBLE_RE.findall(key))
        collapse = (
            bool(key)
            and visible_chars <= max_visible_chars
            and end - index >= min_repetitions
        )
        if collapse:
            output.append(items[index])
            if end < len(items):
                output.append(separators[end - 1])
        else:
            for item_index in range(index, end):
                output.append(items[item_index])
                if item_index < len(separators):
                    output.append(separators[item_index])
        index = end
    return "".join(output) + boundary


def _comma_item_key(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def collapse_repeated_short_utterances(
    text: str,
    *,
    min_repetitions: int = 2,
    max_visible_chars: int = 8,
) -> str:
    """Collapse adjacent identical short utterances repeated two times or more.

    Sentence punctuation is a chunk boundary in the streaming pipeline, so a
    run such as ``可恶！可恶！可恶！`` cannot reliably be handled by a
    per-chunk neural cleanup pass. This exact-match rule runs after chunks are
    joined. Every exact adjacent repetition from the second occurrence onward
    is removed, including deliberate spoken emphasis.
    """

    if not text or min_repetitions < 2 or max_visible_chars < 1:
        return text

    utterances = [match.group(0) for match in _UTTERANCE_RE.finditer(text)]
    if not utterances:
        return text

    output: list[str] = []
    index = 0
    while index < len(utterances):
        key = _utterance_key(utterances[index])
        end = index + 1
        while end < len(utterances) and _utterance_key(utterances[end]) == key:
            end += 1
        visible_chars = len(_VISIBLE_RE.findall(key))
        if key and visible_chars <= max_visible_chars and end - index >= min_repetitions:
            output.append(utterances[index])
        else:
            output.extend(utterances[index:end])
        index = end
    return "".join(output)


def _utterance_key(value: str) -> str:
    body = _TRAILING_BOUNDARY_RE.sub("", value).strip()
    return re.sub(r"\s+", "", body)
