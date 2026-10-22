"""Literal glossary keys for mixed Chinese/Latin spellings, with offsets.

Only spaces inside an alphanumeric code and a trailing spoken Chinese integer
are normalized. No missing letters, numeric value changes or punctuation
deletion are inferred. Original offsets survive Unicode width conversion.
"""
import re
import unicodedata

_CODE = re.compile(
    r'(?:[A-Za-z](?:[ \t]+[A-Za-z])+(?![A-Za-z])|[A-Za-z0-9]+)'
    r'(?:[ \t]*[0-9]+)?(?:[ \t]*[零〇一二两三四五六七八九十]{1,3})?')
_NUMBER = re.compile(r'[零〇一二两三四五六七八九十]+$')
_DIGITS = {c: str(i) for i, c in enumerate('零一二三四五六七八九')}
_DIGITS.update({'〇': '0', '两': '2'})


def _number(value):
    if '十' not in value:
        return ''.join(_DIGITS[c] for c in value)
    if value.count('十') != 1:
        return None
    left, right = value.split('十')
    if len(left) > 1 or len(right) > 1 or left in ('零', '〇'):
        return None
    return str((int(_DIGITS[left]) if left else 1)*10 + (int(_DIGITS[right]) if right else 0))


def mixed_key(text):
    """Return normalized text and original [start,end) offsets per character."""
    expanded, positions = [], []
    for i, char in enumerate(text):
        for folded in unicodedata.normalize('NFKC', char):
            expanded.append(folded)
            positions.append((i, i+1))
    source = ''.join(expanded)
    output, offsets, cursor = [], [], 0
    for match in _CODE.finditer(source):
        for i in range(cursor, match.start()):
            output.append(source[i].casefold()); offsets.append(positions[i])
        number = _NUMBER.search(match.group())
        number_start = match.start()+number.start() if number else match.end()
        for i in range(match.start(), number_start):
            if source[i] in ' \t':
                continue
            output.append(source[i].casefold()); offsets.append(positions[i])
        if number:
            replacement = _number(number.group())
            if replacement is None:
                for i in range(number_start, match.end()):
                    output.append(source[i]); offsets.append(positions[i])
            else:
                for char in replacement:
                    output.append(char)
                    offsets.append((positions[number_start][0], positions[match.end()-1][1]))
        cursor = match.end()
    for i in range(cursor, len(source)):
        output.append(source[i].casefold()); offsets.append(positions[i])
    return ''.join(output), offsets


def supported_surface(surface):
    return (2 <= len(surface) <= 64 and
            re.fullmatch(r'[A-Za-z0-9\u3400-\u9fff\u0370-\u03ff \-]+',
                         unicodedata.normalize('NFKC', surface)) is not None and
            re.search(r'[A-Za-z\u3400-\u9fff\u0370-\u03ff]',
                      unicodedata.normalize('NFKC', surface)) is not None)
