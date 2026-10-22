"""User glossary spelling with complete syllables and independent boundaries.

The glossary owns the spelling. Audio confirmation is not required for exact
homophones. A narrow near-sound route allows one letter in one syllable for
names of at least three characters. No syllable insertion or reordering.
"""
from collections import defaultdict
from functools import lru_cache
import re

from .entity_boundaries import _tokens, _tokenizer, inferred_tokens, latin_boundaries_valid
from .entity_matcher import _pinyin_tokens
from .entity_mixed_spelling import mixed_key, supported_surface

CHINESE = re.compile(r'[\u3400-\u9fff]+')


@lru_cache(maxsize=4096)
def _tones(text):
    from pypinyin import Style, lazy_pinyin
    return tuple(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))


@lru_cache(maxsize=1)
def _function_words():
    """Independent lexicon: avoid turning grammar such as 关于 into 关羽."""
    import jieba.posseg
    tokenizer = _tokenizer()
    if tokenizer is None:
        return frozenset()
    tags = jieba.posseg.POSTokenizer(tokenizer).word_tag_tab
    return frozenset(word for word, tag in tags.items()
                     if tag and tag[0] in {'p', 'c', 'u', 'r', 'd', 'm', 'q'})


def _one_letter(a, b):
    if abs(len(a)-len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    return any(long[:i]+long[i+1:] == short for i in range(len(long)))


class DirectEntitySpeller:
    def __init__(self, definitions):
        self.exact = defaultdict(list)
        self.near = defaultdict(list)
        self.mixed = defaultdict(list)
        self.lengths = set()
        for definition in definitions:
            if not definition.enabled:
                continue
            for surface in dict.fromkeys((definition.canonical_text, *definition.aliases)):
                if supported_surface(surface) and re.search(r'[A-Za-z\u0370-\u03ff]', surface):
                    key, _ = mixed_key(surface)
                    self.mixed[key].append((definition, surface, ()))
                if not CHINESE.fullmatch(surface) or not 2 <= len(surface) <= 24:
                    continue
                sound = _pinyin_tokens(surface)
                self.lengths.add(len(sound))
                item = (definition, surface, sound)
                self.exact[sound].append(item)
                if len(sound) >= 3:
                    for i in range(len(sound)):
                        self.near[(len(sound), i, sound[:i]+sound[i+1:])].append(item)

    def _matches(self, observed, sound):
        exact = self.exact.get(sound, ())
        if exact:
            return [(item, 'exact_sound') for item in exact
                    if _tones(observed) == _tones(item[1])]
        if len(sound) < 3:
            return []
        matches = {}
        for i, syllable in enumerate(sound):
            for item in self.near.get((len(sound), i, sound[:i]+sound[i+1:]), ()):
                definition, surface, target = item
                if (_one_letter(syllable, target[i]) and set(observed) & set(surface)):
                    matches[(definition.entity_id, surface)] = (item, 'near_sound')
        return list(matches.values())

    @lru_cache(maxsize=16)
    def select(self, text, final=False):
        tokens = _tokens(text)
        if not tokens:
            return []
        tokens = (*tokens, *inferred_tokens(text))
        spans = []
        if self.mixed:
            key, positions = mixed_key(text)
            from .entity_code_spelling import code_spans
            spans.extend(code_spans(text, key, positions, self.mixed))
            for surface_key, items in self.mixed.items():
                offset = key.find(surface_key)
                while offset >= 0:
                    start = positions[offset][0]
                    end = positions[offset+len(surface_key)-1][1]
                    # Reject matching part of a normalized integer or code.
                    left = key[offset-1] if offset else ''
                    right = key[offset+len(surface_key):offset+len(surface_key)+1]
                    begins_code = surface_key[0].isascii() and surface_key[0].isalnum()
                    ends_code = surface_key[-1].isascii() and surface_key[-1].isalnum()
                    if not ((begins_code and left.isascii() and (left.isalnum() or left=='_')) or
                            (ends_code and right.isascii() and (right.isalnum() or right=='_'))):
                        spans.append((start, end, text[start:end], [(item, 'mixed_form') for item in items]))
                    offset = key.find(surface_key, offset+1)
        for run in CHINESE.finditer(text):
            sounds = _pinyin_tokens(run.group())
            for size in sorted(self.lengths, reverse=True):
                for offset in range(len(sounds)-size+1):
                    start, end = run.start()+offset, run.start()+offset+size
                    observed = text[start:end]
                    matches = self._matches(observed, sounds[offset:offset+size])
                    if not matches:
                        continue
                    # Full registered names protect their interiors, including
                    # preserve-only entries and spellings not being changed.
                    spans.append((start, end, observed, matches))
        proposals = []
        for start, end, observed, matches in spans:
            if observed in _function_words():
                continue
            if end == len(text) and not final:
                continue
            if not latin_boundaries_valid(text, start, end):
                continue
            if any(a < start < b or a < end < b for _, a, b in tokens):
                continue
            if any(a <= start and end <= b and b-a > end-start for a, b, _, _ in spans):
                continue
            owners = {item[0].canonical_text for item, _ in matches}
            if len(owners) != 1:
                continue
            definition, surface, _ = matches[0][0]
            if definition.normalization_policy != 'normalize' or observed == definition.canonical_text:
                continue
            rule = matches[0][1]
            proposals.append(dict(start=start, end=end, observed=observed,
                canonical=definition.canonical_text, entity_id=definition.entity_id,
                evidence_surface=surface, evidence_source='glossary_spelling',
                direct_rule=rule))
        # Full phrases take precedence; reject overlaps rather than allowing
        # a later short patch to overwrite part of a longer accepted phrase.
        chosen = []
        for patch in sorted(proposals, key=lambda p: (-(p['end']-p['start']), p['start'])):
            if not any(patch['start'] < p['end'] and p['start'] < patch['end'] for p in chosen):
                chosen.append(patch)
        return sorted(chosen, key=lambda p: p['start'])
