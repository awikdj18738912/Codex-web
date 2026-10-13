"""Source-based repetition preservation shared by every deletion commit path.

These conservative constraints are a safety policy, not a learned intent
classifier. Text cannot establish whether every repetition is a stutter.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import re
import threading
import unicodedata

_PAIR = re.compile(r'(?P<unit>[\u3400-\u9fff0-9.%％]{1,24})(?P<separator>[，,、。！？!?；;…\s]*)(?P=unit)')
_NEGATIONS = frozenset('不没无未别莫勿')
_DEGREE_PREFIXES = ('很', '非常', '特别', '十分', '极其', '真的', '太', '更加', '最为')
_INTENT_CUE = re.compile(r'逐字|原样|(?:反复|再三|一再)强调|'
                        r'(?:强调|说|提醒).{0,5}(?:[零〇一二两三四五六七八九十百千万0-9]+|多)(?:遍|次)|'
                        r'(?:重复|反复).{0,8}(?:提醒|说明|警告)|'
                        r'(?:连续|刻意|特意).{0,8}(?:遍|次)|'
                        r'重复.{0,8}(?:不能|不可|不得).{0,4}(?:省|删)|'
                        r'(?:保留|保持).{0,6}重复')
_HARD_ENDINGS = frozenset('。！？!?；;\n')
_REFERENCE_PREFIX = r'(?:前面|前文|刚才|之前)(?:的)?[“「"\s]*'
_REFERENCE_SUFFIX = (r'[”」"，,、：:\s]*(?:这句话[，,\s]*)?'
                     r'(?:说|强调|重复).{0,5}(?:两|二|2|多)(?:遍|次)')
_NAMED_REFERENCE = re.compile(_REFERENCE_PREFIX + r'[\u3400-\u9fff0-9.%％]{1,24}' + _REFERENCE_SUFFIX)
_CLAUSE_ENDINGS = frozenset('，,、。！？!?；;\n')
_POS_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _pos_tagger():
    # Isolate grammatical tags from entity matchers' custom dictionaries.
    import jieba
    import jieba.posseg
    return jieba.posseg.POSTokenizer(jieba.Tokenizer())


@lru_cache(maxsize=1024)
def _tags(text):
    try:
        with _POS_LOCK:return tuple((w.word,w.flag) for w in _pos_tagger().cut(text))
    except ImportError:
        return ()


def warm_source_repetition_analysis():
    """Load the existing CPU tagger once at application startup."""
    return bool(_tags('语句'))


def _cross_clause_noun_reuse(text,pair):
    """Conservative POS evidence for object/topic reuse, not a dependency parse.

    Require a punctuation boundary, a nominal unit, a preceding lexical verb,
    and a following predicate. Contiguous noun stutters and repeated verbs do
    not qualify. Use original ASR boundaries before a Refiner can erase them.
    """
    unit=pair.group('unit')
    if len(unit)<2 or not any(c in _CLAUSE_ENDINGS for c in pair.group('separator')):return False
    left=pair.start()
    while left>0 and text[left-1] not in _CLAUSE_ENDINGS:left-=1
    right=pair.end()
    while right<len(text) and text[right] not in _CLAUSE_ENDINGS:right+=1
    prefix=_surface(text[left:pair.start()])[-48:]
    suffix=_surface(text[pair.end():right])[:32]
    if not prefix or not suffix:return False
    nominal=_tags(unit)
    if not nominal or not all(flag.startswith('n') for _,flag in nominal):return False
    preceding=_tags(prefix)
    while preceding and preceding[-1][1] in {'ul','ug','uz'}:preceding=preceding[:-1]
    if not preceding or not(preceding[-1][1].startswith('v') and len(preceding[-1][0])>=2):return False
    following=_tags(suffix)
    for word,flag in following[:4]:
        if flag.startswith(('v','a')):return True
        if not flag.startswith(('d','c','v')):return False
    return False


def _surface(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text)
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


@dataclass(frozen=True)
class ProtectedRepetition:
    start: int
    end: int
    source: str
    unit: str
    reason: str


@lru_cache(maxsize=128)
def protected_repetitions(text: str) -> tuple[ProtectedRepetition, ...]:
    pairs = list(_PAIR.finditer(text))
    boundaries=[0]
    for character in text:boundaries.append(boundaries[-1]+(character in _HARD_ENDINGS))
    starts=[p.start() for p in pairs];ends=[p.end() for p in pairs]
    cue_spans = set()
    named_reference_clauses = []
    for ref in _NAMED_REFERENCE.finditer(text):
        left = ref.start()
        while left > 0 and text[left-1] not in _HARD_ENDINGS:
            left -= 1
        right = ref.end()
        while right < len(text) and text[right] not in _HARD_ENDINGS:
            right += 1
        named_reference_clauses.append((left,right))
    # A distant cue must explicitly name the earlier unit. Distance alone must
    # never associate an instruction with an unrelated stutter.
    for pair in pairs:
        reference = re.compile(_REFERENCE_PREFIX + re.escape(pair.group('unit')) + _REFERENCE_SUFFIX)
        references = list(reference.finditer(text, pair.end()))
        if references and not any(p.group('unit') == pair.group('unit')
                                  and pair.end() <= p.start() < references[0].start()
                                  for p in pairs):
            cue_spans.add(pair.span())
    for cue in _INTENT_CUE.finditer(text):
        if any(left <= cue.start() < right for left,right in named_reference_clauses):
            continue
        # Attach a trailing intent statement to its nearest prior repetition.
        # At most one intervening sentence, so another paragraph cannot protect
        # an unrelated stutter. Do not protect every candidate in a window.
        prior=bisect_right(ends,cue.start())-1
        if prior>=0 and boundaries[cue.start()]-boundaries[ends[prior]]<=1:
            cue_spans.add(pairs[prior].span())
        else:
            # Leading instructions apply only to the nearest following pair
            # in the same sentence, e.g. "原样记录：完成，完成".
            following=bisect_left(starts,cue.end())
            if following<len(pairs) and boundaries[starts[following]]==boundaries[cue.end()]:
                cue_spans.add(pairs[following].span())
    result=[]
    for pair in pairs:
        unit=pair.group('unit');reason=None
        if len(unit)>=2 and any(c in _NEGATIONS for c in unit):reason='repeated_negated_content'
        elif len(unit)>=2 and unit.startswith(_DEGREE_PREFIXES):reason='degree_reduplication'
        elif re.fullmatch(r'一[\u3400-\u9fff]{1,2}',unit):reason='distributive_reduplication'
        elif pair.span() in cue_spans:reason='explicit_repetition_intent'
        elif _cross_clause_noun_reuse(text,pair):reason='cross_clause_noun_reuse'
        if reason:
            # Keep the full multiplicity, not only the first two copies.
            end=pair.end()
            continuation=re.compile(r'[，,、。！？!?；;…\s]*'+re.escape(unit))
            while (next_copy:=continuation.match(text,end)) is not None:
                end=next_copy.end()
            result.append(ProtectedRepetition(pair.start(),end,text[pair.start():end],unit,reason))
    return tuple(result)


@lru_cache(maxsize=128)
def _lost_repetitions(source: str, target: str) -> tuple[ProtectedRepetition, ...]:
    """Require source-owned repeated surfaces and their multiplicity to survive.

    Punctuation changes are allowed. This is intentionally conservative across
    name/typo changes inside a protected span; safe unrelated edits still pass.
    """
    if source==target:return ()
    spans=protected_repetitions(source)
    if not spans:return ()
    before,offsets='', [0]
    fragments=[]
    for character in source:
        fragment=_surface(character);fragments.append(fragment);offsets.append(offsets[-1]+len(fragment))
    before=''.join(fragments);after=_surface(target)
    prefix=0
    while prefix<min(len(before),len(after)) and before[prefix]==after[prefix]:prefix+=1
    suffix=0
    while suffix<min(len(before),len(after))-prefix and before[len(before)-1-suffix]==after[len(after)-1-suffix]:suffix+=1
    source_end=len(before)-suffix;target_end=len(after)-suffix
    affected=[p for p in spans if offsets[p.end]>prefix and offsets[p.start]<source_end]
    if not affected:return ()
    blocks=SequenceMatcher(None,before[prefix:source_end],after[prefix:target_end],autojunk=False).get_matching_blocks()
    lost=[]
    for span in affected:
        start=max(prefix,offsets[span.start])-prefix
        end=min(source_end,offsets[span.end])-prefix
        # Require the specific source occurrence to remain aligned. Another
        # identical pair elsewhere in the transcript cannot replace it.
        if not any(b.a<=start and end<=b.a+b.size for b in blocks):lost.append(span)
    return tuple(lost)


def repetition_preservation_violations(source: str, target: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(span.reason for span in _lost_repetitions(source,target)))


def source_repetition_edit_violations(source: str, before: str, after: str) -> tuple[str, ...]:
    """Block newly lost source-owned spans; earlier losses cannot block all edits."""
    existing=set(_lost_repetitions(source,before))
    return tuple(dict.fromkeys(span.reason for span in _lost_repetitions(source,after) if span not in existing))


def preserve_repetition_cleanup(source: str, target: str) -> str:
    return source if repetition_preservation_violations(source,target) else target


def restore_preserved_repetitions(source: str, target: str) -> tuple[str, list[dict]]:
    """Restore only aligned deletions of whole copies; retain other edits.

    A complete unchanged copy must survive at the source-owned location. Any
    replacement inside the pair (including entity/ITN changes) is ambiguous and
    is left alone. No whole-transcript rollback or global string replacement.
    """
    def surface_offsets(text):
        surface, positions = [], []
        for i, character in enumerate(text):
            fragment = _surface(character)
            surface.append(fragment)
            positions.extend([i] * len(fragment))
        return ''.join(surface), positions

    before, src_positions = surface_offsets(source)
    after, dst_positions = surface_offsets(target)
    spans=protected_repetitions(source)
    if not spans:
        return target, []
    opcodes = SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
    proposals = []
    for span in spans:
        start = bisect_left(src_positions, span.start)
        end = bisect_left(src_positions, span.end)
        changes = [op for op in opcodes if op[0] != 'equal' and op[1] < end and op[2] > start]
        unit = _surface(span.unit)
        if not changes or any(tag != 'delete' or a < start or b > end or (a != start and b != end)
                              or not before[a:b] or before[a:b] != unit * (len(before[a:b]) // len(unit))
                              for tag, a, b, _, _ in changes):
            continue
        retained = ''.join(before[max(a,start):min(b,end)] for tag,a,b,_,_ in opcodes
                           if tag == 'equal' and a < end and b > start)
        if not retained or retained != unit * (len(retained) // len(unit)):
            continue
        for _, a, b, c, _ in changes:
            # Include the source separator before the next surviving copy.
            raw_start = src_positions[a] if a == start else src_positions[a-1] + 1
            raw_end = src_positions[b] if b < end else src_positions[b-1] + 1
            raw_end = min(raw_end, span.end)
            destination = dst_positions[c] if c < len(dst_positions) else (dst_positions[-1]+1 if dst_positions else 0)
            proposals.append(dict(start=destination, end=destination,
                                  target=source[raw_start:raw_end], source='',
                                  source_start=raw_start, source_end=raw_end,
                                  reason=span.reason))
    for item in sorted(proposals, key=lambda p:p['start'], reverse=True):
        target = target[:item['start']] + item['target'] + target[item['start']:]
    # A retained pair can still become an apparent stutter when its source
    # clause boundary is erased. Restore only a missing separator, preserving
    # existing output punctuation and all unrelated corrections.
    after,dst_positions=surface_offsets(target)
    blocks=SequenceMatcher(None,before,after,autojunk=False).get_matching_blocks()
    separators=[]
    for span in spans:
        if span.reason!='cross_clause_noun_reuse':continue
        start=bisect_left(src_positions,span.start);end=bisect_left(src_positions,span.end)
        block=next((b for b in blocks if b.a<=start and end<=b.a+b.size),None)
        if block is None:continue
        middle=block.b+start-block.a+len(_surface(span.unit))
        if middle>=len(dst_positions):continue
        left=dst_positions[middle-1]+1;right=dst_positions[middle]
        if target[left:right].strip():continue
        separator=span.source[len(span.unit):len(span.source)-len(span.unit)]
        separators.append(dict(start=left,end=right,source=target[left:right],target=separator,
            source_start=span.start+len(span.unit),source_end=span.end-len(span.unit),
            reason='source_clause_boundary_restoration'))
    for item in sorted(separators,key=lambda p:p['start'],reverse=True):
        target=target[:item['start']]+item['target']+target[item['end']:]
    proposals.extend(separators)
    return target, proposals
