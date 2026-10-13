"""Opt-in spelling of registered Chinese names with contextual evidence.

This verifies a glossary spelling, not an arbitrary semantic correction.
An additional opt-in permits bounded internal omissions in known long terms.
Ambiguous registered spellings and unmatched contexts abstain.
"""
from functools import lru_cache
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .entity_store import EntityDefinition

_CHINESE_NAME = re.compile(r'[\u3400-\u9fff]{3,16}')
_SENTENCE_END = re.compile(r'[。！？!?\n\r]')
_NUMERIC_DATE_CHARACTERS = frozenset('零〇一二三四五六七八九十百千万萬亿億两兩年月日号號时時分秒点點')
_FACT_MODIFIERS = _NUMERIC_DATE_CHARACTERS | frozenset('不无非有左右上下前后内外大小高低急慢良恶阴阳单双多旁性鼻皮口肛')
_ANATOMICAL_CHARACTERS = frozenset('肝胆肺胃肠心肾脾脑骨筋髓腺胰眼耳颈胸腹盆股膝髋颅椎脊血尿脂纤指肢手足臂腿趾牙舌肩掌颌喉膀胱卵精髂腕膈瓣动静脉黏膜皮')
_NEAR_SENSITIVE = _FACT_MODIFIERS | _ANATOMICAL_CHARACTERS | frozenset('自异同癌瘤炎菌病毒碳氮氢氧磷硫铁镁钾钠钙氯氟锌铜锰钒有是在了也因或与但正')


def supports_disease_definition(definition):
    return (definition.enabled and definition.normalization_policy=='normalize'
        and definition.entity_type=='TERM' and definition.domain=='medical'
        and re.fullmatch(r'[\u3400-\u9fff]{4,16}',definition.canonical_text) is not None
        and definition.canonical_text.endswith(('病','症','炎','癌','疝','障','感染','结石')))


def disease_definition_spelling_evidence(text,start,end,definition,namespace):
    """Full-sound registered disease name in an explicit definition clause.

    Requires an existing standard name, never invents a diagnosis or corrects
    another anatomical/numeric source character. A definition supplies the
    role evidence even when no spelling-context profile has been declared.
    """
    source=text[start:end];canonical=definition.canonical_text
    if (not supports_disease_definition(definition) or not _CHINESE_NAME.fullmatch(source)
        or len(source)!=len(canonical) or source==canonical):return ()
    signature=pronunciation_signature(canonical)
    if pronunciation_signature(source)!=signature or namespace.get(signature)!=frozenset({canonical}):return ()
    if any(a!=b and a in (_ANATOMICAL_CHARACTERS|_FACT_MODIFIERS) for a,b in zip(source,canonical)):return ()
    left=text[max(0,start-32):start];right=text[end:end+32]
    if medical_person_slot(text,start,end) or left.endswith(('患者','病人','医生','医师','护士','主任','教授')):return ()
    if any(w in left+right for w in ('小说','故事','虚构','比喻','作者','作品','书信','手稿','签名','写法','拼写','读作','这个词')):return ()
    if text[:start].rfind('《')>text[:start].rfind('》'):return ()
    before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
    if before in {'@','/','\\','_'} or after in {'/','\\','_'} or any(c.isascii() and c.isalnum() for c in (before,after)):return ()
    if re.match(r'^是(?:一)?(?:种|类)(?:相当|比较|较为|非常|十分|很)?(?:常见|罕见|多见|严重|慢性|急性)?的?(?:疾病|病症|炎症|病变)(?=[。！？!?，,；;：:\s]|$)',right) is None:return ()
    return ('medical_disease_definition_role',)


def medical_person_slot(text,start,end):
    left=text[max(0,start-16):start];right=text[end:end+12]
    titles=r'(医生|医师|护士|主任|教授|先生|女士)'
    if re.search(titles+r'(叫|名叫|名为)?\s*$',left):return True
    if re.search(r'(叫|名叫|名为)\s*$',left) and re.match(r'的?'+titles,right):return True
    if re.search(r'患者(叫|名叫|名为)\s*$',left):return True
    return left.endswith('患者') and re.match(r'(说|表示|告诉|回来|离开)',right) is not None


def registered_alias_spelling_evidence(text, start, end, definition, surface, namespace):
    """Verify the full sound of a declared medical alias in an object slot.

    Aliases must be registered equivalent names, not generated misspellings.
    Complete syllables, namespace uniqueness and role evidence are required;
    no generic fuzzy threshold is relaxed.
    """
    from .entity_boundaries import candidate_boundary_issue, _tokens
    source=text[start:end]
    if (not definition.enabled or definition.normalization_policy!='normalize'
        or definition.domain!='medical' or definition.entity_type!='TERM'
        or surface not in definition.aliases or surface==definition.canonical_text
        or not _CHINESE_NAME.fullmatch(surface) or not _CHINESE_NAME.fullmatch(source)
        or len(source)!=len(surface) or source==surface):return ()
    signature=pronunciation_signature(surface)
    if pronunciation_signature(source)!=signature or namespace.get(signature)!=frozenset({definition.canonical_text}):return ()
    left=text[max(0,start-16):start];right=text[end:end+16]
    if not re.search(r'(台|套|使用|启动|连接|上|做|接受|进行)(了|过)?(这个|这|一)?$',left):return ()
    if medical_person_slot(text,start,end) or re.match(r'的?(医生|医师|护士|主任|教授|先生|女士)',right):return ()
    if any(w in left+right for w in ('作者','签名','书信','礼物','小说','手稿','写法','拼写','读作','写成','这个词','人名')):return ()
    before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
    if before in {'@','/','\\','_'} or after in {'/','\\','_'} or any(c.isascii() and c.isalnum() for c in (before,after)):return ()
    if text[:start].rfind('《')>text[:start].rfind('》'):return ()
    issue=candidate_boundary_issue(text,start,end)
    if issue:
        # Jieba may join an alias's last character with the next 不. A device
        # quantity plus an explicit negative predicate supplies the boundary.
        # All other conflicts remain blocked.
        if not (issue=='chinese_word_boundary_conflict'
                and re.search(r'(台|套)$',left) and re.match(r'不(但|是|能|会|用|够|足|需要)',right)
                and not any(a<start<b for _,a,b in _tokens(text))
                and all(word==source[-1]+'不' and a==end-1 and b==end+1
                        for word,a,b in _tokens(text) if a<end<b)):return ()
    return ('registered_medical_alias_role',)


def registered_alias_near_evidence(text,start,end,definition,surface,namespace):
    """Opt-in one-letter sound confusion for a registered medical acronym.

    The declared alias supplies the complete spelling; no guessed error alias
    or clinical fact is inserted. Quantity, adjacent device noun or explicit
    operation slots supply role evidence instead of a general medical topic.
    """
    from rapidfuzz.distance import Levenshtein
    from .entity_boundaries import candidate_boundary_issue,_tokens,_tokenizer
    p=definition.spelling_context;source=text[start:end]
    if (p is None or p.allow_near_spelling is not True or not definition.enabled
        or definition.normalization_policy!='normalize' or definition.domain!='medical'
        or definition.entity_type!='TERM'
        or (p.canonical_text,p.entity_type,p.domain)!=(definition.canonical_text,definition.entity_type,definition.domain)
        or not re.fullmatch(r'[A-Za-z]{3,12}(?:手术|检查|检测|治疗)?',definition.canonical_text)
        or surface not in definition.aliases or not _CHINESE_NAME.fullmatch(surface)
        or not _CHINESE_NAME.fullmatch(source) or len(source)!=len(surface) or source==surface):return ()
    sound=pronunciation_signature(source);target=pronunciation_signature(surface)
    different=[i for i,(a,b) in enumerate(zip(sound,target)) if a!=b]
    if len(different)!=1 or Levenshtein.distance(sound[different[0]],target[different[0]])!=1:return ()
    i=different[0]
    if source[i] in _NEAR_SENSITIVE or _tokenizer() is None or _tokenizer().FREQ.get(source,0):return ()
    if any(a<=start+i<b and b-a>1 for _,a,b in _tokens(text)):return ()
    rivals=set()
    for signature,names in namespace.items():
        if (len(signature)==len(sound) and sum(a!=b for a,b in zip(signature,sound))<=1
            and sum(Levenshtein.distance(a,b) for a,b in zip(signature,sound))<=1):rivals.update(names)
    if rivals!={definition.canonical_text}:return ()
    left=text[max(0,start-24):start];right=text[end:end+16]
    if medical_person_slot(text,start,end) or re.match(r'的?(医生|医师|护士|主任|教授|先生|女士)',right):return ()
    if any(w in left+right for w in ('作者','签名','书信','礼物','小说','手稿','写法','拼写','读作','写成','这个词','人名','演员','假装')):return ()
    before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
    if before in {'@','/','\\','_'} or after in {'/','\\','_'} or any(c.isascii() and c.isalnum() for c in (before,after)):return ()
    if text[:start].rfind('《')>text[:start].rfind('》') or candidate_boundary_issue(text,start,end):return ()
    quantity=re.search(r'(台|套)$',left)
    device=re.match(r'的?(机器|设备|耗材|机身|管路)',right)
    action=re.search(r'(上|使用|启动|连接|做|接受|进行|完成|安排)(了|过)?(([一二三四五六七八九十百数几多]+|[0-9]+)(例|次|项|台|套))?$',left)
    if not(quantity or device or action):return ()
    return ('registered_medical_alias_near_role',)


def procedure_spelling_evidence(text, start, end, definition, namespace):
    """A declared medical procedure in an explicit operation slot.

    Allows a plain registered standard term without requiring hand-maintained
    misspelling aliases. Never infer a different anatomical source character.
    """
    canonical=definition.canonical_text
    observed=text[start:end]
    if (not definition.enabled or definition.normalization_policy!='normalize'
        or definition.entity_type!='TERM' or definition.domain!='medical'
        or not re.fullmatch(r'[\u3400-\u9fff]{4,16}',canonical)
        or not canonical.endswith(('手术','术')) or len(observed)!=len(canonical)
        or not _CHINESE_NAME.fullmatch(observed) or observed==canonical):
        return ()
    if medical_person_slot(text,start,end):return ()
    signature=pronunciation_signature(canonical)
    if pronunciation_signature(observed)!=signature or namespace.get(signature)!=frozenset({canonical}):
        return ()
    if any(a!=b and a in (_ANATOMICAL_CHARACTERS | _FACT_MODIFIERS) for a,b in zip(observed,canonical)):
        return ()
    before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
    if before in {'@','/','\\','_'} or after in {'/','\\','_'} or any(c.isascii() and c.isalnum() for c in (before,after)):
        return ()
    standalone=text.strip(' \t\r\n。！？!?，,；;：:')==observed
    if not standalone and not low_character_medical_role(text,start,end,definition):return ()
    surrounding=text[max(0,start-16):start]+text[end:end+16]
    if any(w in surrounding for w in ('写法','拼写','写成','读作','字形','这个词','错别字')):return ()
    return ('medical_procedure_role',)


def supports_procedure_stem(definition):
    """A declared anatomical name, not a generated complete procedure."""
    return (definition.enabled and definition.entity_type=='TERM'
        and definition.domain=='medical' and definition.normalization_policy=='normalize'
        and re.fullmatch(r'[\u3400-\u9fff]{3,16}',definition.canonical_text) is not None
        and definition.canonical_text.endswith(('肠','腺','神经','血管','瓣膜','胆囊','子宫','卵巢')))


def procedure_stem_spelling_evidence(text,start,end,definition,namespace):
    """Verify a registered anatomical stem before a literal procedure suffix.

    Restore only the registered stem. Never infer an action or complete a
    missing procedure. Prepositional 以...为/作为 constructions abstain.
    """
    source=text[start:end];canonical=definition.canonical_text
    if (not supports_procedure_stem(definition) or not _CHINESE_NAME.fullmatch(source)
        or len(source)!=len(canonical) or source==canonical):return ()
    signature=pronunciation_signature(canonical)
    if pronunciation_signature(source)!=signature or namespace.get(signature)!=frozenset({canonical}):return ()
    if any(a!=b and a in (_ANATOMICAL_CHARACTERS|_FACT_MODIFIERS) for a,b in zip(source,canonical)):return ()
    suffix=re.match(r'(?:切除|修补|置换|移植|切开|造口|消融|成形)(?:手术|术)',text[end:])
    if suffix is None:return ()
    procedure_end=end+suffix.end()
    complete_targets=namespace.get(pronunciation_signature(text[start:procedure_end]),frozenset())
    if complete_targets and complete_targets!=frozenset({canonical+suffix.group()}):return ()
    if procedure_end<len(text) and text[procedure_end] not in ' \t\r\n。！？!?，,；;：:':return ()
    left=text[max(0,start-32):start];right=text[procedure_end:procedure_end+32]
    if medical_person_slot(text,start,end):return ()
    if any(w in left+right for w in ('写法','拼写','写成','读作','字形','这个词','错别字','作者','签名','手稿','书信','礼物','小说','故事','作品','文献','依据','为例','举例','什么','哪一','哪种')):return ()
    if text[:start].rfind('《')>text[:start].rfind('》'):return ()
    before=text[start-1] if start else ''
    if before in {'@','/','\\','_'} or (before.isascii() and before.isalnum()):return ()
    # The continuation can be in the following ASR sentence/window.
    if re.match(r'^[。！？!?，,；;：:\s]*(?:为|作为|当作)',right):return ()
    clause_left=_SENTENCE_END.split(left)[-1]
    clause_right=_SENTENCE_END.split(right)[0]
    if clause_right.strip(' \t，,；;：:'):return ()
    standalone_clause=not clause_left.strip(' \t，,；;：:')
    explicit_action=re.search(r'(?:接受|进行|实施|安排|完成|施行|开展|展开|做)(?:了|过)?(?:一次|这项|该)?\s*$',clause_left)
    if not standalone_clause and explicit_action is None:return ()
    return ('medical_procedure_stem_role',)


def near_spelling_forms(canonical):
    if not re.fullmatch(r'[\u3400-\u9fff]{5,16}',canonical):return ()
    if not canonical.endswith(('手术','术','症','病','炎','综合征','潴留','衰竭')):return ()
    return tuple((i,canonical[:i],canonical[i+1:]) for i,c in enumerate(canonical) if c not in _NEAR_SENSITIVE)


def context_near_spelling_evidence(text,start,end,definition,namespace):
    """Explicitly enabled single-character, single-segment sound confusion."""
    from rapidfuzz.distance import Levenshtein
    from .entity_boundaries import _tokenizer, _tokens
    p=definition.spelling_context;canonical=definition.canonical_text;source=text[start:end]
    if (p is None or p.allow_near_spelling is not True or not definition.enabled
        or definition.normalization_policy!='normalize' or definition.entity_type!='TERM'
        or definition.domain!='medical'
        or (p.canonical_text,p.entity_type,p.domain)!=(canonical,definition.entity_type,definition.domain)
        or len(source)!=len(canonical) or not re.fullmatch(r'[\u3400-\u9fff]{5,16}',source)):
        return ()
    if medical_person_slot(text,start,end):return ()
    differences=[i for i,(a,b) in enumerate(zip(source,canonical)) if a!=b]
    if len(differences)!=1:return ()
    i=differences[0];key=(i,source[:i],source[i+1:])
    if key not in near_spelling_forms(canonical) or source[i] in _NEAR_SENSITIVE:
        return ()
    if namespace.get(key)!=frozenset({canonical}):return ()
    tokenizer=_tokenizer()
    if tokenizer is None or tokenizer.FREQ.get(source,0):return ()
    if any(a<=start+i<b and b-a>1 for _,a,b in _tokens(text)):return ()
    a,b=pronunciation_signature(source),pronunciation_signature(canonical)
    if len(a)!=len(b) or any(x!=y for j,(x,y) in enumerate(zip(a,b)) if j!=i):return ()
    if a[i]==b[i] or Levenshtein.distance(a[i],b[i])!=1:return ()
    before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
    if before in {'@','/','\\','_'} or after in {'/','\\','_'} or any(c.isascii() and c.isalnum() for c in (before,after)):return ()
    surrounding=text[max(0,start-16):start]+text[end:end+16]
    if any(w in surrounding for w in ('写法','拼写','写成','读作','字形','这个词','错别字')):return ()
    return _context_terms(text,start,end,p)


def internal_omission_forms(canonical):
    """Literal single-character omissions; never infer a clinical qualifier."""
    if not re.fullmatch(r'[\u3400-\u9fff]{5,16}', canonical):
        return ()
    return tuple((canonical[:i] + canonical[i+1:], i)
                 for i in range(1, len(canonical)-2)
                 if canonical[i] not in _FACT_MODIFIERS)


def context_omission_evidence(text, start, end, definition, namespace):
    """Verify an internal omission in an independently known long term.

    A glossary entry and scene context are necessary. The tokenizer remains
    independent of the glossary; known source words and word interiors veto
    completion, including a different anatomical name embedded in a phrase.
    """
    from .entity_boundaries import _tokenizer, _tokens
    policy = definition.spelling_context
    canonical = definition.canonical_text
    observed = text[start:end]
    if (policy is None or policy.allow_internal_omission is not True
        or not definition.enabled or definition.entity_type != 'TERM'
        or definition.normalization_policy != 'normalize'
        or (policy.canonical_text, policy.entity_type, policy.domain)
           != (canonical, definition.entity_type, definition.domain)
        or len(observed) != len(canonical)-1
        or namespace.get(observed) != frozenset({canonical})):
        return ()
    positions = [i for form, i in internal_omission_forms(canonical) if form == observed]
    if len(positions) != 1:
        return ()
    before = text[start-1] if start else ''
    after = text[end] if end < len(text) else ''
    if (before in {'@','/','\\','_'} or after in {'/','\\','_'}
        or any(c.isascii() and c.isalnum() for c in (before, after))):
        return ()
    surrounding = text[max(0,start-16):start] + text[end:end+16]
    if any(w in surrounding for w in ('写法','拼写','写成','读作','字形','这个词','错别字')):
        return ()
    tokenizer = _tokenizer()
    if tokenizer is None or tokenizer.FREQ.get(observed, 0):
        return ()
    if not tokenizer.FREQ.get(canonical, 0):
        # Rare verified monoclonal names are often absent from Jieba. Their
        # stable 单抗 suffix plus a local drug role supplies independent
        # evidence; the opt-in, uniqueness and insertion guards still apply.
        if definition.domain != 'medical' or not canonical.endswith('单抗') or not observed.endswith('单抗'):
            return ()
        left=_SENTENCE_END.split(text[max(0,start-24):start])[-1]
        right=_SENTENCE_END.split(text[end:end+24])[0]
        if medical_person_slot(text,start,end) or re.match(r'^的?(医生|医师|护士|主任|教授|先生|女士)',right):
            return ()
        if not any(w in left+right for w in ('药物','靶向药','抗体药','用药','抑制剂','抗体','抑制','治疗')):
            return ()
    insertion = start + positions[0]
    if any(a < insertion < b for _, a, b in _tokens(text)):
        return ()
    return _context_terms(text, start, end, policy)


def _context_terms(text, start, end, policy):
    left = _SENTENCE_END.split(text[max(0,start-32):start])[-1]
    right = _SENTENCE_END.split(text[end:end+32])[0]
    surrounding = left+'\0'+right
    evidence = tuple(word for word,weight in policy.keywords if word in surrounding)
    score = sum(weight for word,weight in policy.keywords if word in evidence)
    return evidence if score >= policy.minimum_score else ()


def low_character_medical_role(text, start, end, definition):
    """Extra role evidence before bypassing the mixed similarity floor.

    A sparse character match is admitted only in a medical procedure or
    diagnosis slot, not merely because a doctor/patient is mentioned nearby.
    """
    if definition.entity_type != 'TERM' or definition.domain != 'medical':
        return False
    right = _SENTENCE_END.split(text[end:end+16])[0]
    if any(w in right for w in ('礼物','书信','手稿','签名','笔迹','作者')):
        return False
    left = re.split(r'[，,；;。！？!?\n\r]',text[max(0,start-24):start])[-1]
    if definition.canonical_text.endswith(('手术','术','检查','检验','检测')):
        return re.search(r'(接受|进行|实施|安排|完成|施行|开展|做)(了|过)?(一次|一项|这个|这种|该)?\s*$',left) is not None
    return re.search(r'(患有|患上|诊断为|确诊为|诊断出|得了)\s*$',left) is not None


@lru_cache(maxsize=8192)
def pronunciation_signature(text: str) -> tuple[str, ...]:
    """Compare complete syllables; ASR's guessed characters do not prove tones."""
    from pypinyin import Style, lazy_pinyin
    return tuple(lazy_pinyin(text,style=Style.NORMAL))


def context_spelling_evidence(text, start, end, definition, namespace):
    """Return matched context terms for an explicitly enabled exact-sound name."""
    policy = definition.spelling_context
    canonical = unicodedata.normalize('NFKC',definition.canonical_text)
    observed = text[start:end]
    if (
        policy is None or not definition.enabled
        or definition.normalization_policy != 'normalize'
        or (policy.canonical_text,policy.entity_type,policy.domain)
           != (definition.canonical_text,definition.entity_type,definition.domain)
        or definition.entity_type not in {'PERSON','ORG','TERM','PROJECT','MODEL'}
        or _CHINESE_NAME.fullmatch(canonical) is None
        or _CHINESE_NAME.fullmatch(observed) is None
        or len(observed) != len(canonical)
        or observed == canonical
        or (definition.entity_type == 'PERSON' and observed[0] != canonical[0])
        or all(c in _NUMERIC_DATE_CHARACTERS for c in observed)
        or re.fullmatch(r'星期[一二三四五六日天]',observed)
    ):
        return ()
    if definition.domain=='medical' and definition.entity_type=='TERM':
        if medical_person_slot(text,start,end):return ()
        if canonical.endswith(('手术','术')) and any(a!=b and a in (_ANATOMICAL_CHARACTERS|_FACT_MODIFIERS) for a,b in zip(observed,canonical)):
            return ()
    # Registered real names do not authorize editing identifiers or titles.
    before = text[start-1] if start else ''
    after = text[end] if end<len(text) else ''
    if before in {'@','/','\\','_'} or after in {'/','\\','_'}:
        return ()
    if any(c.isascii() and c.isalnum() for c in (before,after)):
        return ()
    if definition.entity_type=='PERSON' and (
        text[:start].rfind('《')>text[:start].rfind('》')
        or text[max(0,start-2):start].endswith('姓')
    ):
        return ()
    signature = pronunciation_signature(canonical)
    if pronunciation_signature(observed) != signature:
        return ()
    if namespace.get(signature) != frozenset({definition.canonical_text}):
        return ()
    return _context_terms(text, start, end, policy)
