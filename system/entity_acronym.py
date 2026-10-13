"""Equivalent spoken number spellings for registered medical symbols.

Only explicit letters and the same number are normalized. No approximate
Chinese pronunciation, new symbol, qualifier, or numeric value is inferred.
"""
from functools import lru_cache
import re

_CHINESE_DIGITS='零一二三四五六七八九'
_ENGLISH_SMALL=('zero','one','two','three','four','five','six','seven','eight','nine',
    'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen')
_ENGLISH_TENS={20:'twenty',30:'thirty',40:'forty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',90:'ninety'}
_ROLE_WORDS=('基因','突变','抗体','抗原','受体','配体','激酶','蛋白','表达','阳性','阴性','靶点','靶向','细胞','治疗','抑制剂','分子')
_DIGIT_CHARACTERS=frozenset('0123456789零〇一二两三四五六七八九十百千')
_SENTENCE_END=re.compile(r'[。！？!?\n\r]')
_DISCUSSION=('写法','拼写','写成','读作','这个词','字形','人名','签名','文件名','用户名')


def symbol_key(value):
    match=re.fullmatch(r'([A-Za-z]+(?:-[A-Za-z]+)*)-?([0-9]{1,2})',value)
    if match is None or len(match[2])>1 and match[2].startswith('0'):return None
    return (match[1].replace('-','').upper(),int(match[2]))


def _number_forms(number):
    if number<10:chinese=_CHINESE_DIGITS[number]
    else:
        tens,ones=divmod(number,10)
        chinese=('' if tens==1 else _CHINESE_DIGITS[tens])+'十'+(_CHINESE_DIGITS[ones] if ones else '')
    chinese_forms={str(number),chinese,''.join(_CHINESE_DIGITS[int(c)] for c in str(number))}
    if number==2:chinese_forms.add('两')
    if number<20:english=_ENGLISH_SMALL[number]
    else:
        tens,ones=divmod(number,10)
        english=_ENGLISH_TENS[tens*10]+(f' {_ENGLISH_SMALL[ones]}' if ones else '')
    return chinese_forms,english


@lru_cache(maxsize=512)
def _pattern(letters,number):
    prefix=r'[ \t-]{0,3}'.join(re.escape(c) for c in letters)
    chinese,english=_number_forms(number)
    english_pattern=re.escape(english).replace(r'\ ',r'[ \t-]')
    suffix='|'.join(re.escape(x) for x in sorted(chinese,key=lambda x:(-len(x),x)))
    # English number words need a separator; HERtwo is an unrelated word.
    return re.compile(prefix+r'(?:[ \t-]{1,3}(?:'+english_pattern+r')|[ \t-]{0,3}(?:'+suffix+r'))',re.IGNORECASE)


class MedicalSymbolSpellings:
    def __init__(self,definitions):
        self.entries=[];namespace={}
        for d in definitions:
            if not d.enabled:continue
            for surface in (d.canonical_text,*d.aliases):
                key=symbol_key(surface)
                if key is not None:namespace.setdefault(key,set()).add(d.canonical_text)
        for d in definitions:
            key=symbol_key(d.canonical_text)
            if (key is not None and 2<=len(key[0])<=10 and d.enabled and d.domain=='medical'
                and d.entity_type=='TERM' and d.normalization_policy=='normalize'
                and namespace[key]=={d.canonical_text}):
                self.entries.append((d,_pattern(*key)))

    def occurrences(self,text):
        from .entity_pronunciation import medical_person_slot
        for definition,pattern in self.entries:
            for match in pattern.finditer(text):
                start,end=match.span();source=match[0]
                if source==definition.canonical_text:continue
                before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
                if (before and before in '@/\\_.') or (after and after in '/\\_.') or before.isascii() and before.isalnum():continue
                if after in _DIGIT_CHARACTERS:continue
                if after.isascii() and after.isalnum():
                    # Adjacent complete symbols occur in lists (PD一PDL一).
                    # Never accept an arbitrary suffix inside an identifier.
                    if source[-1].isascii() or not any(p.match(text,end) for _,p in self.entries):continue
                left=_SENTENCE_END.split(text[max(0,start-24):start])[-1]
                right=_SENTENCE_END.split(text[end:end+24])[0]
                if any(w in left+right for w in _DISCUSSION) or medical_person_slot(text,start,end):continue
                if left.endswith(('医生','医师','护士','主任','教授','先生','女士')):continue
                if re.match(r'^的?(医生|医师|护士|主任|教授|先生|女士)',right):continue
                if re.match(r'^\s*(?:年|月|日|天|小时|分钟|秒|岁|次|个|例|毫克|毫升|厘米|公斤|%|％|days?\b|years?\b|months?\b)',right,re.IGNORECASE):continue
                if not any(w in left+right for w in _ROLE_WORDS):continue
                yield definition,start,end,source
