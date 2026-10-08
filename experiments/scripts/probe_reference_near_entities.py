"""Isolated one-character near-sound probe; never imported by production."""
from collections import Counter
from dataclasses import replace
import json
import re

from rapidfuzz.distance import Levenshtein
from evaluate_reference_entity_vocab import OUT
from evaluate_entity_library import read_jsonl, write_json, normalized
from system.entity_store import EntityStore
from system.entity_pipeline import prepare_entity_segment
from system.entity_matcher import EntityCandidateMatcher
from system.entity_boundaries import candidate_boundary_issue, _tokens, _tokenizer
from system.entity_pronunciation import pronunciation_signature, _context_terms, _FACT_MODIFIERS
from system.protection import EntityProtector

SENSITIVE = _FACT_MODIFIERS | frozenset('肝胆肺胃肠心肾脾脑骨筋髓腺胰眼耳颈胸腹盆股膝髋颅椎脊血尿自异同癌瘤炎菌病毒碳氮氢氧磷硫铁镁钾钠钙氯氟锌铜锰钒有是在了也因或与但正')


def definitions():
    ds=EntityStore(OUT/'reference_terms.db').list_entities(domain='medical')
    names=[d.canonical_text for d in ds]
    # Same-domain terms already present in the prebuilt list provide context;
    # no reference sentences, ASR errors, or per-record answers are added.
    return tuple(replace(d,spelling_context=replace(d.spelling_context,
        keywords=tuple(dict([*d.spelling_context.keywords,*[(w,1.0) for w in names if w not in d.canonical_text]]).items())))
        if d.spelling_context else d for d in ds)


def apply(text, ds):
    protected=EntityProtector(ds,protect_numeric_spans=False).protect(text)
    candidates=[];tokenizer=_tokenizer()
    for d in ds:
        target=d.canonical_text;p=d.spelling_context
        if not d.enabled or d.normalization_policy!='normalize' or not p or not re.fullmatch(r'[\u3400-\u9fff]{5,16}',target):continue
        if not target.endswith(('手术','术','症','病','炎','综合征','潴留','衰竭')):continue
        target_sound=pronunciation_signature(target)
        for match in re.finditer(r'[\u3400-\u9fff]+',text):
            for offset in range(len(match.group())-len(target)+1):
                start=match.start()+offset;end=start+len(target);source=text[start:end]
                different=[i for i,(a,b) in enumerate(zip(source,target)) if a!=b]
                if len(different)!=1:continue
                i=different[0]
                if source[i] in SENSITIVE or target[i] in SENSITIVE:continue
                if any(start<s.end and s.start<end for s in protected.spans):continue
                if candidate_boundary_issue(text,start,end):continue
                if tokenizer is None or tokenizer.FREQ.get(source,0):continue
                if any(a<=start+i<b and b-a>1 for _,a,b in _tokens(text)):continue
                source_sound=pronunciation_signature(source)
                if any(a!=b for j,(a,b) in enumerate(zip(source_sound,target_sound)) if j!=i):continue
                if source_sound[i]==target_sound[i] or Levenshtein.distance(source_sound[i],target_sound[i])!=1:continue
                before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
                if any(c.isascii() and c.isalnum() for c in (before,after)) or before in '@/\\_' or after in '/\\_':continue
                surrounding=text[max(0,start-16):start]+text[end:end+16]
                if any(w in surrounding for w in ['写法','拼写','写成','读作','字形','这个词','错别字']):continue
                evidence=_context_terms(text,start,end,p)
                if not evidence:continue
                candidates.append(dict(start=start,end=end,observed=source,canonical=target,evidence=evidence))
    chosen=[]
    for c in candidates:
        rivals={r['canonical'] for r in candidates if r['start']<c['end'] and c['start']<r['end']}
        if len(rivals)!=1 or any(r['start']<c['end'] and c['start']<r['end'] for r in chosen):continue
        chosen.append(c)
    output=text
    for c in sorted(chosen,key=lambda c:c['start'],reverse=True):output=output[:c['start']]+c['canonical']+output[c['end']:]
    return output,chosen


def run():
    ds=definitions();protector=EntityProtector(ds,protect_numeric_spans=False);matcher=EntityCandidateMatcher(ds,selected_domain='medical',mode='auto')
    rows=read_jsonl(OUT/'asr.jsonl')
    cmeee=json.load(open('results/medical_entity_20261003/source/CMeEE_train.json'))
    import random
    random.Random(20261007).shuffle(cmeee)
    rows+= [dict(id=f'correct_medical_{i}',variant='original',raw_text=r['text'],reference=r['text']) for i,r in enumerate(cmeee[:1000])]
    stats=Counter();changes=[]
    for r in rows:
        before=prepare_entity_segment(r['raw_text'],protector,matcher,allow_auto=True).baseline_text
        after,edits=apply(before,ds)
        a,b=Levenshtein.distance(normalized(r['reference']),normalized(before)),Levenshtein.distance(normalized(r['reference']),normalized(after))
        stats.update(samples=1,changed=int(before!=after),improved=int(b<a),regressed=int(b>a),error_delta=b-a)
        if edits:changes.append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],source=r['raw_text'],baseline=before,output=after,edits=edits,error_delta=b-a))
    write_json(OUT/'near_probe.json',dict(stats=dict(stats),changes=changes,role='Development-only probe, no production enablement'))
    print(json.dumps(dict(stats),ensure_ascii=False),flush=True)
    for c in changes:print(c,flush=True)


if __name__=='__main__':run()
