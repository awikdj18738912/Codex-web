"""Compare fixed entity rule candidates, including explicit negative cases."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import re

from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,normalized
from evaluate_medical_entities import lexicon_occurrences
from system.entity_store import EntityStore
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.entity_pronunciation import pronunciation_signature,_context_terms,_NEAR_SENSITIVE,medical_person_slot
from system.entity_boundaries import candidate_boundary_issue,_tokens,_tokenizer
from system.protection import EntityProtector

OUT=ROOT/'results/entity_rule_sweep_20261008'
OLD=ROOT/'results/entity_reference_vocab_20261007'
VARIANTS=['current','near_distance2','near_length4','near_multi_glyph','ignore_word_interior','official_alias_exact','official_alias_sound']
ALIAS_SOURCE='https://tb.chinacdc.cn/jkjyzyk/twl/202103/W020210321669295949344.pdf'


def definitions(aliases=False):
    ds=EntityStore(OLD/'optimized_terms.db').list_entities(domain='medical')
    if aliases:
        ds=tuple(replace(d,aliases=tuple(dict.fromkeys((*d.aliases,'艾克膜' if d.canonical_text=='ECMO' else '艾克膜手术'))))
                 if d.canonical_text in {'ECMO','ECMO手术'} else d for d in ds)
    return ds


def alias_sound(text,ds):
    namespace={}
    for d in ds:
        for w in (d.canonical_text,*d.aliases):
            if re.fullmatch(r'[\u3400-\u9fff]{3,16}',w):namespace.setdefault(pronunciation_signature(w),set()).add(d.canonical_text)
    blocked=EntityProtector(ds,protect_numeric_spans=False).protect(text).spans
    edits=[]
    for d in ds:
        if d.entity_type!='TERM' or d.normalization_policy!='normalize' or not d.enabled:continue
        for a in d.aliases:
            if not re.fullmatch(r'[\u3400-\u9fff]{3,16}',a):continue
            target=pronunciation_signature(a)
            if namespace.get(target)!={d.canonical_text}:continue
            for run in re.finditer(r'[\u3400-\u9fff]+',text):
                for offset in range(len(run.group())-len(a)+1):
                    start=run.start()+offset;end=start+len(a);s=text[start:end]
                    if s==a or pronunciation_signature(s)!=target:continue
                    if any(start<p.end and p.start<end for p in blocked):continue
                    if candidate_boundary_issue(text,start,end) or medical_person_slot(text,start,end):continue
                    before=text[max(0,start-12):start];after=text[end:end+12]
                    # Explicit device quantity/usage or a procedure object slot.
                    if not (re.search(r'(台|套|使用|启动|连接|上|做|接受|进行)(了|过)?(这个|这|一)?$',before)):
                        continue
                    if any(w in after for w in ('作者','签名','书信','礼物','小说')):continue
                    edits.append(dict(start=start,end=end,observed=s,canonical=d.canonical_text,rule='registered_alias_sound'))
    return render(text,edits)


def render(text,edits):
    chosen=[]
    for e in sorted(edits,key=lambda e: (-(e['end']-e['start']),e['start'])):
        if len({r['canonical'] for r in edits if r['start']<e['end'] and e['start']<r['end']})>1:continue
        if any(r['start']<e['end'] and e['start']<r['end'] for r in chosen):continue
        chosen.append(e)
    for e in sorted(chosen,key=lambda e:e['start'],reverse=True):text=text[:e['start']]+e['canonical']+text[e['end']:]
    return text,chosen


def near(text,ds,variant):
    edits=[];blocked=EntityProtector(ds,protect_numeric_spans=False).protect(text).spans
    minimum=4 if variant=='near_length4' else 5
    max_glyphs=3 if variant=='near_multi_glyph' else 1
    max_distance=2 if variant in {'near_distance2','near_multi_glyph'} else 1
    for d in ds:
        c=d.canonical_text;p=d.spelling_context
        if not p or not re.fullmatch(r'[\u3400-\u9fff]{'+str(minimum)+r',16}',c):continue
        if not c.endswith(('术','症','病','炎','综合征','潴留','衰竭')) and variant!='near_multi_glyph':continue
        target=pronunciation_signature(c)
        for run in re.finditer(r'[\u3400-\u9fff]+',text):
            for offset in range(len(run.group())-len(c)+1):
                start=run.start()+offset;end=start+len(c);s=text[start:end]
                changes=[i for i,(a,b) in enumerate(zip(s,c)) if a!=b]
                if not 1<=len(changes)<=max_glyphs:continue
                if any(s[i] in _NEAR_SENSITIVE or c[i] in _NEAR_SENSITIVE for i in changes):continue
                if any(start<b.end and b.start<end for b in blocked):continue
                if candidate_boundary_issue(text,start,end) or medical_person_slot(text,start,end):continue
                if variant!='ignore_word_interior' and any(a<=start+i<b and b-a>1 for i in changes for _,a,b in _tokens(text)):continue
                sound=pronunciation_signature(s);different=[i for i,(a,b) in enumerate(zip(sound,target)) if a!=b]
                if len(different)>1 or any(Levenshtein.distance(sound[i],target[i])>max_distance for i in different):continue
                if not different:continue
                if not _context_terms(text,start,end,p):continue
                before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
                if before in {'@','/','\\','_'} or after in {'/','\\','_'} or any(x.isascii() and x.isalnum() for x in (before,after)):continue
                if any(w in text[max(0,start-16):start]+text[end:end+16] for w in ['写法','拼写','读作','作者','签名']):continue
                edits.append(dict(start=start,end=end,observed=s,canonical=c,rule=variant))
    return render(text,edits)


def run():
    ds=definitions();ads=definitions(True)
    tools={k:(EntityProtector(d,protect_numeric_spans=False),EntityCandidateMatcher(d,selected_domain='medical',mode='auto'))
           for k,d in [('normal',ds),('aliases',ads)]}
    rows=read_jsonl(OLD/'asr.jsonl')
    controls=json.load(open(ROOT/'results/medical_entity_20261003/source/CMeEE_train.json'));random.Random(20261008).shuffle(controls)
    rows+= [dict(id=f'correct_medical_{i}',variant='original',reference=r['text'],raw_text=r['text']) for i,r in enumerate(controls[:1000])]
    counts={v:Counter() for v in VARIANTS};changes={v:[] for v in VARIANTS}
    for r in rows:
        baseline=prepare_entity_segment(r['raw_text'],*tools['normal'],allow_auto=True).baseline_text
        ref=normalized(r['reference']);be=Levenshtein.distance(ref,normalized(baseline))
        for v in VARIANTS:
            output=baseline;edits=[]
            if v.startswith('official_alias'):
                output=prepare_entity_segment(r['raw_text'],*tools['aliases'],allow_auto=True).baseline_text
                if v=='official_alias_sound':output,edits=alias_sound(output,ads)
            elif v!='current':output,edits=near(output,ds,v)
            error=Levenshtein.distance(ref,normalized(output));s=counts[v]
            s.update(samples=1,changed=int(output!=baseline),improved=int(error<be),regressed=int(error>be),error_delta=error-be)
            if output!=baseline:changes[v].append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],baseline=baseline,output=output,edits=edits,error_delta=error-be))
    write_json(OUT/'development_sweep.json',dict(counts={v:dict(s) for v,s in counts.items()},changes=changes,
        official_alias='艾克膜',official_alias_source=ALIAS_SOURCE,role='Previously seen development audio plus text controls; not independent acceptance'))
    print(json.dumps({v:dict(s) for v,s in counts.items()},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
