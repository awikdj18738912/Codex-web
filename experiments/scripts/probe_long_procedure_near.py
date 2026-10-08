"""Development-only long procedure probes after initial fresh-audio inspection."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,normalized
from evaluate_medical_entities import lexicon_occurrences
from evaluate_chinese_entity_rules import OUT,controls
from sweep_entity_rules import render
from system.entity_store import EntityStore,EntityDefinition,EntitySpellingContext
from system.entity_pronunciation import pronunciation_signature,_FACT_MODIFIERS,_context_terms,medical_person_slot
from system.entity_boundaries import candidate_boundary_issue,_tokens,_tokenizer
from system.entity_pipeline import prepare_entity_segment
from system.entity_matcher import EntityCandidateMatcher
from system.protection import EntityProtector

ACTION_WORDS=('切除','修补','置换','移植','整形','缝合','消融')


def long_evidence(text,start,end,d,namespace,strict=True):
    source=text[start:end];canonical=d.canonical_text;p=d.spelling_context
    if (not p or p.allow_near_spelling is not True or not d.enabled or d.domain!='medical'
        or d.entity_type!='TERM' or d.normalization_policy!='normalize'
        or (p.canonical_text,p.entity_type,p.domain)!=(canonical,d.entity_type,d.domain)
        or not re.fullmatch(r'[\u3400-\u9fff]{8,16}',canonical)
        or not canonical.endswith(('手术','术')) or len(source)!=len(canonical)
        or not re.fullmatch(r'[\u3400-\u9fff]{8,16}',source)):return ()
    changed=[i for i,(a,b) in enumerate(zip(source,canonical)) if a!=b]
    if len(changed)<4:return ()
    if any(source[i] in _FACT_MODIFIERS or canonical[i] in _FACT_MODIFIERS for i in changed):return ()
    sound=pronunciation_signature(source);target=pronunciation_signature(canonical)
    differences=[i for i,(a,b) in enumerate(zip(sound,target)) if a!=b]
    if len(differences)!=1:return ()
    i=differences[0]
    if Levenshtein.distance(sound[i],target[i])!=1:return ()
    action=set()
    for word in ACTION_WORDS:
        for match in re.finditer(re.escape(word)+r'(?=手术$|术$)',canonical):action.update(range(match.start(),match.end()))
    if strict and i not in action:return ()
    rivals=set()
    for sig,names in namespace.items():
        if len(sig)==len(sound) and sum(a!=b for a,b in zip(sig,sound))<=1 and sum(Levenshtein.distance(a,b) for a,b in zip(sig,sound))<=1:rivals.update(names)
    if rivals!={canonical}:return ()
    if _tokenizer() is None or _tokenizer().FREQ.get(source,0):return ()
    if strict and any(a<=start+j<b and b-a>1 for j in changed for _,a,b in _tokens(text)):return ()
    if candidate_boundary_issue(text,start,end) or medical_person_slot(text,start,end):return ()
    left=text[max(0,start-32):start];right=text[end:end+32]
    if any(w in left+right for w in ('作者','签名','手稿','书信','小说','故事','作品','文献','礼物','写法','拼写','读作','写成','这个词','人名','演员','假装')):return ()
    if text[:start].rfind('《')>text[:start].rfind('》'):return ()
    before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
    if before in {'@','/','\\','_'} or after in {'/','\\','_'} or any(c.isascii() and c.isalnum() for c in (before,after)):return ()
    operation=re.search(r'(接受|进行|实施|安排|完成|施行|开展|展开|做)(了|过)?(一次|这个|该)?$',left)
    timed_description=(re.search(r'(一般|常规|传统)的$',left)
        and re.match(r'(只)?需要(约|大约|预计)?[一二三四五六七八九十百半个0-9]+(个)?小时',right))
    if not(operation or timed_description):return ()
    evidence=_context_terms(text,start,end,p)
    return evidence if evidence else ()


def definitions():
    ds=EntityStore(OUT/'terms.db').list_entities(domain='medical');result=[]
    for d in ds:
        if re.fullmatch(r'[\u3400-\u9fff]{8,16}',d.canonical_text) and d.canonical_text.endswith(('手术','术')) and d.spelling_context:
            d=replace(d,spelling_context=replace(d.spelling_context,allow_near_spelling=True))
        result.append(d)
    return tuple(result)


def propose(text,ds,strict):
    namespace={}
    for d in ds:
        if d.enabled:
            for word in (d.canonical_text,*d.aliases):namespace.setdefault(pronunciation_signature(word),set()).add(d.canonical_text)
    blocked=EntityProtector(ds,protect_numeric_spans=False).protect(text).spans;edits=[]
    for d in ds:
        if not 8<=len(d.canonical_text)<=16:continue
        for run in re.finditer(r'[\u3400-\u9fff]+',text):
            for offset in range(len(run.group())-len(d.canonical_text)+1):
                a=run.start()+offset;b=a+len(d.canonical_text)
                if any(a<s.end and s.start<b for s in blocked):continue
                if long_evidence(text,a,b,d,namespace,strict):edits.append(dict(start=a,end=b,observed=text[a:b],canonical=d.canonical_text))
    return render(text,edits)


def run():
    ds=definitions()
    c='经皮甲状旁腺消融术';p=EntitySpellingContext((('患者',1.0),('需要',1.0)),1,'Constructed anatomical route control',c,'TERM','medical',allow_near_spelling=True)
    ds=(*ds,EntityDefinition(10001,c,'TERM','medical','normalize',0,(),spelling_context=p))
    c='胸腔镜肺叶切除术';p=replace(p,canonical_text=c)
    ds=(*ds,EntityDefinition(10002,c,'TERM','medical','normalize',0,(),spelling_context=p))
    c='截肢再植修补手术';p=replace(p,canonical_text=c)
    ds=(*ds,EntityDefinition(10003,c,'TERM','medical','normalize',0,(),spelling_context=p))
    rows=[]
    for folder in ['entity_reference_vocab_20261007','entity_rule_sweep_20261008/holdout','entity_alias_near_20261008/holdout','entity_chinese_rules_20261008']:rows+=read_jsonl(ROOT/'results'/folder/'asr.jsonl')
    data=json.loads((ROOT/'results/medical_entity_20261003/source/CMeEE_train.json').read_text());random.Random(20261011).shuffle(data)
    rows+=[dict(id=f'correct_{i}',variant='original',reference=r['text'],raw_text=r['text']) for i,r in enumerate(data[:1000])]
    negatives=controls()+[
        '患者接受京鼻甲壮旁现萧容术，手术需要一小时。',
        '患者接受腹股沟疝修骨手术，手术需要一小时。',
        '患者接受匈强敬胃页切厨术，手术需要一小时。',
        '患者接受节指再置修浦手术，手术需要一小时。',
        '作者介绍一般的复骨钩扇修骨手术只需要约一个半小时就能完成。',
    ]
    rows+=[dict(id=f'negative_{i}',variant='constructed',reference=t,raw_text=t) for i,t in enumerate(negatives)]
    p=EntityProtector(ds,protect_numeric_spans=False);m=EntityCandidateMatcher(ds,selected_domain='medical',mode='auto')
    counts={v:Counter() for v in ['wide','action_only']};changes={v:[] for v in counts};records=[];names={d.canonical_text for d in ds}
    for r in rows:
        baseline=prepare_entity_segment(r['raw_text'],p,m,allow_auto=True).baseline_text;old=Levenshtein.distance(normalized(r['reference']),normalized(baseline));outputs={}
        for v in counts:
            output,edits=propose(baseline,ds,v=='action_only');new=Levenshtein.distance(normalized(r['reference']),normalized(output))
            stats=counts[v];stats.update(samples=1,changed=int(output!=baseline),improved=int(new<old),regressed=int(new>old),error_delta=new-old)
            if r['variant'] in {'clean','noise'}:
                expected=Counter(e['text'] for e in lexicon_occurrences(r['reference'],names));have=Counter(e['text'] for e in lexicon_occurrences(output,names))
                stats['expected_mentions']+=sum(expected.values());stats['recovered_mentions']+=sum(min(n,have[w]) for w,n in expected.items())
            if output!=baseline:changes[v].append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],baseline=baseline,output=output,edits=edits,error_delta=new-old))
            outputs[v]=output
        records.append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],raw_text=r['raw_text'],baseline=baseline,outputs=outputs))
    write_json(OUT/'long_near_probe.json',dict(role='Development probe after fresh inspection; not independent acceptance',counts={v:dict(s) for v,s in counts.items()},changes=changes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (OUT/'long_near_outputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));print(json.dumps({v:dict(s) for v,s in counts.items()},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
