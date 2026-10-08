"""Frozen entity-only alias near-sound experiments; inference never reads gold."""
from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import random
import re
from rapidfuzz.distance import Levenshtein
from evaluate_entity_library import ROOT,read_jsonl,write_json,normalized,decode
from evaluate_medical_entities import lexicon_occurrences
from sweep_entity_rules import definitions,render
from system.entity_store import EntitySpellingContext
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.protection import EntityProtector
from system.entity_pronunciation import pronunciation_signature,medical_person_slot,_NEAR_SENSITIVE
from system.entity_boundaries import candidate_boundary_issue,_tokens,_tokenizer

OUT=ROOT/'results/entity_alias_near_20261008'
PREVIOUS=ROOT/'results/entity_rule_sweep_20261008'
VARIANTS=['current','repeat_exact','near_quantity','near_device_context','near_actions','near_actions_repeat']


def defs():
    result=[]
    for d in definitions(True):
        if re.fullmatch(r'[A-Za-z]{3,12}(?:手术|检查|检测|治疗)?',d.canonical_text) and d.aliases:
            p=EntitySpellingContext((('机器',1.0),('设备',1.0),('耗材',1.0),('机身',1.0),('管路',1.0),('手术',1.0)),1.0,
                'Explicit near-sound permission for registered medical acronym aliases',d.canonical_text,d.entity_type,d.domain,allow_near_spelling=True)
            d=replace(d,spelling_context=p)
        result.append(d)
    return tuple(result)


def propose(text,ds,variant):
    namespace={}
    for d in ds:
        if d.enabled:
            for surface in (d.canonical_text,*d.aliases):
                if re.fullmatch(r'[\u3400-\u9fff]{3,16}',surface):namespace.setdefault(pronunciation_signature(surface),set()).add(d.canonical_text)
    protected=EntityProtector(ds,protect_numeric_spans=False).protect(text).spans;edits=[]
    for d in ds:
        p=d.spelling_context
        if d.domain!='medical' or d.entity_type!='TERM' or not d.enabled or d.normalization_policy!='normalize':continue
        if variant!='repeat_exact' and (not p or not p.allow_near_spelling or not re.fullmatch(r'[A-Za-z]{3,12}(?:手术|检查|检测|治疗)?',d.canonical_text)):continue
        for alias in d.aliases:
            if not re.fullmatch(r'[\u3400-\u9fff]{3,16}',alias):continue
            target=pronunciation_signature(alias)
            for run in re.finditer(r'[\u3400-\u9fff]+',text):
                for offset in range(len(run.group())-len(alias)+1):
                    start=run.start()+offset;end=start+len(alias);source=text[start:end]
                    if source==alias or any(start<s.end and s.start<end for s in protected):continue
                    left=text[max(0,start-24):start];right=text[end:end+16]
                    if medical_person_slot(text,start,end) or re.match(r'的?(医生|医师|护士|主任|教授|先生|女士)',right):continue
                    if any(w in left+right for w in ('作者','签名','书信','礼物','小说','手稿','写法','拼写','读作','写成','这个词','人名','演员','假装')):continue
                    before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
                    if before in {'@','/','\\','_'} or after in {'/','\\','_'} or any(c.isascii() and c.isalnum() for c in (before,after)):continue
                    if text[:start].rfind('《')>text[:start].rfind('》') or candidate_boundary_issue(text,start,end):continue
                    sound=pronunciation_signature(source)
                    different=[i for i,(a,b) in enumerate(zip(sound,target)) if a!=b]
                    if variant=='repeat_exact':
                        if different or namespace.get(target)!={d.canonical_text}:continue
                        if not re.search(r'(用|使用)(了|过)?$',left):continue
                        # An earlier exact registered occurrence supplies the
                        # topic; mention alone does not override other guards.
                        if not any(s.replacement==d.canonical_text and max(0,start-48)<=s.start<s.end<=start
                                   and not re.search(r'[。！？!?\n\r]',text[s.end:start]) for s in protected):continue
                    else:
                        if len(different)!=1 or Levenshtein.distance(sound[different[0]],target[different[0]])!=1:continue
                        i=different[0]
                        if source[i] in _NEAR_SENSITIVE or _tokenizer() is None or _tokenizer().FREQ.get(source,0):continue
                        if any(a<=start+i<b and b-a>1 for _,a,b in _tokens(text)):continue
                        # Every equal or equally near registered sound votes;
                        # a rival name or alias forces abstention.
                        rivals=set()
                        for sig,names in namespace.items():
                            if len(sig)==len(sound) and sum(a!=b for a,b in zip(sig,sound))<=1 and sum(Levenshtein.distance(a,b) for a,b in zip(sig,sound))<=1:rivals.update(names)
                        if rivals!={d.canonical_text}:continue
                        quantity=re.search(r'(台|套)$',left) is not None
                        device=re.match(r'的?(机器|设备|耗材|机身|管路)',right) is not None
                        action=re.search(r'(上|使用|启动|连接|做|接受|进行|完成|安排)(了|过)?(([一二三四五六七八九十百数几多]+|[0-9]+)(例|次|项|台|套))?$',left) is not None
                        if variant=='near_quantity' and not quantity:continue
                        if variant=='near_device_context' and not(quantity or device):continue
                        if variant=='near_actions' and not(quantity or device or action):continue
                    edits.append(dict(start=start,end=end,observed=source,canonical=d.canonical_text,rule=variant))
    edits=[e for e in edits if not any(r['start']==e['start'] and r['end']>e['end'] and r['canonical']==e['canonical']+text[e['end']:r['end']] for r in edits)]
    return render(text,edits)


def controls():
    return ['医生埃克姆正在查房。','患者请埃克姆医生看报告。','这台埃克姆医生推荐的机器很贵。',
        '患者埃克姆说头痛。','使用埃克姆作为人名。','医生让孩子假装螃蟹练习动作。',
        '这台艾克曼运行正常。','这台埃克蒙运行正常。','使用@埃克姆_01。','连接/埃克姆/file。',
        '使用《埃克姆》这本小说。','连接爱科目教学设备。','医生在机器旁讨论埃克姆的拼写。',
        'ECMO用于生命支持，医生用埃克莫作为人名。']


def evaluate(label):
    ds=defs();p=EntityProtector(ds,protect_numeric_spans=False);m=EntityCandidateMatcher(ds,selected_domain='medical',mode='auto')
    if label=='development':
        rows=read_jsonl(ROOT/'results/entity_reference_vocab_20261007/asr.jsonl')+read_jsonl(PREVIOUS/'holdout/asr.jsonl')
        data=json.loads((ROOT/'results/medical_entity_20261003/source/CMeEE_train.json').read_text());random.Random(20261009).shuffle(data)
        rows += [dict(id=f'correct_text_{i}',variant='original',raw_text=r['text'],reference=r['text']) for i,r in enumerate(data[:1000])]
        rows += [dict(id=f'constructed_{i}',variant='constructed',raw_text=t,reference=t) for i,t in enumerate(controls())]
    else:rows=read_jsonl(OUT/'holdout/asr.jsonl')
    counts={v:Counter() for v in VARIANTS};changes={v:[] for v in VARIANTS};records=[]
    names={d.canonical_text for d in ds}
    for r in rows:
        baseline=prepare_entity_segment(r['raw_text'],p,m,allow_auto=True).baseline_text
        ref=normalized(r['reference']);old=Levenshtein.distance(ref,normalized(baseline));outputs={}
        for v in VARIANTS:
            if v=='near_actions_repeat':
                output,edits=propose(baseline,ds,'near_actions')
                output,extra=propose(output,ds,'repeat_exact');edits+=extra
            else:output,edits=(baseline,[]) if v=='current' else propose(baseline,ds,v)
            new=Levenshtein.distance(ref,normalized(output))
            stats=counts[v];stats.update(samples=1,changed=int(output!=baseline),improved=int(new<old),regressed=int(new>old),error_delta=new-old)
            if r['variant'] in {'clean','noise'}:
                expected=Counter(e['text'] for e in lexicon_occurrences(r['reference'],names));have=Counter(e['text'] for e in lexicon_occurrences(output,names))
                stats['expected_mentions']+=sum(expected.values());stats['recovered_mentions']+=sum(min(n,have[w]) for w,n in expected.items())
            if output!=baseline:changes[v].append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],baseline=baseline,output=output,edits=edits,error_delta=new-old))
            outputs[v]=output
        records.append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],raw_text=r['raw_text'],outputs=outputs))
    write_json(OUT/f'{label}.json',dict(counts={v:dict(c) for v,c in counts.items()},changes=changes))
    (OUT/f'{label}_outputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    print(json.dumps({v:dict(c) for v,c in counts.items()},ensure_ascii=False,indent=2),flush=True)


def prepare():
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    output=OUT/'holdout';output.mkdir(exist_ok=True)
    if (output/'selection.json').exists():raise FileExistsError('Selection frozen')
    excluded=set()
    for folder in ['entity_medical_scenes_20261006','entity_medical_scenes_20261006/holdout','entity_reference_vocab_20261007','entity_rule_sweep_20261008/holdout']:
        for r in json.loads((ROOT/'results'/folder/'selection.json').read_text())['samples']:excluded.add(r['source_row'])
    names={d.canonical_text for d in defs()};pools={'acronym':[],'other_medical':[],'control':[]}
    source=ROOT/'results/medical_entity_20261003/source/chinese_train.parquet'
    rows=pq.read_table(source).to_pylist()
    for index,r in enumerate(rows):
        if index in excluded or not 3<=r['duration']<=20:continue
        if re.search(r'[A-Za-z]',re.sub('TAVI|ENB|PSA|MRI|ECMO','',r['text'])):continue
        if not 2<=sum('\u3400'<=c<='\u9fff' for c in r['text'])/r['duration']<=8:continue
        entities=lexicon_occurrences(r['text'],names);group='acronym' if any(e['text'].startswith('ECMO') for e in entities) else 'other_medical' if entities else 'control'
        pools[group].append((index,r,entities))
    samples=[]
    for group,n in [('acronym',10),('other_medical',18),('control',8)]:
        pool=pools[group];random.Random('20261009:'+group).shuffle(pool)
        for index,r,entities in pool[:n]:
            record_id=f'MultiMed_Chinese_train_{index:04d}';path=output/'audio'/(record_id+'.wav');path.parent.mkdir(exist_ok=True)
            data,rate=sf.read(io.BytesIO(r['audio']['bytes']),dtype='float32')
            if data.ndim>1:data=data.mean(axis=1)
            if rate!=16000:
                factor=gcd(rate,16000);data=resample_poly(data,16000//factor,rate//factor).astype('float32')
            sf.write(path,data,16000,subtype='FLOAT')
            samples.append(dict(id=record_id,source_row=index,audio_path=str(path),reference=r['text'],group=group,entities=entities,original_audio_sha256=hashlib.sha256(r['audio']['bytes']).hexdigest()))
    write_json(output/'selection.json',dict(seed=20261009,noise_snr_db=5,samples=samples,pool_counts={k:len(v) for k,v in pools.items()},previous_rows_excluded=sorted(excluded),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    write_json(OUT/'protocol.json',dict(variants=VARIANTS,rule_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection_sha256=hashlib.sha256((output/'selection.json').read_bytes()).hexdigest(),criterion='No new text/audio regressions; choose most validated entity fixes; no aliases guessed from ASR',reference_at_inference=False))
    print('FROZEN',len(samples),'recordings',Counter(r['group'] for r in samples),flush=True)


if __name__=='__main__':
    import argparse
    from types import SimpleNamespace
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','prepare','asr','holdout']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='asr':decode(SimpleNamespace(output=OUT/'holdout',asr_url='http://127.0.0.1:8766'))
    else:evaluate(args.stage)
