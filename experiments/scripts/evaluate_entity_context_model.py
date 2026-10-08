"""Train-only entity spelling experiment with a small character context model.

References are used by report(), never by candidate construction or scoring.
This prototype runs outside production while its precision is measured.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import time

from evaluate_entity_library import ROOT, normalized, read_jsonl, append_jsonl, write_json
from system.entity_matcher import _pinyin_tokens
from system.entity_boundaries import _tokens
from system.protection import strip_refiner_key_suffix
from rapidfuzz.distance import JaroWinkler, Levenshtein

OUT=ROOT/'results/entity_context_20261006'
MED=ROOT/'results/medical_entity_20261003'


def clean(text):
    return re.sub(r'[^\u3400-\u9fff]', '|', re.sub(r'<[^>]*>','',text))


def train():
    import pyarrow.parquet as pq
    cmeee=json.loads((MED/'source/CMeEE_train.json').read_text())
    speech=pq.read_table(MED/'source/chinese_train.parquet',columns=['text']).to_pylist()
    counts=[Counter() for _ in range(4)]
    for rows,weight in [(cmeee,1),(speech,3)]:
        for row in rows:
            for run in re.split(r'\|+',clean(row['text'])):
                if not run:continue
                s='^^^'+run+'$'
                for n in range(1,5):
                    for i in range(3,len(s)):
                        counts[n-1][s[i-n+1:i+1]]+=weight
    vocabulary=json.loads((MED/'vocabulary.json').read_text())
    write_json(OUT/'character_model.json',dict(version=1,counts=[dict(c) for c in counts],
         training_sources=[dict(path=str(MED/'source'/name),sha256=hashlib.sha256((MED/'source'/name).read_bytes()).hexdigest())
                           for name in ['CMeEE_train.json','chinese_train.parquet']],
         training_rows=dict(cmeee=len(cmeee),multimed=len(speech))))
    print('Model counts',[len(c) for c in counts],flush=True)


class CharacterModel:
    def __init__(self,path):
        pack=json.loads(Path(path).read_text());self.counts=pack['counts']
        self.contexts=[Counter() for _ in self.counts]
        for n,grams in enumerate(self.counts):
            for gram,count in grams.items():self.contexts[n][gram[:-1]]+=count
        self.total=sum(self.counts[0].values());self.size=len(self.counts[0])

    def score(self,text):
        total=0.0
        for run in re.split(r'\|+',clean(text)):
            if not run:continue
            s='^^^'+run+'$'
            for i in range(3,len(s)):
                gram=s[i];p=(self.counts[0].get(gram,0)+0.1)/(self.total+0.1*self.size)
                for n in range(1,4):
                    gram=s[i-n:i+1];context=self.contexts[n].get(gram[:-1],0)
                    if context:
                        p=(self.counts[n].get(gram,0)+5*p)/(context+5)
                total+=math.log(max(p,1e-12))
        return total


class ContextMatcher:
    def __init__(self,words,model):
        self.words=words;self.model=model
        self.index=defaultdict(list)
        self.targets=[]
        for word in words:
            if not re.fullmatch(r'[\u3400-\u9fff]{2,10}',word):continue
            tokens=_pinyin_tokens(word)
            self.targets.append((word,tokens))
            for initial in set(p[0] for p in tokens if p):
                self.index[initial].append((word,tokens))

    def candidates(self,text):
        results=[]
        exact=[(m.start(),m.end()) for w in self.words for m in re.finditer(re.escape(w),text)]
        for run in re.finditer(r'[\u3400-\u9fff]+',text):
            s=run.group();py=_pinyin_tokens(s)
            for i in range(len(s)):
                for length in range(2,min(10,len(s)-i)+1):
                    observed=s[i:i+length];observed_py=py[i:i+length]
                    pool={w:(w,p) for token in observed_py for w,p in self.index.get(token[0],()) if len(w)==length}
                    a=run.start()+i;b=a+length
                    # Existing canonical words are evidence of a complete span;
                    # do not cut into or rewrite them.
                    if any(a<e and d<b for d,e in exact):continue
                    prefix=text[max(0,a-8):a];suffix=text[b:b+8]
                    original_score=self.model.score(prefix+observed+suffix)
                    for word,target_py in pool.values():
                        if observed==word:continue
                        sound=JaroWinkler.normalized_similarity(''.join(observed_py),''.join(target_py))
                        if sound<0.8:continue
                        gain=self.model.score(prefix+word+suffix)-original_score
                        if gain<0:continue
                        results.append(dict(start=a,end=b,observed=observed,canonical=word,
                                            sound=round(sound,5),gain=round(gain,5)))
        return sorted(results,key=lambda r:(-r['gain'],-r['sound'],r['start']))


def evaluate(args):
    model=CharacterModel(OUT/'character_model.json')
    vocabulary=OUT/'expanded_vocabulary.json' if args.expanded else MED/'vocabulary.json'
    words=[v['canonical_text'] for v in json.loads(vocabulary.read_text())]
    matcher=ContextMatcher(words,model)
    src=MED/('development/asr.jsonl' if args.split=='dev' else 'asr.jsonl')
    rows=read_jsonl(src)
    if args.split=='general':rows=read_jsonl(ROOT/'results/entity_library_20261003/asr.jsonl')
    if args.split in {'fresh','mmed_dev','mmed_test'}:rows=read_jsonl(OUT/f'{args.split}_asr.jsonl')
    if args.expanded:
        from evaluate_medical_entities import lexicon_occurrences
        rows=[dict(r,entities=lexicon_occurrences(r['reference'],words)) for r in rows]
    prefix='ex_' if args.expanded else ''
    dest=OUT/f'{prefix}char_{args.split}.jsonl'
    if dest.exists():raise ValueError('Refusing to overwrite frozen experiment')
    for i,row in enumerate(rows):
        cands=matcher.candidates(row['raw_text'])
        append_jsonl(dest,dict(id=row['id'],variant=row['variant'],source=row['raw_text'],
                              reference=row['reference'],entities=row['entities'],candidates=cands))
        if i%20==0:print(i+1,len(rows),len(cands),flush=True)
    report(args.split,prefix)


def output_for(row,threshold,sound):
    edits=[]
    for c in row['candidates']:
        if c['gain']<threshold or c['sound']<sound:continue
        if any(c['start']<e['end'] and e['start']<c['end'] for e in edits):continue
        edits.append(c)
    text=row['source']
    for e in sorted(edits,key=lambda c:c['start'],reverse=True):text=text[:e['start']]+e['canonical']+text[e['end']:]
    return text,edits


def report(split,prefix=''):
    rows=read_jsonl(OUT/f'{prefix}char_{split}.jsonl')
    summaries={}
    for threshold in [3,5,7,9,12,15]:
        for sound in [0.8,0.85,0.9,0.95]:
            stats=Counter();changes=[]
            for r in rows:
                text,edits=output_for(r,threshold,sound)
                a,b,ref=normalized(r['source']),normalized(text),normalized(r['reference'])
                ae,be=Levenshtein.distance(ref,a),Levenshtein.distance(ref,b)
                stats['samples']+=1;stats['improved']+=be<ae;stats['worsened']+=be>ae
                stats['error_delta']+=be-ae;stats['changed']+=a!=b
                for ent in r['entities']:
                    term=normalized(ent['text'])
                    stats['gain']+=term in b and term not in a;stats['loss']+=term in a and term not in b
                if edits:changes.append(dict(id=r['id'],variant=r['variant'],source=r['source'],output=text,reference=r['reference'],edits=edits,error_delta=be-ae))
            key=f'{threshold}:{sound}';summaries[key]=dict(stats)
            if threshold==9 and sound==0.85:write_json(OUT/f'{prefix}char_{split}_changes.json',changes)
    write_json(OUT/f'{prefix}char_{split}_summary.json',summaries)
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)


def filtered_candidates(row):
    bounds={x for _,a,b in _tokens(row['source']) for x in [a,b]}
    candidates=[]
    for c in row['candidates']:
        if c['gain']<3 or c['start'] not in bounds or c['end'] not in bounds:continue
        a,b=_pinyin_tokens(c['observed']),_pinyin_tokens(c['canonical'])
        same=sum(x==y for x,y in zip(a,b))
        if same<len(a)-1 or (len(a)>2 and same/len(a)<.66):continue
        candidates.append(c)
    return candidates[:16]


def neural(args):
    from system.live_qwen_refiner import TransformersRefiner
    from system.refinement_protocol import apply_structured_patch
    from difflib import SequenceMatcher
    model=TransformersRefiner(Path('/home/aim0/data/models/ASR/AgenticASR-Refiner'),'cuda:0',256)
    src=OUT/f'char_{args.split}.jsonl';dest=OUT/f'neural_{args.split}.jsonl'
    complete={(r['id'],r['variant']) for r in read_jsonl(dest)}
    for i,row in enumerate(read_jsonl(src)):
        if (row['id'],row['variant']) in complete:continue
        candidates=filtered_candidates(row)
        response=None;elapsed=0;edits=[];output=row['source']
        if candidates:
            hints=list(dict.fromkeys(c['canonical'] for c in candidates))
            response,elapsed=model.refine_with_numeric_rules(row['source'],entity_hints=hints,
                entity_candidates=[dict(observed=c['observed'],canonical=c['canonical'],type='TERM') for c in candidates])
            rendered,payload,error=apply_structured_patch(row['source'],response)
            if error is None:
                rendered=strip_refiner_key_suffix(rendered)
                # Align to actual output, not the KEY list. Only a complete
                # candidate replacement at the original span is evidence.
                mapping=SequenceMatcher(None,row['source'],rendered,autojunk=False).get_opcodes()
                for c in candidates:
                    if row['source'].count(c['observed'])!=1:continue
                    if rendered.count(c['canonical'])<=row['source'].count(c['canonical']):continue
                    if rendered.count(c['observed'])>=row['source'].count(c['observed']):continue
                    blocks=[b for b in mapping if b[1]<c['end'] and c['start']<b[2] and b[0]!='equal']
                    if not blocks:continue
                    lo=min(b[3] for b in blocks);hi=max(b[4] for b in blocks)
                    if c['canonical'] not in rendered[max(0,lo-len(c['canonical'])):hi+len(c['canonical'])]:continue
                    if any(c['start']<e['end'] and e['start']<c['end'] for e in edits):continue
                    edits.append(c)
                for c in sorted(edits,key=lambda c:c['start'],reverse=True):
                    output=output[:c['start']]+c['canonical']+output[c['end']:]
        append_jsonl(dest,dict(id=row['id'],variant=row['variant'],source=row['source'],reference=row['reference'],entities=row['entities'],
                              output=output,candidates=candidates,response=response,model_ms=elapsed,edits=edits))
        if candidates or i%20==0:print(i+1,row['id'],row['variant'],'EDITS',edits,'MODEL',response,flush=True)
    neural_report(args.split)


def neural_report(split):
    stats=Counter();changes=[]
    for r in read_jsonl(OUT/f'neural_{split}.jsonl'):
        a,b,ref=normalized(r['source']),normalized(r['output']),normalized(r['reference'])
        ae,be=Levenshtein.distance(ref,a),Levenshtein.distance(ref,b)
        stats['samples']+=1;stats['improved']+=be<ae;stats['worsened']+=be>ae
        stats['error_delta']+=be-ae;stats['changed']+=a!=b;stats['model_calls']+=r['response'] is not None
        for ent in r['entities']:
            term=normalized(ent['text']);stats['gain']+=term in b and term not in a;stats['loss']+=term in a and term not in b
        if r['edits']:changes.append(dict(**r,error_delta=be-ae))
    write_json(OUT/f'neural_{split}_summary.json',dict(stats));write_json(OUT/f'neural_{split}_changes.json',changes)
    print(dict(stats),flush=True)


INSTRUCT_PROMPT='''你核对医学语音转写中的实体候选。候选可能全部错误。
结合原句语法和语境判断，正常表达必须保持。不要把普通动词、虚词、数字、人名、
手术衣或手术视野等正确词改成相似术语。读音相近和词库存在都不证明应该替换。
只有原文在此处不合理、候选明确符合句意且是同一实体的识别错字时选择候选。
有歧义返回 {"action":"keep"}。
确定纠正时返回 {"action":"replace","choice":"对应候选ID"}。
只返回一个JSON对象，每次最多选择一处，不要解释。'''


def instruct(args):
    import httpx
    prefix='ex_' if args.expanded else ''
    tag='_'+args.tag if args.tag else ''
    dest=OUT/f'{prefix}instruct_{args.split}{tag}.jsonl'
    complete={(r['id'],r['variant']) for r in read_jsonl(dest)}
    with httpx.Client(base_url=args.url,timeout=45) as client:
        for i,row in enumerate(read_jsonl(OUT/f'{prefix}char_{args.split}.jsonl')):
            if (row['id'],row['variant']) in complete:continue
            candidates=filtered_candidates(row)[:8]
            response=None;elapsed=0;edits=[];output=row['source'];status='no_candidates'
            if candidates:
                low=max(0,min(c['start'] for c in candidates)-45)
                high=min(len(row['source']),max(c['end'] for c in candidates)+45)
                # Chinese context is bounded to the model's 1024 token window.
                if high-low>240:high=low+240
                candidates=[c for c in candidates if c['end']<=high]
                message=dict(text=row['source'][low:high],candidates=[dict(id=str(j),source=c['observed'],target=c['canonical']) for j,c in enumerate(candidates)])
                started=time.perf_counter()
                try:
                    result=client.post('/v1/chat/completions',json=dict(model=args.model,
                        temperature=0,max_tokens=96,response_format={'type':'json_object'},
                        messages=[dict(role='system',content=INSTRUCT_PROMPT),dict(role='user',content=json.dumps(message,ensure_ascii=False))])).raise_for_status().json()
                    response=result['choices'][0]['message']['content'];patch=json.loads(response)
                    status='keep'
                    if patch.get('action')=='replace':
                        choice=patch.get('choice');allowed={str(j):c for j,c in enumerate(candidates)}
                        if choice not in allowed:raise ValueError('unknown choice')
                        c=allowed[choice]
                        edits=[c];output=output[:c['start']]+c['canonical']+output[c['end']:];status='replace'
                except (ValueError,KeyError,TypeError,httpx.HTTPError) as error:
                    status='fallback:'+type(error).__name__
                elapsed=(time.perf_counter()-started)*1000
            append_jsonl(dest,dict(id=row['id'],variant=row['variant'],source=row['source'],reference=row['reference'],entities=row['entities'],
                                 output=output,candidates=candidates,response=response,model_ms=elapsed,edits=edits,status=status))
            if candidates or i%20==0:print(i+1,row['id'],row['variant'],status,edits,response,flush=True)
    instruct_report(args.split,prefix,tag)


def instruct_report(split,prefix='',tag=''):
    stats=Counter();changes=[]
    for r in read_jsonl(OUT/f'{prefix}instruct_{split}{tag}.jsonl'):
        a,b,ref=normalized(r['source']),normalized(r['output']),normalized(r['reference'])
        ae,be=Levenshtein.distance(ref,a),Levenshtein.distance(ref,b)
        stats['samples']+=1;stats['improved']+=be<ae;stats['worsened']+=be>ae
        stats['error_delta']+=be-ae;stats['changed']+=a!=b;stats['model_calls']+=bool(r['candidates'])
        for ent in r['entities']:
            term=normalized(ent['text']);stats['gain']+=term in b and term not in a;stats['loss']+=term in a and term not in b
        if r['edits']:changes.append(dict(**r,error_delta=be-ae))
    write_json(OUT/f'{prefix}instruct_{split}{tag}_summary.json',dict(stats));write_json(OUT/f'{prefix}instruct_{split}{tag}_changes.json',changes)
    print(dict(stats),flush=True)


DETECT_PROMPT='''核对标记词的语音识别拼写。只处理同音或近音的医学术语错字。
原词合理时keep，包括普通词、人名、病情概念、尚未说完的句子。禁止改变事实或补写。
如果确定错字，按原读音还原词语，输出source和target；不确定就keep。
请简短判断，不复述任务。只输出JSON：{"action":"keep"}或
{"action":"replace","source":"标记词原文","target":"正确拼写"}。'''


def detect(args):
    import httpx
    prefix='ex_' if args.expanded else '';tag='_'+args.tag if args.tag else ''
    dest=OUT/f'{prefix}detect_{args.split}{tag}.jsonl'
    complete={(r['id'],r['variant']) for r in read_jsonl(dest)}
    with httpx.Client(base_url=args.url,timeout=90) as client:
        for i,row in enumerate(read_jsonl(OUT/f'{prefix}char_{args.split}.jsonl')):
            if (row['id'],row['variant']) in complete:continue
            candidates=filtered_candidates(row)[:12]
            response=None;reasoning=None;elapsed=0;edits=[];output=row['source'];status='no_candidates'
            if candidates:
                low=max(0,min(c['start'] for c in candidates)-45)
                high=min(len(row['source']),max(c['end'] for c in candidates)+45)
                if high-low>260:high=low+260
                candidates=[c for c in candidates if c['end']<=high]
                suspects=list(dict.fromkeys(c['observed'] for c in candidates))
                message=dict(text=row['source'][low:high],suspects=[dict(text=w,pinyin=' '.join(_pinyin_tokens(w))) for w in suspects])
                started=time.perf_counter()
                try:
                    result=client.post('/v1/chat/completions',json=dict(model=args.model,temperature=0,max_tokens=1024,
                        response_format={'type':'json_object'},messages=[dict(role='system',content=DETECT_PROMPT),
                        dict(role='user',content=json.dumps(message,ensure_ascii=False))])).raise_for_status().json()
                    m=result['choices'][0]['message'];response=m['content'];reasoning=m.get('reasoning_content');patch=json.loads(response)
                    status='keep'
                    if patch.get('action')=='replace':
                        options=[c for c in candidates if c['observed']==patch.get('source') and c['canonical']==patch.get('target')]
                        if len(options)!=1:status='unverified_proposal'
                        else:
                            c=options[0];edits=[c];output=output[:c['start']]+c['canonical']+output[c['end']:];status='replace'
                except (ValueError,KeyError,TypeError,httpx.HTTPError) as error:status='fallback:'+type(error).__name__
                elapsed=(time.perf_counter()-started)*1000
            append_jsonl(dest,dict(id=row['id'],variant=row['variant'],source=row['source'],reference=row['reference'],entities=row['entities'],
                    output=output,candidates=candidates,response=response,reasoning=reasoning,model_ms=elapsed,edits=edits,status=status))
            if candidates or i%20==0:print(i+1,row['id'],row['variant'],status,edits,response,flush=True)
    # Same paired report shape; keep detector and selector artifacts separate.
    detection_report(args.split,prefix,tag)


def detection_report(split,prefix='',tag=''):
    stats=Counter();changes=[]
    for r in read_jsonl(OUT/f'{prefix}detect_{split}{tag}.jsonl'):
        a,b,ref=normalized(r['source']),normalized(r['output']),normalized(r['reference'])
        ae,be=Levenshtein.distance(ref,a),Levenshtein.distance(ref,b)
        stats['samples']+=1;stats['improved']+=be<ae;stats['worsened']+=be>ae;stats['error_delta']+=be-ae
        stats['changed']+=a!=b;stats['model_calls']+=bool(r['candidates']);stats[r['status']]+=1
        for ent in r['entities']:
            term=normalized(ent['text']);stats['gain']+=term in b and term not in a;stats['loss']+=term in a and term not in b
        if r['edits']:changes.append(dict(**r,error_delta=be-ae))
    write_json(OUT/f'{prefix}detect_{split}{tag}_summary.json',dict(stats));write_json(OUT/f'{prefix}detect_{split}{tag}_changes.json',changes)
    print(dict(stats),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate','report','neural','neural-report','instruct','instruct-report','detect','detect-report']);p.add_argument('--split',choices=['dev','test','general','fresh','mmed_dev','mmed_test'],default='dev')
    p.add_argument('--expanded',action='store_true');p.add_argument('--url',default='http://127.0.0.1:8001');p.add_argument('--model',default='Qwen/Qwen2.5-1.5B-Instruct');p.add_argument('--tag',default='');a=p.parse_args()
    {'train':train,'evaluate':lambda:evaluate(a),'report':lambda:report(a.split,'ex_' if a.expanded else ''),
     'neural':lambda:neural(a),'neural-report':lambda:neural_report(a.split),
     'instruct':lambda:instruct(a),'instruct-report':lambda:instruct_report(a.split,'ex_' if a.expanded else '',('_'+a.tag) if a.tag else ''),
     'detect':lambda:detect(a),'detect-report':lambda:detection_report(a.split,'ex_' if a.expanded else '',('_'+a.tag) if a.tag else '')}[a.stage]()

if __name__=='__main__':main()
