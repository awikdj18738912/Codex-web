"""Second, frozen entity-only comparison. No gold text enters a rule."""
from sweep_entity_rules import *
from system.entity_matcher import EntityDecision
from system.entity_boundaries import latin_boundaries_valid
from system.entity_boundaries import _tokens
from system.entity_pronunciation import _context_terms
import hashlib

EXTRA=['alias_nested_guarded','alias_device_boundary','sound_distance2_relaxed','component_bridge']


def alias_v2(text,ds,device_boundary=False):
    namespace={}
    for d in ds:
        if not d.enabled:continue
        for w in (d.canonical_text,*d.aliases):
            if re.fullmatch(r'[\u3400-\u9fff]{3,16}',w):namespace.setdefault(pronunciation_signature(w),set()).add(d.canonical_text)
    blocked=EntityProtector(ds,protect_numeric_spans=False).protect(text).spans
    edits=[]
    for d in ds:
        if d.domain!='medical' or d.entity_type!='TERM' or d.normalization_policy!='normalize' or not d.enabled:continue
        for a in d.aliases:
            if not re.fullmatch(r'[\u3400-\u9fff]{3,16}',a):continue
            target=pronunciation_signature(a)
            if namespace.get(target)!={d.canonical_text}:continue
            for run in re.finditer(r'[\u3400-\u9fff]+',text):
                for offset in range(len(run.group())-len(a)+1):
                    start=run.start()+offset;end=start+len(a);s=text[start:end]
                    if s==a or pronunciation_signature(s)!=target:continue
                    if any(start<p.end and p.start<end for p in blocked):continue
                    left=text[max(0,start-16):start];right=text[end:end+16]
                    if not re.search(r'(台|套|使用|启动|连接|上|做|接受|进行)(了|过)?(这个|这|一)?$',left):continue
                    if medical_person_slot(text,start,end) or re.match(r'的?(医生|医师|护士|主任|教授|先生|女士)',right):continue
                    if any(w in left+right for w in ('作者','签名','书信','礼物','小说','手稿','写法','拼写','读作','写成','这个词')):continue
                    before=text[start-1] if start else '';after=text[end] if end<len(text) else ''
                    if before in {'@','/','\\','_'} or after in {'/','\\','_'} or any(c.isascii() and c.isalnum() for c in (before,after)):continue
                    if text[:start].rfind('《')>text[:start].rfind('》'):continue
                    issue=candidate_boundary_issue(text,start,end)
                    if issue:
                        # A known alias in a device quantity slot can end before
                        # a negative predicate even when jieba joins its last
                        # character with 不. Do not relax other word boundaries.
                        if not (device_boundary and issue=='chinese_word_boundary_conflict'
                                and re.search(r'(台|套)$',left) and re.match(r'不(但|是|能|会|用|够|足|需要)',right)
                                and not any(a<start<b for _,a,b in _tokens(text))
                                and all(word==s[-1]+'不' and a==end-1 and b==end+1
                                        for word,a,b in _tokens(text) if a<end<b)):
                            continue
                    edits.append(dict(start=start,end=end,observed=s,canonical=d.canonical_text,rule='alias_device_boundary' if device_boundary else 'alias_nested_guarded'))
    # Prefer a longer alias only when it literally preserves the shorter
    # normalization plus the unchanged source suffix. Conflicting targets abstain.
    edits=[e for e in edits if not any(r['start']==e['start'] and r['end']>e['end']
            and r['canonical']==e['canonical']+text[e['end']:r['end']] for r in edits)]
    return render(text,edits)


def relaxed_sound(text,ds):
    edits=[];blocked=EntityProtector(ds,protect_numeric_spans=False).protect(text).spans
    for d in ds:
        c=d.canonical_text;p=d.spelling_context
        if not p or not re.fullmatch(r'[\u3400-\u9fff]{4,16}',c):continue
        target=' '.join(pronunciation_signature(c))
        for run in re.finditer(r'[\u3400-\u9fff]+',text):
            for offset in range(len(run.group())-len(c)+1):
                start=run.start()+offset;end=start+len(c);s=text[start:end]
                if s==c or sum(a!=b for a,b in zip(s,c))>3:continue
                if any(start<p.end and p.start<end for p in blocked):continue
                if candidate_boundary_issue(text,start,end) or medical_person_slot(text,start,end):continue
                if Levenshtein.distance(' '.join(pronunciation_signature(s)),target)>2:continue
                if not _context_terms(text,start,end,p):continue
                edits.append(dict(start=start,end=end,observed=s,canonical=c,rule='sound_distance2_relaxed'))
    return render(text,edits)


def bridge(text,ds):
    protector=EntityProtector(ds,protect_numeric_spans=False)
    protected=protector.protect(text).spans
    matcher=EntityCandidateMatcher(ds,selected_domain='medical',mode='auto')
    report=matcher.match(text,allow_auto=True,blocked_spans=[(p.start,p.end) for p in protected if p.entity_id is None])
    edits=[]
    for m in report.auto_matches:
        if m.match_type not in {'CONTEXT_PINYIN','CONTEXT_NEAR_PINYIN'} or len(m.observed)!=len(m.canonical):continue
        if m.observed[-1] in '正在了的和与或也是':continue
        overlaps=[p for p in protected if m.start<p.end and p.start<m.end]
        if not overlaps or any(not(m.start<=p.start and p.end<=m.end)
            or m.canonical[p.start-m.start:p.end-m.start]!=p.replacement for p in overlaps):continue
        edits.append(dict(start=m.start,end=m.end,observed=m.observed,canonical=m.canonical,rule='component_bridge'))
    return render(text,edits)


def evaluate(rows,label):
    ds=definitions();ads=definitions(True)
    tools={k:(EntityProtector(d,protect_numeric_spans=False),EntityCandidateMatcher(d,selected_domain='medical',mode='auto'))
           for k,d in [('normal',ds),('aliases',ads)]}
    variants=VARIANTS+EXTRA;counts={v:Counter() for v in variants};changes={v:[] for v in variants};records=[]
    for index,r in enumerate(rows):
        baseline=prepare_entity_segment(r['raw_text'],*tools['normal'],allow_auto=True).baseline_text
        aliases=prepare_entity_segment(r['raw_text'],*tools['aliases'],allow_auto=True).baseline_text
        ref=normalized(r['reference']);be=Levenshtein.distance(ref,normalized(baseline));outputs={}
        for v in variants:
            output=baseline;edits=[]
            if v.startswith('official_alias'):
                output=aliases
                if v=='official_alias_sound':output,edits=alias_sound(output,ads)
            elif v.startswith('alias_'):output,edits=alias_v2(aliases,ads,v=='alias_device_boundary')
            elif v=='sound_distance2_relaxed':output,edits=relaxed_sound(output,ds)
            elif v=='component_bridge':output,edits=bridge(output,ds)
            elif v!='current':output,edits=near(output,ds,v)
            error=Levenshtein.distance(ref,normalized(output));s=counts[v]
            s.update(samples=1,changed=int(output!=baseline),improved=int(error<be),regressed=int(error>be),error_delta=error-be)
            outputs[v]=output
            if output!=baseline:changes[v].append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],baseline=baseline,output=output,edits=edits,error_delta=error-be))
        records.append(dict(id=r['id'],variant=r['variant'],reference=r['reference'],raw_text=r['raw_text'],outputs=outputs))
        if index%100==0:print(label,index+1,len(rows),flush=True)
    write_json(OUT/f'{label}_v2.json',dict(counts={v:dict(s) for v,s in counts.items()},changes=changes))
    (OUT/f'{label}_outputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    print(json.dumps({v:dict(s) for v,s in counts.items()},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('label',choices=['development','holdout']);args=p.parse_args()
    if args.label=='development':
        rows=read_jsonl(OLD/'asr.jsonl')
        controls=json.loads((ROOT/'results/medical_entity_20261003/source/CMeEE_train.json').read_text());random.Random(20261008).shuffle(controls)
        rows += [dict(id=f'correct_medical_{i}',variant='original',reference=r['text'],raw_text=r['text']) for i,r in enumerate(controls[:1000])]
    else:rows=read_jsonl(OUT/'holdout/asr.jsonl')
    evaluate(rows,args.label)
