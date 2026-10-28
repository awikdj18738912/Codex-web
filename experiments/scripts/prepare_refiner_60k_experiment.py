#!/usr/bin/env python3
"""Freeze a derived 12-class refiner set and a stratified audit queue.

CPU-only data preparation. This script never generates examples or trains models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from pathlib import Path

OLD = Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02')
CATEGORIES = {'A01','A02','A03','B01','B02','C01','D01','D02','E03','E05','E06','E07'}
SPLITS = ('train','dev','test')
PUBLIC_DATASETS = {'CS2W', 'FCGEC'}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f'{path}:{n}: {e}') from e


def norm(s: str) -> str:
    # Match evaluation.metric_text, with casefold as an additional guard.
    return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def fingerprint(row):
    return norm(row['input']), norm(row['reference'])


def source_name(row):
    value = row.get('dataset') or row.get('source')
    if not isinstance(value, str) or value.strip().lower() in {'unknown','none','null',''}:
        raise ValueError(f"unknown source for {row.get('id')}")
    value = value.strip()
    if value not in PUBLIC_DATASETS | {'Gemma4'}:
        raise ValueError(f"source must be one of {sorted(PUBLIC_DATASETS | {'Gemma4'})}: {value}")
    return value


def find_scene(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get('scene'), str) and obj['scene'].strip(): return obj['scene'].strip()
        for key in ('plan', 'prior_ledger_payload'):
            if key in obj:
                value=find_scene(obj[key])
                if value: return value
        for value in obj.values():
            if isinstance(value, (dict,list)):
                found=find_scene(value)
                if found: return found
    elif isinstance(obj, list):
        for value in obj:
            found=find_scene(value)
            if found: return found
    return None


def audit_flags(row):
    """Flag unexplained edits; planned repairs and filler/repetition are expected.

    The flags are screening suspicions, never a claim that a pair is wrong.
    Saved teacher approval deliberately does not bypass these checks.
    """
    source, target = row['input'], row['reference']
    category=row['category']; flags=[]
    def text(value):
        return ''.join(c for c in unicodedata.normalize('NFKC',value)
                       if not c.isspace() and unicodedata.category(c)[0] not in {'P','Z'})
    def nested(obj,key):
        if isinstance(obj,dict):
            if key in obj:return obj[key]
            for value in obj.values():
                if isinstance(value,(dict,list)):
                    result=nested(value,key)
                    if result is not None:return result
        elif isinstance(obj,list):
            for value in obj:
                result=nested(value,key)
                if result is not None:return result
        return None
    a,b=text(source),text(target)
    if category in {'B01','B02'}:
        raw=nested(row,'raw_teacher_sample') or {}
        values=[]
        for key in ('old_candidate','final_candidate'):
            if isinstance(raw.get(key),str):values.append(raw[key])
        sequence=raw.get('candidate_sequence')
        if isinstance(sequence,str):
            try:sequence=json.loads(sequence)
            except json.JSONDecodeError:sequence=[]
        if isinstance(sequence,list):values.extend(v for v in sequence if isinstance(v,str))
        # Remove only the explicit candidate values, on both sides. Remaining
        # facts must still agree after literal repair markers are removed.
        values=sorted({text(v) for v in values if v},key=len,reverse=True)
        if not values:flags.append('reversal_candidates_missing_for_fact_check')
        for value in values:
            a=a.replace(value,'');b=b.replace(value,'')
        markers=r'啊不对|哦不对|不对|哦不|啊不|不是|应该改成|应该是|还是改成|改成|改为|更正为|更正一下|说错了|口误|等一下|等会儿|等等|不[是对]'
        a=re.sub(markers,'',a);b=re.sub(markers,'',b)
    # Removing an exact adjacent repeat or an explicit filler is expected even
    # when that repeat contains a number or a negation.
    expected_fillers={'呃','嗯','额','啊','那个','这个','然后','就是','那','就是说','那个那个','嗯嗯'}
    ops=SequenceMatcher(None,a,b,autojunk=False).get_opcodes()
    changed=[(op,i,j,k,l) for op,i,j,k,l in ops if op!='equal']
    if all(op=='delete' and (a[i:j] in expected_fillers or a[max(0,i-(j-i)):i]==a[i:j] or a[j:j+(j-i)]==a[i:j]) for op,i,j,k,l in changed):
        return flags
    numeric=lambda s:re.findall(r'\d+(?:[.,:/%-]\d+)*|[零〇一二两三四五六七八九十百千万亿]+',s)
    if numeric(a)!=numeric(b):flags.append('number_or_range_change_check')
    protected=re.compile(r'没有|并非|不能|不会|只有|除非|只要|至少|至多|以上|以下|左右|以内|以外|超过|大约|不|没|未|无|仅|只')
    if protected.findall(a)!=protected.findall(b):flags.append('negation_condition_range_check')
    english=lambda s:re.findall(r'[A-Za-z][A-Za-z0-9_+/#-]*',s)
    if english(a)!=english(b):flags.append('english_or_placeholder_change_check')
    fillers={'呃','嗯','额','啊','那个','这个','然后','就是','那','就是说','那个那个','嗯嗯'}
    matcher=SequenceMatcher(None,a,b,autojunk=False)
    for tag,i,j,k,l in matcher.get_opcodes():
        if tag=='equal':continue
        before,after=a[i:j],b[k:l]
        if tag=='delete' and before in fillers:continue
        if tag=='delete' and (a[max(0,i-len(before)):i]==before or a[j:j+len(before)]==before):continue
        if category=='C01' and tag=='replace' and 1<=len(before)<=4 and 1<=len(after)<=4:continue
        if before and (len(before)>=2 or before in {'有','不','没','未','无','仅','只','都','也','还','才','再','的'}):
            flags.append('unexplained_unique_content_deletion_check')
        if after:flags.append('unexplained_addition_or_replacement_check')
    return sorted(set(flags))


def load_old():
    manifest = json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        if digest(OLD/name) != expected: raise RuntimeError(f'old frozen hash mismatch: {name}')
    rows = list(read_jsonl(OLD/'split_audit.jsonl'))
    by_id = {}
    for r in rows:
        if r['id'] in by_id: raise ValueError(f'duplicate old id {r["id"]}')
        r['source']=source_name(r) if r.get('source') or r.get('dataset') else str(r.get('source','CS2W'))
        r['scene']=find_scene(r) or 'scene_unspecified'
        by_id[r['id']] = r
    return manifest, by_id


def load_candidates(path):
    rows = list(read_jsonl(path))
    ids = set()
    for r in rows:
        for k in ('id','category','group_id','input','reference'):
            if not isinstance(r.get(k), str) or not r[k].strip(): raise ValueError(f'missing/empty {k}: {r.get("id")}')
        if r['id'] in ids: raise ValueError(f'duplicate candidate id {r["id"]}')
        ids.add(r['id'])
        if r['category'] not in CATEGORIES: raise ValueError(f'disabled/unknown category {r["category"]}: {r["id"]}')
        r['source']=source_name(r)
        r['scene']=find_scene(r) or 'scene_unspecified'
    return rows


def choose_split(key, row, occupied):
    prior = row.get('prior_frozen_split')
    if prior in ('train','dev','test','isolated_review_do_not_train'): return prior
    # Deterministic group-level 90/5/5 assignment, category and scene balanced.
    scene = str(row.get('scene') or 'scene_unspecified')
    h = int(hashlib.sha256(f'{row["category"]}\0{scene}\0{key}'.encode()).hexdigest()[:12], 16) % 1000
    proposed = 'train' if h < 900 else ('dev' if h < 950 else 'test')
    prior_splits = occupied.get(key, set())
    if prior_splits and (proposed not in prior_splits or len(prior_splits) > 1): return 'isolated_review_do_not_train'
    return proposed


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def make_audit_rows(rows, old_rows):
    out=[]
    for r in rows:
        flags = audit_flags(r)
        public = r['source'] in PUBLIC_DATASETS
        saved=r.get('audit')
        saved_pass=(isinstance(saved,dict) and all(saved.get(k) is True for k in ('source_valid','category_valid','meaning_preserved','minimal_edit','cleanup_complete')) and not saved.get('issues'))
        status=r.get('machine_review_status') or ('skipped_public_dataset' if public else ('saved_audit_pass' if saved_pass else r.get('semantic_review','saved_audit_unverified')))
        notes=[]
        nums=lambda s:re.findall(r'\d+(?:[.,:/%-]\d+)*|[零〇一二两三四五六七八九十百千万亿]+(?=(?:个|名|位|年|月|日|号|次|元|岁|公里|分钟|小时|%|十|百|千|万|亿))',s)
        if r.get('category') in {'B01','B02'} and nums(r['input'])!=nums(r['reference']) and saved_pass and re.search(r'不对|哦不|不是|应该是|应该改成|改成|改为|而是',r['input']):
            notes.append('expected_numeric_reversal_in_B_category_confirm_saved_audit')
        out.append({**r, 'audit_flags': flags, 'audit_notes':notes, 'semantic_audit': 'skipped_public_dataset' if public else 'required',
                    'human_review_status': r.get('human_review_status','pending'), 'machine_review_status': status})
    return out


def stratified_sample(rows, old_rows, n=1200):
    buckets=defaultdict(lambda: defaultdict(list))
    for r in rows:
        prior=r.get('prior_frozen_split')
        if prior == 'isolated_review_do_not_train': continue
        # Keep holdout utility visible while allowing old dev/test review.
        frozen = r['id'] in old_rows or r.get('prior_frozen_split') in (*SPLITS,'isolated_review_do_not_train')
        buckets[r['category']][('old' if frozen else 'new')].append(r)
    selected=[]
    rng=random.Random(6072026)
    def round_robin(pool, limit):
        by_scene=defaultdict(list)
        for item in pool: by_scene[item['scene']].append(item)
        for values in by_scene.values(): rng.shuffle(values)
        scenes=sorted(by_scene); out=[]
        while scenes and len(out)<limit:
            remain=[]
            for scene in scenes:
                if by_scene[scene] and len(out)<limit: out.append(by_scene[scene].pop())
                if by_scene[scene]: remain.append(scene)
            scenes=remain
        return out
    for cat in sorted(CATEGORIES):
        olds=buckets[cat]['old']; news=buckets[cat]['new']
        old=round_robin(olds,min(50,len(olds))); new=round_robin(news,min(50,len(news)))
        pool=[]
        for i in range(max(len(old),len(new))):
            if i<len(old): pool.append(old[i])
            if i<len(new): pool.append(new[i])
        if len(pool)<100:
            selected_ids={r['id'] for r in pool}
            remaining=[r for q in (olds,news) for r in q if r['id'] not in selected_ids]
            pool.extend(round_robin(remaining,100-len(pool)))
        selected.extend(pool[:100])
    # Exactly 1200 when feasible, deterministic top-up from remaining candidates.
    have={r['id'] for r in selected}
    rest=[r for r in rows if r['id'] not in have and r.get('prior_frozen_split')!='isolated_review_do_not_train']
    rng.shuffle(rest)
    selected.extend(rest[:max(0,n-len(selected))])
    return selected[:n]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--report-dir', required=True, type=Path)
    p.add_argument('--audit-results', type=Path)
    p.add_argument('--sample-only', action='store_true')
    a=p.parse_args()
    if a.output.exists() or a.report_dir.exists(): raise FileExistsError('refusing to overwrite output/report directory')
    old_manifest, old=load_old()
    rows=load_candidates(a.candidates)
    # Old IDs are immutable and retain their original split. Incoming rows that
    # duplicate an old sample are treated as same sample only if texts agree.
    for r in rows:
        if r['id'] in old:
            o=old[r['id']]
            if r['input']!=o['input'] or r['reference']!=o['reference']: raise ValueError(f'old id text changed byte-for-byte: {r["id"]}')
            r['prior_frozen_split']=o['split']
            r['split']=o['split']
            # Prefer the richer ledger payload while retaining the old split.
            old[r['id']]={**o,**r,'split':o['split'],'prior_frozen_split':o['split']}
    rows=[r for r in rows if r['id'] not in old]
    # Union-find over group_id, normalized input, and normalized reference.
    # Include frozen rows as immutable anchors; a new row in an old component
    # is review-isolated, preserving the old evaluation cohort exactly.
    old_ids=set(old)
    nodes=list(old.values())+rows
    parent=list(range(len(nodes)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        x,y=root(i),root(j)
        if x!=y: parent[y]=x
    seen={}
    for i,r in enumerate(nodes):
        keys=('g:'+r['group_id'],'t:'+norm(r['input']),'t:'+norm(r['reference']))
        for key in keys:
            if key in seen: union(i,seen[key])
            else: seen[key]=i
    components=defaultdict(list)
    for i,r in enumerate(nodes): components[root(i)].append(r)
    component_split={}
    for comp, members in components.items():
        frozen=[r for r in members if r['id'] in old_ids or r.get('prior_frozen_split') in (*SPLITS,'isolated_review_do_not_train')]
        old_splits={r.get('prior_frozen_split',r.get('split')) for r in frozen}
        if 'train' in old_splits and old_splits & {'dev','test'}:
            for r in members:
                if r.get('split',r.get('prior_frozen_split'))=='train' or r.get('prior_frozen_split')=='train':
                    r['_forced_isolated_split']=True
        if frozen:
            component_split[comp]='isolated_review_do_not_train'
        else:
            representative=min((r for r in members),key=lambda x:x['id'])
            key='|'.join(sorted({r['group_id'] for r in members}))
            component_split[comp]=choose_split(key,representative,{})
    for r in old.values():
        if r.get('_forced_isolated_split'): r['split']='isolated_review_do_not_train'
    old_train_conflicts=set()
    for members in components.values():
        anchors=[r for r in members if r['id'] in old_ids]
        if any(r['split'] in ('dev','test') for r in anchors):
            old_train_conflicts.update(r['id'] for r in anchors if r['split']=='train')
    staged=list(old.values())
    for i,r in enumerate(nodes):
        if r['id'] in old_ids: continue
        prior=r.get('prior_frozen_split')
        r['split']='isolated_review_do_not_train' if r.get('_forced_isolated_split') else (prior if prior in (*SPLITS,'isolated_review_do_not_train') else component_split[root(i)])
        staged.append(r)
    # Keep split assignments for frozen anchors unchanged.
    # The only frozen split adjustment is train-to-isolation when linked to an
    # official old dev/test component; frozen eval rows retain their exact split.
    for r in staged: r.pop('_forced_isolated_split',None)
    audit=make_audit_rows(staged,old)
    # Apply machine review decisions without ever upgrading them to human review.
    if a.audit_results:
        results={}
        for x in read_jsonl(a.audit_results):
            if not isinstance(x.get('id'),str) or not x['id']: raise ValueError('audit result missing id')
            if x['id'] in results: raise ValueError(f'duplicate audit result id {x["id"]}')
            results[x['id']]=x
        known={r['id'] for r in audit}
        if set(results)-known: raise ValueError(f'audit results contain unknown IDs: {sorted(set(results)-known)[:5]}')
        for r in audit:
            x=results.get(r['id'])
            if x:
                status=x.get('verdict',x.get('review_status',x.get('status','unresolved')))
                r['machine_review_status']=str(status)
                r['machine_review_payload']=x
                r['human_review_status']=x.get('human_review_status','pending')
                if r['human_review_status'] in ('pass','passed','approved','humanpass'):
                    raise ValueError(f'machine audit cannot certify human pass: {r["id"]}')
    a.output.mkdir(parents=True); a.report_dir.mkdir(parents=True)
    if a.sample_only:
        sample=stratified_sample(audit,old)
        for r in sample: r['sample_review_holdout']=True
        write_jsonl(a.output/'sample1200.jsonl',sample)
        # Human-readable review sheet grouped by category and scene.
        lines=['# Refiner 60k 分层抽样审核队列\n\n> 本样本是待审核抽样，不是人工 gold。旧 train 样本不构成独立测试集。\n']
        for cat in sorted(CATEGORIES):
            part=[r for r in sample if r['category']==cat]
            if not part: continue
            lines += [f'\n## {cat}\n']
            for r in sorted(part,key=lambda x:(str(x.get('scene','')),x['id'])):
                lines += [f'\n### {r["id"]} · {r.get("scene","scene_unspecified")}\n',f'**输入：** {r["input"]}\n',f'**参考：** {r["reference"]}\n',f'**来源：** {r["source"]} · 原用途 {r.get("prior_frozen_split","new candidate")}\n',f'**审核意见：** 待人工审核；启发式疑点：{", ".join(r["audit_flags"]) or "无"}\n']
        (a.report_dir/'sample1200_review.md').write_text(''.join(lines),encoding='utf-8')
        write_jsonl(a.report_dir/'review_queue.jsonl',[r for r in audit if r['audit_flags']])
        report={'status':'audit_sample_only','count':len(sample),'category_counts':dict(Counter(r['category'] for r in sample)),'scene_counts':dict(Counter(r['scene'] for r in sample)),'source_counts':dict(Counter(r['source'] for r in sample)),'warning':'Pending audit sample; no human gold certification is claimed.'}
        (a.report_dir/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False)); return
    # Reserve the same review sample in a normal freeze. New candidates in this
    # queue stay outside train/dev/test. Existing frozen train rows remain train
    # for lineage compatibility and are explicitly not an independent test set.
    sample=stratified_sample(audit,old)
    sample_ids={r['id'] for r in sample}
    for r in sample: r['sample_review_holdout']=True
    # Build train/dev/test while retaining old official eval rows verbatim.
    system=old_manifest['system_prompt']; conv={s:[] for s in SPLITS}; side=[]
    quarantine=[]
    for r in audit:
        s=r['split']
        reasons=[]
        if r['id'] in old_train_conflicts: reasons.append('old_train_text_overlaps_frozen_eval')
        if r['audit_flags']:
            # Missing historical construction metadata is a screening reason,
            # not proof that all B-class edits are bad. A strict second-pass
            # pair audit can resolve that uncertainty provisionally.
            resolved_reversal=(r['category'] in {'B01','B02'} and r['machine_review_status']=='passed_second_model_audit')
            if resolved_reversal:
                r['risk_resolution']='second_teacher_request_passed_still_pending_human_review'
            else:
                reasons.extend(r['audit_flags'])
        if r['id'] in sample_ids and r['id'] not in old:
            reasons.append('sample_review_holdout')
        if s=='train' and r['machine_review_status'] not in {'passed_second_model_audit','skipped_public_dataset','saved_audit_pass'}:
            reasons.append('machine_audit_not_approved_for_training')
        if (s=='train' or r['id'] not in old) and r['machine_review_status'] not in {'passed_second_model_audit','skipped_public_dataset','saved_audit_pass'}:
            reasons.append('machine_review_unresolved_or_failed')
        if (s=='train' or r['id'] not in old) and reasons:
            quarantine.append({**r,'quarantine_reasons':sorted(set(reasons)),'retained_official_eval':False})
            continue
        if r['id'] in sample_ids and r['id'] not in old:
            quarantine.append({**r,'quarantine_reasons':['sample_review_holdout'],'retained_official_eval':False})
            continue
        if s not in SPLITS:
            if s=='isolated_review_do_not_train': quarantine.append({**r,'quarantine_reasons':['split_leakage_or_conflicting_frozen_anchor'],'retained_official_eval':False})
            continue
        # Historical dev/test rows remain in the official cohort even if risk flags exist.
        conv[s].append({'messages':[{'role':'system','content':system},{'role':'user','content':r['input']},{'role':'assistant','content':r['reference']}]})
        # Preserve the candidate's original payload, saved audit and provenance
        # in the audit ledger, including rows that pass without review flags.
        side.append(dict(r))
    for s in SPLITS: (a.output/f'{s}.json').write_text(json.dumps(conv[s],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_jsonl(a.output/'sample1200.jsonl',sample)
    write_jsonl(a.output/'split_audit.jsonl',side)
    info={f'refiner60k_{s}':{'file_name':f'{s}.json','formatting':'sharegpt','columns':{'messages':'messages'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant','system_tag':'system'}} for s in ('train','dev')}
    (a.output/'dataset_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_jsonl(a.report_dir/'review_queue.jsonl',[r for r in audit if r['audit_flags']])
    write_jsonl(a.output/'quarantine.jsonl',quarantine)
    # Leakage assertion applies to the emitted train/eval sets.
    train_text={norm(x['input']) for x in side if x['split']=='train'} | {norm(x['reference']) for x in side if x['split']=='train'}
    eval_text={norm(x[k]) for x in side if x['split'] in ('dev','test') for k in ('input','reference')}
    overlap=train_text & eval_text
    if overlap: raise RuntimeError(f'train/eval text leakage after frozen-anchor handling: {len(overlap)} normalized texts')
    manifest={'status':'candidate_data_frozen_pending_human_audit','source_candidates':str(a.candidates),'source_candidates_sha256':digest(a.candidates),'source_old_dataset':str(OLD),'source_old_manifest_sha256':digest(OLD/'manifest.json'),'system_prompt':system,'split_counts':{s:len(conv[s]) for s in SPLITS}|{'isolated_review_do_not_train':sum(x['split']=='isolated_review_do_not_train' for x in audit)},'sample_review_holdout_count':len(sample),'quarantine_count':len(quarantine),'scene_counts':dict(Counter(x['scene'] for x in audit)),'source_counts':dict(Counter(x['source'] for x in audit)),'class_counts':{s:dict(Counter(x['category'] for x in side if x['split']==s)) for s in SPLITS}|{'isolated_review_do_not_train':dict(Counter(x['category'] for x in audit if x['split']=='isolated_review_do_not_train'))},'files_sha256':{n:digest(a.output/n) for n in ('train.json','dev.json','test.json','split_audit.jsonl','dataset_info.json','sample1200.jsonl','quarantine.jsonl')},'warning':'Candidate rows are pending audit; no human gold certification is claimed.'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__': main()
