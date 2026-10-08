"""Isolated, paired entity rollout experiments; never edits the live database.

Candidate construction sees only ASR hypotheses and frozen train dictionaries.
References are accessed only by the evaluator, after model requests are built.
No Web, ASR, numeric normalization, or cleanup implementation is replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from evaluate_entity_library import ROOT, append_jsonl, normalized, read_jsonl, write_json
from rapidfuzz.distance import Levenshtein
from system.entity_boundaries import latin_boundaries_valid
from system.entity_matcher import (
    EntityCandidateMatcher, _bigrams, _length_compatible,
    _normalize_with_offsets, _pinyin_tokens, _possible_window_lengths, _text_runs,
)
from system.entity_pipeline import prepare_entity_segment, finalize_entity_segment
from system.entity_store import EntityStore
from system.protection import EntityProtector
from system.refinement_protocol import apply_structured_patch, NUMERIC_RULES_REFINER_SYSTEM_PROMPT

OUT = ROOT / 'results/entity_rollout_20261005'
MED = ROOT / 'results/medical_entity_20261003'
GENERAL = ROOT / 'results/entity_library_20261003'
TYPE_DESCRIPTIONS = {
    'dis': '疾病或诊断', 'sym': '症状或体征', 'bod': '身体部位或组织',
    'dru': '药物', 'pro': '检查或治疗操作', 'equ': '医疗器械',
    'ite': '医学检验项目', 'dep': '科室', 'mic': '微生物',
    'PERSON': '人名', 'ORG': '机构或组织名称', 'LOC': '地名', 'TERM': '领域术语',
}


def load_sets():
    medical_types = {v['canonical_text']: v.get('cmeee_type', 'TERM')
                     for v in json.loads((MED/'vocabulary.json').read_text())}
    return [
        ('medical', read_jsonl(MED/'asr.jsonl'), MED/'medical.db', medical_types),
        ('general', read_jsonl(GENERAL/'asr.jsonl'), GENERAL/'typed.db', {}),
    ]


def retrieve(matcher, text, blocked, limit=8):
    """Keep up to three plausible entities per span, without declaring them true.

    The score threshold stays at the existing 0.8. Chinese segmentation gates
    are intentionally not used for recall: they remain reasons for caution.
    Latin-word and punctuation boundaries are still mandatory.
    """
    groups = defaultdict(dict)
    lengths = _possible_window_lengths(matcher._lengths)
    for run_start, run in _text_runs(text):
        folded, offsets = _normalize_with_offsets(run, run_start)
        pinyin = _pinyin_tokens(folded)
        for i in range(len(folded)):
            for length in lengths:
                j = i + length
                if j > len(folded):
                    continue
                start, end = offsets[i], offsets[j-1]+1
                if any(start < b and a < end for a,b in blocked):
                    continue
                observed = text[start:end]
                if not latin_boundaries_valid(text,start,end) or re.search(r'[，,；;：:]',observed):
                    continue
                indexes = set()
                for gram in _bigrams(folded[i:j]):
                    indexes.update(matcher._char_index.get(gram,()))
                for gram in _bigrams(pinyin[i:j]):
                    indexes.update(matcher._pinyin_index.get(gram,()))
                for index in indexes:
                    surface = matcher._surfaces[index]
                    if observed.casefold() == surface.value.casefold():
                        continue
                    if not _length_compatible(length,len(surface.normalized)):
                        continue
                    scored = matcher._score(folded[i:j],pinyin[i:j],surface)
                    if scored.final_score < matcher.config.hint_score:
                        continue
                    entity = surface.definition
                    if entity.canonical_text == observed:
                        continue
                    candidate = dict(entity_id=entity.entity_id,canonical=entity.canonical_text,
                                     type=entity.entity_type,score=scored.final_score)
                    old = groups[(start,end)].get(entity.entity_id)
                    if old is None or candidate['score'] > old['score']:
                        groups[(start,end)][entity.entity_id] = candidate
    ranked_groups = []
    for (start,end), candidates in groups.items():
        ranked = sorted(candidates.values(),key=lambda c:(-c['score'],c['entity_id']))[:3]
        ranked_groups.append(dict(start=start,end=end,observed=text[start:end],candidates=ranked))
    ranked_groups.sort(key=lambda g:(-g['candidates'][0]['score'],-(g['end']-g['start']),g['start']))
    selected = []
    for group in ranked_groups:
        if any(group['start'] < g['end'] and g['start'] < group['end'] for g in selected):
            continue
        selected.append(group)
        if len(selected) == limit:
            break
    return sorted(selected,key=lambda g:g['start'])


def prepare():
    destination = OUT/'prepared.jsonl'
    if destination.exists():
        raise ValueError('Refusing to overwrite frozen model inputs')
    summary = {}
    for domain,rows,db,kinds in load_sets():
        definitions = EntityStore(db).list_entities(domain=domain)
        matcher = EntityCandidateMatcher(definitions,selected_domain=domain,mode='auto')
        protector = EntityProtector(definitions)
        stats = Counter()
        for row in rows:
            prepared = prepare_entity_segment(row['raw_text'],protector,matcher,allow_auto=True)
            groups = retrieve(matcher,row['raw_text'],[(s.start,s.end) for s in prepared.protection.spans])
            for group in groups:
                for i,c in enumerate(group['candidates']):
                    c['choice'] = f"{group['start']}:{group['end']}:{i}"
                    kind = kinds.get(c['canonical'],c['type'])
                    c['description'] = TYPE_DESCRIPTIONS.get(kind,'医学专业术语')
            # Requests have no reference fields; this manifest can be audited.
            request = dict(domain=domain,source=row['raw_text'],groups=groups)
            record = dict(id=row['id'],variant=row['variant'],domain=domain,
                          source=row['raw_text'],reference=row['reference'],entities=row['entities'],
                          baseline=prepared.baseline_text,current_candidates=prepared.refiner_candidates,
                          groups=groups,request_sha256=hashlib.sha256(json.dumps(request,ensure_ascii=False,sort_keys=True).encode()).hexdigest())
            append_jsonl(destination,record)
            stats['samples'] += 1
            stats['current_candidate_samples'] += bool(prepared.refiner_candidates)
            stats['retrieval_candidate_samples'] += bool(groups)
            expected = [e['text'] for e in row['entities'] if e.get('covered',True)]
            for term in expected:
                if normalized(term) in normalized(row['raw_text']):
                    continue
                stats['missing_entity_occurrences'] += 1
                stats['current_recalled'] += any(c['canonical']==term for c in prepared.refiner_candidates)
                stats['retrieval_recalled'] += any(c['canonical']==term for g in groups for c in g['candidates'])
        summary[domain] = dict(stats)
    write_json(OUT/'retrieval_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


RESOLVER_PROMPT = '''你是语音转写的实体核对器。输入是未必正确的转写和检索候选。
候选只是音近或字近词，不能因为它在词库中就替换。结合完整句子和候选类别判断。
只允许选择给定 choice，不能润色其他内容。保留正常语法、普通词和原有事实；
不把“手术失败”“手术是”“骨碎了”等普通表达拼成术语；人名没有充分证据时保持原文。
如果有歧义、候选不符合句意、原文已经正确，返回 {"action":"keep"}。
只有明确属于同一实体的识别错误，返回 {"action":"replace","choice":"候选choice"}。
只输出一个JSON对象，每次最多修正一处。'''


def model_experiment(args):
    from system.live_qwen_refiner import TransformersRefiner
    model = TransformersRefiner(Path('/home/aim0/data/models/ASR/AgenticASR-Refiner'),'cuda:0',128)
    sets = {d:(EntityProtector(EntityStore(db).list_entities(domain=d)),
               EntityCandidateMatcher(EntityStore(db).list_entities(domain=d),selected_domain=d,mode='auto'))
            for d,_,db,_ in load_sets()}
    suffix = '' if args.protocol=='choice' else '_'+args.protocol
    destination = OUT/f'model{suffix}.jsonl'
    notes = json.loads((OUT/'network_notes.json').read_text())['entities'] if args.protocol=='network' else {}
    complete = {(r['id'],r['variant']) for r in read_jsonl(destination)}
    rows = read_jsonl(OUT/'prepared.jsonl')
    for index,row in enumerate(rows):
        if (row['id'],row['variant']) in complete:
            continue
        protector,matcher = sets[row['domain']]
        prepared = prepare_entity_segment(row['source'],protector,matcher,allow_auto=True)
        current = prepared.baseline_text
        baseline_response = None
        baseline_ms = 0
        if prepared.refiner_candidates:
            baseline_response,baseline_ms = model.refine_with_numeric_rules(
                prepared.protection.masked_text,entity_hints=prepared.hints,
                entity_candidates=prepared.refiner_candidates)
            patched,patch,error = apply_structured_patch(prepared.protection.masked_text,baseline_response)
            if error is None:
                final = finalize_entity_segment(patched,prepared,protector,entity_patch=patch)
                current = final.text
        proposed = prepared.baseline_text
        response = None
        model_ms = 0
        status = 'no_candidates'
        if row['groups']:
            request = dict(domain=row['domain'],source=row['source'],groups=row['groups'])
            if args.protocol in {'patch','network'}:
                # Match the checkpoint's familiar raw-ASR + KEY input format.
                request = row['source'] + '\n<ENTITY_CANDIDATES>' + json.dumps([
                    dict(observed=g['observed'],canonical=c['canonical'],type=c['description'],
                         **notes.get(c['canonical'],{}))
                    for g in row['groups'] for c in g['candidates']],ensure_ascii=False)
                prompt = NUMERIC_RULES_REFINER_SYSTEM_PROMPT + (
                    '本次只核对实体，禁止修改口癖、重复、标点和数字。候选未被确认。'
                    '仅上下文明确支持时，返回单个局部JSON补丁，source等于observed，'
                    'target等于canonical，reason为entity_correction。歧义时keep。')
            else:
                request = json.dumps(request,ensure_ascii=False)
                prompt = RESOLVER_PROMPT
            response,model_ms = model._refine_with_system_prompt(
                request,prompt,entity_hints=(),strict_placeholders=False)
            try:
                patch = json.loads(response)
                if not isinstance(patch,dict):
                    raise ValueError('expected object')
                if patch.get('action') == 'keep':
                    status = 'keep'
                elif patch.get('action') == 'replace':
                    options = [(g,c) for g in row['groups'] for c in g['candidates']
                               if (c['choice']==patch.get('choice') if args.protocol=='choice'
                                   else g['observed']==patch.get('source') and c['canonical']==patch.get('target'))]
                    if len(options) != 1:
                        raise ValueError('unlisted choice')
                    g,c = options[0]
                    # Raw offsets refer to raw text; preserve verified alias edits.
                    # Multiple equal surfaces need a full offset map: abstain for now.
                    if proposed.count(g['observed']) != 1:
                        raise ValueError('ambiguous occurrence')
                    proposed = proposed.replace(g['observed'],c['canonical'],1)
                    status = 'replace'
                else:
                    raise ValueError('unsupported action')
            except (ValueError,TypeError,KeyError):
                status = 'invalid_fallback'
        append_jsonl(destination,dict(id=row['id'],variant=row['variant'],domain=row['domain'],
                     current=current,proposed=proposed,status=status,response=response,
                     baseline_response=baseline_response,baseline_ms=baseline_ms,model_ms=model_ms))
        if row['groups'] or index % 50 == 0:
            print(index+1,'/',len(rows),row['id'],row['variant'],status,response,flush=True)
    report_model(args.protocol)


def report_model(protocol='choice'):
    prepared = {(r['id'],r['variant']):r for r in read_jsonl(OUT/'prepared.jsonl')}
    summary = {}
    changed = []
    for domain in ['medical','general']:
        suffix = '' if protocol=='choice' else '_'+protocol
        rows = [r for r in read_jsonl(OUT/f'model{suffix}.jsonl') if r['domain']==domain]
        stats = Counter()
        for row in rows:
            p = prepared[(row['id'],row['variant'])]
            ref = normalized(p['reference'])
            a,b = normalized(row['current']),normalized(row['proposed'])
            stats['samples'] += 1
            stats['current_errors'] += Levenshtein.distance(ref,a)
            stats['proposed_errors'] += Levenshtein.distance(ref,b)
            stats['reference_chars'] += len(ref)
            stats['improved'] += Levenshtein.distance(ref,b) < Levenshtein.distance(ref,a)
            stats['worsened'] += Levenshtein.distance(ref,b) > Levenshtein.distance(ref,a)
            stats[row['status']] += 1
            stats['changed_samples'] += a != b
            for entity in p['entities']:
                term = normalized(entity['text'])
                stats['expected_entities'] += 1
                stats['current_recovered'] += term in a
                stats['proposed_recovered'] += term in b
                stats['entity_losses'] += term in a and term not in b
            if a!=b:
                changed.append(dict(**p,model=row))
        stats['retention_gate_passed'] = int(stats['proposed_recovered'] > stats['current_recovered']
                                               and stats['proposed_errors'] < stats['current_errors']
                                               and stats['worsened']==0 and stats['entity_losses']==0)
        summary[domain] = dict(stats)
    suffix = '' if protocol=='choice' else '_'+protocol
    write_json(OUT/f'model{suffix}_summary.json',summary)
    write_json(OUT/f'model{suffix}_changes.json',changed)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','model','report'])
    parser.add_argument('--protocol',choices=['choice','patch','network'],default='choice')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'model':lambda:model_experiment(args),'report':lambda:report_model(args.protocol)}[args.stage]()


if __name__=='__main__':
    main()
