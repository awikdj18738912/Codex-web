"""Paired entity-context and selective rerecognition pilot on real audio.

All three arms decode exactly the same PCM. Static context uses a frozen
training glossary; dynamic context reads the newly decoded baseline ASR only.
Neither prompt builder has access to reference transcripts or annotations.
This is an isolated pilot, not a modification to the running Web service.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import re
import time
from types import SimpleNamespace

import httpx
from rapidfuzz.distance import JaroWinkler, Levenshtein
from evaluate_entity_library import ROOT, read_jsonl, append_jsonl, write_json, audio_bytes, normalized
from system.entity_asr_context import build_entity_asr_context, context_digest
from system.entity_boundaries import latin_boundaries_valid
from system.entity_matcher import _normalize, _normalize_with_offsets, _pinyin_tokens, _text_runs
from system.entity_store import EntityStore

OUT = ROOT/'results/entity_rollout_20261005'
MED = ROOT/'results/medical_entity_20261003'
OLD_REGRESSIONS = {f'MultiMed_Chinese_test_{n:04d}' for n in [13,15,48,101,118,185,216]}


def choose_audio():
    """Stratification may use coverage labels; prompts never use them."""
    medical = read_jsonl(MED/'asr.jsonl')
    originals = {r['id']:r for r in medical if r['variant']=='clean' and r['id'] not in OLD_REGRESSIONS}
    rng = random.Random(20261005)
    chosen = []
    for has_entity in [True,False]:
        pool = sorted(k for k,r in originals.items() if bool(r['entities'])==has_entity)
        rng.shuffle(pool)
        chosen.extend(pool[:16])
    rows = [dict(r,domain='medical') for r in medical if r['id'] in chosen]
    general = read_jsonl(ROOT/'results/entity_library_20261003/asr.jsonl')
    pool = sorted({r['id'] for r in general if not r['entities']})
    rng.shuffle(pool)
    rows += [dict(r,domain='general_control') for r in general if r['id'] in pool[:8]]
    return rows


def dynamic_terms(text, definitions):
    """Experimental soft phonetic shortlist: at most eight canonical names.

    Unlike automatic textual replacement, this can consider two-character
    terms. Approximate sound is a retrieval clue, never a truth label.
    """
    runs = []
    for start,run in _text_runs(text):
        folded,offsets = _normalize_with_offsets(run,start)
        runs.append((folded,offsets,_pinyin_tokens(folded)))
    ranked = []
    for entity in definitions:
        word = _normalize(entity.canonical_text)
        if len(word)<2 or len(word)>16 or (word.isascii() and len(word)<3):
            continue
        target = ''.join(_pinyin_tokens(word))
        best = 0.0
        for folded,offsets,pinyin in runs:
            for length in range(max(2,len(word)-1),len(word)+2):
                for i in range(len(folded)-length+1):
                    j = i+length
                    a,b = offsets[i],offsets[j-1]+1
                    if not latin_boundaries_valid(text,a,b) or re.search(r'[，,；;：:]',text[a:b]):
                        continue
                    surface = folded[i:j]
                    if word == surface:
                        score = 1.0
                    else:
                        sound = JaroWinkler.normalized_similarity(target,''.join(pinyin[i:j]))
                        if sound<0.85:
                            continue
                        score = 0.8*sound+0.2*Levenshtein.normalized_similarity(word,surface)
                    best = max(best,score)
        if best>=0.8:
            ranked.append((best,entity.canonical_text,entity.entity_id))
    return [r[1] for r in sorted(ranked,key=lambda r:(-r[0],r[2]))[:8]]


def context_for_terms(terms):
    if not terms:
        return ''
    return '参考术语仅用于辅助拼写，不代表音频内容；未在音频中听到的词不要补充。术语列表：'+json.dumps(terms,ensure_ascii=False)


def decode(client,pcm,context):
    params = dict(language='Chinese',chunk_seconds=0.5)
    if context:
        params['context'] = context
    ready = client.post('/stream/start',params=params).raise_for_status().json()
    if ready.get('context_applied') != bool(context):
        raise ValueError('Context not applied as requested')
    if context and ready.get('context_digest') != context_digest(context):
        raise ValueError('Context digest mismatch')
    sid = ready['session_id']
    try:
        for offset in range(0,len(pcm),64000):
            client.post('/stream/chunk',params=dict(session_id=sid),content=pcm[offset:offset+64000]).raise_for_status()
        final = client.post('/stream/finish',params=dict(session_id=sid)).raise_for_status().json()
    except BaseException:
        client.post('/stream/cancel',params=dict(session_id=sid))
        raise
    return ready,final


def summarize():
    rows = read_jsonl(OUT/'audio.jsonl')
    lookup = {(r['id'],r['variant'],r['arm']):r for r in rows}
    summaries = {}
    changes = []
    for domain in ['medical','general_control']:
        summaries[domain] = {}
        for arm in ['off','static','dynamic','repeat_off']:
            stats = Counter()
            for r in rows:
                if r['domain']!=domain or r['arm']!=arm:
                    continue
                base = lookup[(r['id'],r['variant'],'off')]
                a,b,ref = normalized(base['text']),normalized(r['text']),normalized(r['reference'])
                ae,be = Levenshtein.distance(ref,a),Levenshtein.distance(ref,b)
                stats['samples'] += 1
                stats['errors'] += be
                stats['reference_chars'] += len(ref)
                stats['improved'] += be<ae
                stats['worsened'] += be>ae
                stats['changed_samples'] += a!=b
                stats['actual_context_decodes'] += bool(r['context'])
                for e in r['entities']:
                    term = normalized(e['text'])
                    stats['expected_entities'] += 1
                    stats['recovered'] += term in b
                    stats['entity_gains'] += term in b and term not in a
                    stats['entity_losses'] += term in a and term not in b
                if a!=b:
                    changes.append(dict(id=r['id'],variant=r['variant'],domain=domain,arm=arm,
                                        baseline=base['text'],proposed=r['text'],reference=r['reference'],
                                        terms=r['terms'],error_delta=be-ae))
            stats['retention_gate_passed'] = int(stats['entity_gains']>0 and stats['entity_losses']==0
                                                   and stats['worsened']==0)
            summaries[domain][arm] = dict(stats)
    write_json(OUT/'audio_summary.json',summaries)
    write_json(OUT/'audio_changes.json',changes)
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--repeat-off',action='store_true',help='Repeat the no-context control after all paired arms')
    args = parser.parse_args()
    if args.report:
        summarize()
        return
    definitions = EntityStore(MED/'medical.db').list_entities(domain='medical')
    static = build_entity_asr_context(definitions,'medical')
    rows = choose_audio()
    protocol_path = OUT/('audio_repeat_protocol.json' if args.repeat_off else 'audio_protocol.json')
    write_json(protocol_path,dict(
        seed=20261005,noise_snr_db=10.0,medical_originals=32,ordinary_originals=8,
        excluded_prior_regressions=sorted(OLD_REGRESSIONS),static_terms=static.terms,
        dictionary_source='Frozen CMeEE + MultiMed train vocabulary, no new test aliases',
        dynamic_input='Fresh off-arm ASR only; references never passed to prompt builder',
        retention='Positive entity gain, zero entity losses and zero worsened samples; otherwise reject',
        sample_ids=[dict(id=r['id'],variant=r['variant'],domain=r['domain']) for r in rows]))
    destination = OUT/'audio.jsonl'
    complete = {(r['id'],r['variant'],r['arm']):r for r in read_jsonl(destination)}
    with httpx.Client(base_url='http://127.0.0.1:8766',timeout=180) as client:
        health = client.get('/health').raise_for_status().json()
        if not health.get('context_supported'):
            raise ValueError('context unavailable')
        write_json(OUT/'audio_health.json',health)
        for index,row in enumerate(rows):
            pcm,duration = audio_bytes(row,row['variant'],SimpleNamespace(seed=20261003,noise_snr=10.0))
            if hashlib.sha256(pcm).hexdigest()!=row['pcm_sha256']:
                raise ValueError('PCM differs from the original experiment')
            dynamic = None
            for arm in (['repeat_off'] if args.repeat_off else ['off','static','dynamic']):
                key = (row['id'],row['variant'],arm)
                if key in complete:
                    if arm=='off':
                        dynamic = dynamic_terms(complete[key]['text'],definitions)
                    continue
                terms = [] if arm in {'off','repeat_off'} else list(static.terms) if arm=='static' else dynamic
                context = context_for_terms(terms)
                started = time.perf_counter()
                ready,final = decode(client,pcm,context)
                record = dict(id=row['id'],variant=row['variant'],domain=row['domain'],arm=arm,
                              text=final['text'],reference=row['reference'],entities=row['entities'],
                              terms=terms,context=context,context_digest=context_digest(context),
                              pcm_sha256=row['pcm_sha256'],duration_seconds=duration,
                              elapsed_seconds=time.perf_counter()-started,ready=ready,result=final)
                append_jsonl(destination,record)
                complete[key] = record
                if arm=='off':
                    dynamic = dynamic_terms(final['text'],definitions)
                print(index+1,'/',len(rows),row['id'],row['variant'],arm,terms,final['text'],flush=True)
    summarize()


if __name__=='__main__':
    main()
