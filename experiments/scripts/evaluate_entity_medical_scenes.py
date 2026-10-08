"""Frozen specialty glossaries on new public human medical recordings.

Only the existing entity layer and existing optional ASR context endpoint are
exercised. This script never writes the live entity database or imports a
failed experimental verifier into production. References are used for sample
stratification and scoring, never passed to decoding or correction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import random
import re
import unicodedata
from types import SimpleNamespace
import time

import httpx
from rapidfuzz.distance import Levenshtein

from evaluate_entity_library import ROOT, append_jsonl, audio_bytes, normalized, read_jsonl, write_json
from evaluate_medical_entities import lexicon_occurrences
from system.entity_asr_context import build_entity_asr_context
from system.entity_boundaries import candidate_boundary_issue, latin_boundaries_valid, _tokenizer
from system.entity_pronunciation import pronunciation_signature
from system.entity_retrieval import sound_similarity
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.entity_store import EntityStore
from system.protection import EntityProtector
from test_entity_audio_rollout import decode as decode_pcm

OUT = ROOT / 'results/entity_medical_scenes_20261006'
MED = ROOT / 'results/medical_entity_20261003'
SEED = 20261007
# Declared from the specialties, before any ASR in this pilot. No misspelling
# aliases, frequency ranking from the evaluated captions, or inferred drugs.
SCENES = {
    'cardiopulmonary': {
        'label': '心肺手术', 'ranges': [(0, 63)],
        'terms': ['TAVI', 'ENB', 'TEE', '经导管主动脉瓣植入术', '主动脉瓣狭窄',
                  '主动脉瓣关闭不全', '主动脉瓣', '电磁导航支气管镜', '肺结节',
                  '鳞状细胞癌', '体外循环', '全身麻醉', '开胸换瓣手术',
                  '心瓣叶', '股动脉', '支气管镜'],
        'anchors': ['心脏', '手术', '瓣膜', '导管', '肺部', '医生'],
    },
    'orthopedics': {
        'label': '骨科与骨质疏松', 'ranges': [(63, 129), (566, 683)],
        'terms': ['骨质疏松症', '骨质疏松', '椎间盘突出', '颈椎间盘',
                  '人工椎间盘', '颈椎病', '关节置换术', '椎间盘', '神经系统',
                  '脊髓', '肌腱', '韧带', '类风湿性关节炎', '髋关节',
                  '骨密度', '维生素D'],
        'anchors': ['骨头', '骨折', '补钙', '手术', '医生', '疼痛', '检查'],
    },
    'urology': {
        'label': '泌尿科与前列腺检查', 'ranges': [(749, 846)],
        'terms': ['前列腺癌', '前列腺', '前列腺增生', '良性前列腺增生',
                  '前列腺特异性抗原', 'PSA', '直肠指检', '活体组织检验',
                  '活检', '腹腔镜', '泌尿科', '尿道', '精囊', '直肠',
                  '淋巴结', '机器人辅助手术'],
        'anchors': ['患者', '医生', '抽血', '检查', '小便', '男性', '癌症'],
    },
}


def occurrences(text, terms):
    """Exclude English acronym matches inside longer ASCII words."""
    entries = lexicon_occurrences(text, terms)
    compact = normalized(text)
    return [e for e in entries if not (
        e['text'].isascii() and (
            e['start'] and compact[e['start'] - 1].isascii() and compact[e['start'] - 1].isalnum()
            or e['end'] < len(compact) and compact[e['end']].isascii() and compact[e['end']].isalnum()
        )
    )]


def prepare(holdout=False):
    import numpy as np
    import pyarrow.parquet as pq
    from scipy.signal import resample_poly
    import soundfile as sf
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection already frozen; use decode or report')
    OUT.mkdir(exist_ok=True)
    source = MED / 'source/chinese_train.parquet'
    rows = pq.read_table(source).to_pylist()
    samples = []
    excluded = set()
    if holdout:
        previous = json.loads((OUT.parent / 'selection.json').read_text())
        excluded = {r['source_row'] for r in previous['samples']}
    dictionary = []
    for scene, spec in SCENES.items():
        store = EntityStore(OUT / f'{scene}.db')
        for priority, term in enumerate(spec['terms'][::-1]):
            entity = store.upsert_entity(term, domain=scene, entity_type='TERM',
                normalization_policy='normalize', priority=priority,
                source='Specialty glossary declared before this ASR pilot; CMeEE/MultiMed terminology')
            if re.fullmatch(r'[\u3400-\u9fff]{3,16}', term):
                anchors = {w: 1 for w in spec['anchors'] if w not in term}
                store.set_spelling_context(entity.entity_id, keywords=anchors, minimum_score=1,
                    source='Frozen specialty context; no evaluated ASR errors or aliases')
        definitions = store.list_entities(domain=scene)
        context = build_entity_asr_context(definitions, scene)
        dictionary.append(dict(scene=scene, **spec, context=context.text, prompt_terms=context.terms))
        groups = {True: [], False: []}
        for lo, hi in spec['ranges']:
            for i in range(lo, hi):
                row = rows[i]
                if i in excluded or not 2 <= row['duration'] <= 30:
                    continue
                entries = occurrences(row['text'], spec['terms'])
                groups[bool(entries)].append((i, row, entries))
        for has_entity, number in [(True, 8 if holdout else 12), (False, 4)]:
            pool = groups[has_entity]
            random.Random(f'{SEED}:{scene}:{has_entity}').shuffle(pool)
            for i, row, entries in pool[:number]:
                audio, rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if rate != 16000:
                    from math import gcd
                    divisor = gcd(rate, 16000)
                    audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype('float32')
                path = OUT / 'audio' / f'MultiMed_Chinese_train_{i:04d}.wav'
                path.parent.mkdir(exist_ok=True)
                sf.write(path, audio, 16000, subtype='FLOAT')
                samples.append(dict(id=path.stem, scene=scene, scene_label=spec['label'],
                    group='covered_entity' if has_entity else 'within_scene_control',
                    audio_path=str(path.resolve()), reference=row['text'], entities=entries,
                    actual_duration_seconds=len(audio) / 16000, source_row=i,
                    source_audio_path=row['audio']['path'],
                    source_audio_sha256=hashlib.sha256(row['audio']['bytes']).hexdigest()))
    write_json(OUT / 'glossaries.json', dictionary)
    write_json(OUT / 'selection.json', dict(seed=SEED, noise_snr_db=5, samples=samples,
        method='12 seeded covered + 4 within-scene controls per specialty, 2–30 seconds, before ASR',
        audio_source='https://huggingface.co/datasets/leduckhai/MultiMed',
        audio_role='Public original human recordings; Chinese official train, previously undecoded in this project',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        caveats=['Small specialty pilot, not independent production acceptance',
                 'Original source clips/speakers may overlap earlier official eval/test material',
                 'Caption omissions and spelling errors possible; counts are glossary mentions, not manual NER',
                 'Clean and noisy versions are paired, not independent recordings']))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [OUT / 'selection.json', OUT / 'glossaries.json', *OUT.glob('*.db')]
              if p.name != 'live_baseline.db'}
    write_json(OUT / 'frozen_protocol.json', dict(hashes=hashes,
        arms=['raw', 'plain_entity', 'context_entity', 'scene_prompt'],
        retain='Positive entity gain, zero lost correct mentions, zero regressed samples; otherwise reject',
        inference_inputs='PCM and predeclared specialty only; no reference or gold entities',
        production_changes=False))
    print('FROZEN', len(samples), dict(Counter(r['scene'] for r in samples)), flush=True)


def compact_with_offsets(text):
    value, offsets = [], []
    for index, char in enumerate(text):
        for folded in unicodedata.normalize('NFKC', char).casefold():
            if unicodedata.category(folded)[0] not in {'P', 'Z', 'C'}:
                value.append(folded)
                offsets.append(index)
    return ''.join(value), offsets


def bounded_copy(source, proposed, definitions, anchors, legal_terms, *, restricted=False):
    """Lab-only extraction: copy registered entity spans, never other edits.

    Gate frozen before the new holdout: unique full-span alignment, unchanged
    two-character neighbors, no numeric edits, word boundaries, legal source
    term veto, and Chinese phonetic plausibility. Scores are retrieval cues.
    Unknown cross-script spellings require a medical anchor and an unknown
    Chinese source word. This is still an experimental, fallible ASR verifier.
    """
    a, offsets = compact_with_offsets(source)
    b, _ = compact_with_offsets(proposed)
    maximal = {(e['text'], e['start'], e['end']) for e in occurrences(
        proposed, [d.canonical_text for d in definitions if d.enabled])}
    operations = Levenshtein.opcodes(a, b)
    def source_boundary(position, right):
        exact = set()
        for op in operations:
            if op.dest_start == position:
                exact.add(op.src_start)
            if op.dest_end == position:
                exact.add(op.src_end)
            if op.tag == 'equal' and op.dest_start < position < op.dest_end:
                exact.add(op.src_start + position - op.dest_start)
            if op.tag != 'equal' and op.dest_start < position < op.dest_end:
                return None
        if not exact:
            return None
        return max(exact) if right else min(exact)
    candidates = []
    for d in definitions:
        if not d.enabled or d.normalization_policy != 'normalize':
            continue
        target = normalized(d.canonical_text)
        if len(target) < 3:
            continue
        for found in re.finditer(re.escape(target), b):
            lo, hi = found.span()
            if (d.canonical_text, lo, hi) not in maximal:
                continue
            i, j = source_boundary(lo, False), source_boundary(hi, True)
            if i is None or j is None or not i < j or j > len(offsets):
                continue
            observed = a[i:j]
            if observed == target or observed in legal_terms:
                continue
            start, end = offsets[i], offsets[j - 1] + 1
            literal = source[start:end]
            if re.fullmatch(r'[\u3400-\u9fffA-Za-z]+', literal) is None:
                continue
            if any(c.isdigit() for c in target):
                continue
            if not latin_boundaries_valid(source, start, end):
                continue
            if a[max(0, i - 2):i] != b[max(0, lo - 2):lo] or a[j:j + 2] != b[hi:hi + 2]:
                continue
            nearby = source[max(0, start - 32):start] + source[end:end + 32]
            if not any(w in nearby for w in anchors if w not in d.canonical_text):
                continue
            if target.isascii():
                if observed.isascii():
                    if len(observed) != len(target) or Levenshtein.distance(observed, target) > 1:
                        continue
                elif re.fullmatch(r'[\u3400-\u9fff]{2,4}', observed):
                    if restricted:
                        # No registered, independently sourced cross-script
                        # reading exists in these packs; abstain.
                        continue
                    tokenizer = _tokenizer()
                    if tokenizer is None or tokenizer.FREQ.get(literal, 0) > 0:
                        continue
                else:
                    continue
            else:
                if not re.fullmatch(r'[\u3400-\u9fff]{3,24}', observed):
                    continue
                if candidate_boundary_issue(source, start, end):
                    continue
                if abs(len(observed) - len(target)) > 1:
                    continue
                if restricted and (len(target) < 4 or observed[0] != target[0]
                                   or observed[-2:] != target[-2:]):
                    continue
                if sound_similarity(pronunciation_signature(observed), pronunciation_signature(target)) < 0.72:
                    continue
                if Levenshtein.normalized_similarity(observed, target) < 1 / 3:
                    continue
            candidates.append(dict(start=start, end=end, observed=literal,
                canonical=d.canonical_text, entity_id=d.entity_id))
    selected = []
    for candidate in sorted(candidates, key=lambda c: (-(c['end'] - c['start']), c['start'])):
        rivals = {c['canonical'] for c in candidates
                  if c['start'] < candidate['end'] and candidate['start'] < c['end']}
        if len(rivals) > 1 or any(c['start'] < candidate['end'] and candidate['start'] < c['end'] for c in selected):
            continue
        selected.append(candidate)
    output = source
    for candidate in sorted(selected, key=lambda c: c['start'], reverse=True):
        output = output[:candidate['start']] + candidate['canonical'] + output[candidate['end']:]
    return output, selected


def decode():
    manifest = json.loads((OUT / 'selection.json').read_text())
    glossaries = {d['scene']: d for d in json.loads((OUT / 'glossaries.json').read_text())}
    complete = {(r['id'], r['variant'], r['arm']): r for r in read_jsonl(OUT / 'asr.jsonl')}
    params = SimpleNamespace(seed=manifest['seed'], noise_snr=manifest['noise_snr_db'])
    with httpx.Client(base_url='http://127.0.0.1:8766', timeout=180) as client:
        write_json(OUT / 'asr_health.json', client.get('/health').raise_for_status().json())
        for i, row in enumerate(manifest['samples']):
            for variant in ['clean', 'noise']:
                pcm, duration = audio_bytes(row, variant, params)
                for arm in ['raw', 'scene_prompt']:
                    key = (row['id'], variant, arm)
                    if key in complete:
                        continue
                    context = '' if arm == 'raw' else glossaries[row['scene']]['context']
                    started = time.perf_counter()
                    ready, final = decode_pcm(client, pcm, context)
                    record = dict(**row, variant=variant, arm=arm, context=context,
                        pcm_sha256=hashlib.sha256(pcm).hexdigest(), duration_seconds=duration,
                        elapsed_seconds=time.perf_counter() - started,
                        raw_text=final['text'], ready=ready, final_payload=final)
                    append_jsonl(OUT / 'asr.jsonl', record)
                    complete[key] = record
                    print(i + 1, '/', len(manifest['samples']), row['scene'], row['id'],
                          variant, arm, final['text'], flush=True)


def summarize():
    rows = read_jsonl(OUT / 'asr.jsonl')
    lookup = {(r['id'], r['variant'], r['arm']): r for r in rows}
    statistics = {scene: {arm: Counter() for arm in ['raw', 'plain_entity', 'context_entity', 'scene_prompt', 'bounded_copy', 'restricted_copy']}
                  for scene in SCENES}
    tools = {}
    all_definitions = {}
    legal_terms = {normalized(d['canonical_text']) for d in json.loads((ROOT / 'results/entity_context_20261006/expanded_vocabulary.json').read_text())}
    # Legitimate alternate abbreviations must not be overwritten by a primer.
    legal_terms.update(w.casefold() for w in ['BSA', 'TSA', 'EMB', 'CT', 'MRI', 'PET', 'PSA', 'TAVI', 'TAVR', 'ENB', 'TEE', 'EBUS', 'ECG', 'EEG', 'ALT', 'AST', 'AFP', 'CEA'])
    for scene in SCENES:
        definitions = EntityStore(OUT / f'{scene}.db').list_entities(domain=scene)
        all_definitions[scene] = definitions
        legal_terms.update(normalized(d.canonical_text) for d in definitions)
        for arm in ['plain_entity', 'context_entity']:
            selected = tuple(replace(d, spelling_context=None) for d in definitions) if arm == 'plain_entity' else definitions
            tools[scene, arm] = (EntityProtector(selected), EntityCandidateMatcher(selected, selected_domain=scene, mode='auto'))
    changes = []
    outputs = []
    for row in rows:
        if row['arm'] != 'raw':
            continue
        scene = row['scene']
        key = (row['id'], row['variant'])
        source = row['raw_text']
        results = {'raw': source}
        for arm in ['plain_entity', 'context_entity']:
            protector, matcher = tools[scene, arm]
            prepared = prepare_entity_segment(source, protector, matcher, allow_auto=True)
            results[arm] = prepared.baseline_text
        prompted = lookup.get((*key, 'scene_prompt'))
        if prompted is None:
            raise ValueError(f'Missing prompt decode: {key}')
        if prompted['pcm_sha256'] != row['pcm_sha256']:
            raise ValueError('Paired PCM changed')
        results['scene_prompt'] = prompted['raw_text']
        results['bounded_copy'], patches = bounded_copy(source, prompted['raw_text'],
            all_definitions[scene], [*SCENES[scene]['anchors'], '患者', '病患'], legal_terms)
        results['restricted_copy'], restricted_patches = bounded_copy(source, prompted['raw_text'],
            all_definitions[scene], [*SCENES[scene]['anchors'], '患者', '病患'], legal_terms, restricted=True)
        ref = normalized(row['reference'])
        before = Levenshtein.distance(ref, normalized(source))
        for arm, output in results.items():
            count = statistics[scene][arm]
            after = Levenshtein.distance(ref, normalized(output))
            count.update(samples=1, reference_chars=len(ref), character_errors=after,
                         improved_samples=int(after < before), regressed_samples=int(after > before),
                         expected_mentions=len(row['entities']), originals=int(row['variant'] == 'clean'),
                         within_scene_controls=int(row['group'] == 'within_scene_control'))
            before_terms = Counter(e['text'] for e in occurrences(source, SCENES[scene]['terms']))
            after_terms = Counter(e['text'] for e in occurrences(output, SCENES[scene]['terms']))
            for entity in row['entities']:
                a = before_terms[entity['text']]
                b = after_terms[entity['text']]
                # Occurrence counts, not repeated "term in text" for multi-mention captions.
                ordinal = sum(e['text'] == entity['text'] and e['start'] <= entity['start'] for e in row['entities'])
                was, now = a >= ordinal, b >= ordinal
                count['recovered_mentions'] += int(now)
                count['entity_gains'] += int(now and not was)
                count['entity_losses'] += int(was and not now)
            if output != source:
                changes.append(dict(id=row['id'], variant=row['variant'], scene=scene, arm=arm,
                    source=source, output=output, reference=row['reference'], error_delta=after - before))
        outputs.append(dict(id=row['id'], variant=row['variant'], scene=scene, results=results,
                            bounded_patches=patches, restricted_patches=restricted_patches))
    result = {}
    for scene, arms in statistics.items():
        result[scene] = {}
        for arm, count in arms.items():
            s = dict(count)
            s['retention_gate_passed'] = bool(s.get('entity_gains', 0) > 0
                and s.get('entity_losses', 0) == 0 and s.get('regressed_samples', 0) == 0)
            result[scene][arm] = s
    write_json(OUT / 'summary.json', result)
    write_json(OUT / 'changes.json', changes)
    write_json(OUT / 'outputs.json', outputs)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'decode', 'report'])
    parser.add_argument('--holdout', action='store_true')
    args = parser.parse_args()
    if args.holdout:
        OUT = OUT / 'holdout'
    if args.stage == 'prepare':
        prepare(args.holdout)
    else:
        {'decode': decode, 'report': summarize}[args.stage]()
