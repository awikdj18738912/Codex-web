"""Paired evaluation of narrowly constrained entity spelling rules."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import re
import time

from evaluate_entity_library import ROOT, read_jsonl, write_json, append_jsonl, normalized
from system.entity_store import EntityStore, EntitySpellingContext
from system.entity_matcher import EntityCandidateMatcher
from system.entity_pipeline import prepare_entity_segment
from system.protection import EntityProtector
from rapidfuzz.distance import Levenshtein

OUT = ROOT / 'results/entity_rule_optimization_20261007'
SCENES = ROOT / 'results/entity_medical_scenes_20261006'
KEYWORDS = ('患者', '医生', '检查', '手术', '症状', '治疗', '疾病', '服用', '开刀')


def profiles(definitions):
    result = []
    for d in definitions:
        if re.fullmatch(r'[\u3400-\u9fff]{5,16}', d.canonical_text) and d.entity_type == 'TERM':
            old = d.spelling_context
            words = dict(old.keywords) if old else {}
            words.update({w: 1.0 for w in KEYWORDS if w not in d.canonical_text})
            policy = EntitySpellingContext(tuple(words.items()), old.minimum_score if old else 1.0,
                'Declared medical context; identical for baseline and proposed rule',
                d.canonical_text, d.entity_type, d.domain, allow_internal_omission=True)
            d = replace(d, spelling_context=policy)
        result.append(d)
    return tuple(result)


def suites():
    for scene in ['cardiopulmonary', 'orthopedics', 'urology']:
        definitions = profiles(EntityStore(SCENES / f'{scene}.db').list_entities(domain=scene))
        rows = [r for folder in [SCENES, SCENES / 'holdout'] for r in read_jsonl(folder / 'asr.jsonl')
                if r['arm'] == 'raw' and r['scene'] == scene]
        yield scene, definitions, scene, rows
    defs = profiles(EntityStore(ROOT/'results/entity_ranker_20261006/expanded_medical.db').list_entities(domain='medical'))
    yield 'medical_430', defs, 'medical', read_jsonl(ROOT/'results/medical_entity_20261003/asr.jsonl')
    data = json.loads((ROOT/'results/medical_entity_20261003/source/CMeEE_train.json').read_text())
    random.Random(20261007).shuffle(data)
    rows = [dict(id=f'correct_medical_{i}', variant='original', raw_text=r['text'], reference=r['text']) for i,r in enumerate(data[:1000])]
    yield 'correct_medical_text', defs, 'medical', rows
    droot = ROOT/'results/entity_ranker_20261006/drug_names'
    yield 'drug_tts', profiles(EntityStore(droot/'context_drug_names.db').list_entities(domain='drug_names')), 'drug_names', read_jsonl(droot/'asr.jsonl')
    nroot = ROOT/'results/entity_context_20261006'
    yield 'news_names', EntityStore(nroot/'context_names.db').list_entities(domain='general'), 'general', read_jsonl(nroot/'name_holdout_asr.jsonl')


def run(label):
    destination = OUT / f'{label}.jsonl'
    if destination.exists():
        raise FileExistsError(destination)
    summaries = {}
    baseline = {(r['suite'],r['id'],r['variant']):r for r in read_jsonl(OUT/'baseline.jsonl')} if label != 'baseline' else {}
    for suite, definitions, domain, rows in suites():
        protector = EntityProtector(definitions, protect_numeric_spans=False)
        matcher = EntityCandidateMatcher(definitions, selected_domain=domain, mode='auto')
        stats = Counter()
        for row in rows:
            started = time.perf_counter()
            p = prepare_entity_segment(row['raw_text'], protector, matcher, allow_auto=True)
            record = dict(suite=suite, id=row['id'], variant=row['variant'], source=row['raw_text'],
                reference=row['reference'], output=p.baseline_text, normalizations=p.normalizations,
                elapsed_ms=(time.perf_counter()-started)*1000)
            append_jsonl(destination, record)
            old = baseline.get((suite,row['id'],row['variant']), record)['output']
            ref = normalized(row['reference'])
            before, after = Levenshtein.distance(ref,normalized(old)), Levenshtein.distance(ref,normalized(p.baseline_text))
            stats.update(samples=1, changed=int(old != p.baseline_text), improved=int(after < before),
                regressed=int(after > before), error_delta=after-before)
        summaries[suite] = dict(stats)
        print(suite, dict(stats), flush=True)
    write_json(OUT/f'{label}_summary.json', summaries)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('label');args=p.parse_args();run(args.label)
