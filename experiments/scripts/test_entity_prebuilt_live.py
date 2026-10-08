"""Preload a frozen entity library, then send real PCM through the current app.

No ASR transport replay, fake Refiner, reference-derived aliases, or production
database writes. Independent app instances share one real local GPU Refiner
for memory efficiency; all sessions use the unmodified WebSocket handler and
the running Qwen ASR endpoint. This measures current behavior, not a new rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time
from types import SimpleNamespace
from urllib.parse import urlencode

from rapidfuzz.distance import Levenshtein

from evaluate_entity_library import ROOT, append_jsonl, audio_bytes, normalized, read_jsonl, write_json
from evaluate_entity_medical_scenes import occurrences
from system.entity_store import EntityStore

OUT = ROOT / 'results/entity_prebuilt_retest_20261007'
SCENES = ROOT / 'results/entity_medical_scenes_20261006'
MED = ROOT / 'results/medical_entity_20261003'
MODEL = Path('/home/aim0/data/models/ASR/AgenticASR-Refiner')
CONDITIONS = ('no_dictionary', 'current_library', 'prebuilt_scene')


def prepare():
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Frozen protocol exists; use run or report')
    previous = json.loads((SCENES / 'holdout/selection.json').read_text())
    vocabularies = {d['scene']: d for d in json.loads((SCENES / 'glossaries.json').read_text())}
    for scene in vocabularies:
        original = EntityStore(SCENES / f'{scene}.db').list_entities(domain=scene)
        store = EntityStore(OUT / f'{scene}.db')
        for d in original:
            copied = store.upsert_entity(d.canonical_text, entity_type=d.entity_type,
                domain='medical', normalization_policy=d.normalization_policy,
                priority=d.priority, aliases=d.aliases, source=d.source)
            if d.spelling_context:
                p = d.spelling_context
                store.set_spelling_context(copied.entity_id, keywords=dict(p.keywords),
                    minimum_score=p.minimum_score, source=p.source)
    # Readonly backup of the real user library. Initialization/migrations, if
    # needed, happen only to this test copy, never the real database.
    with sqlite3.connect(OUT / 'live_baseline.db') as src, sqlite3.connect(OUT / 'current_library.db') as dst:
        src.backup(dst)
    EntityStore(OUT / 'current_library.db').list_entities()
    pool = list(previous['samples'])
    rows = [dict(r, cohort='medical_main', seed=previous['seed'],
                 noise_snr_db=previous['noise_snr_db'], terms=vocabularies[r['scene']]['terms'])
            for r in pool]
    ordinary = read_jsonl(MED / 'controls/asr.jsonl')
    ordinary = {r['id']: r for r in ordinary if r['variant'] == 'clean'}
    for key in sorted(ordinary)[:4]:
        r = ordinary[key]
        rows.append(dict(id=r['id'], audio_path=r['audio_path'], reference=r['reference'],
            entities=[], group='ordinary_control', scene='cardiopulmonary',
            scene_label='普通领域负例', cohort='ordinary_control', seed=20261003,
            noise_snr_db=10, terms=vocabularies['cardiopulmonary']['terms']))
    demos = json.loads((MED / 'development/selection.json').read_text())['samples']
    demo = next(r for r in demos if r['id'] == 'MultiMed_Chinese_eval_0003')
    rows.append(dict(demo, scene='cardiopulmonary', scene_label='已知 TAVI 别名功能检查',
        cohort='known_alias_demo', seed=20261003, noise_snr_db=10,
        terms=['TAVI', 'TAVI手术']))
    # First declared covered recording in each scene, plus the known alias
    # demo, checks the incremental mode as well. No selection by results.
    streaming_ids = [next(r['id'] for r in pool if r['scene'] == scene and r['group'] == 'covered_entity')
                     for scene in vocabularies] + [demo['id']]
    protocol = dict(samples=rows, conditions=list(CONDITIONS),
        primary_mode='offline', streaming_smoke_ids=streaming_ids,
        configuration=dict(asr_url='http://127.0.0.1:8766', language='Chinese',
            domain='medical', entity_fuzzy_mode='auto', entity_asr_context=False,
            gate='tri_state', final_refinement='off', numeric_backend='zh-itn-v90'),
        method='Real PCM -> existing Qwen stream endpoint -> current WebSocket pipeline -> real local GPU Refiner -> entity/numeric guards',
        dictionary_input='Prebuilt 2026-10-06 standard names/contexts unchanged; no new error aliases',
        attribution='Only paired sessions with identical raw ASR text can support entity-library effect',
        cautions=['Retest of previously seen public human recordings, not a fresh holdout',
                  'Known alias demo separate from main medical scores',
                  'Unknown source speakers/video overlaps; no speaker independence claim',
                  'Caption omissions and English missing labels; mention counts are not manual NER'])
    write_json(OUT / 'protocol.json', protocol)
    write_json(OUT / 'frozen_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [OUT / 'protocol.json', *OUT.glob('*.db')]})
    print('FROZEN', len(rows), dict(Counter(r['cohort'] for r in rows)), flush=True)


def run(mode):
    import system.web_app as web
    from fastapi.testclient import TestClient
    from system.live_qwen_refiner import TransformersRefiner
    from system.repetition_plausibility import RepetitionPlausibility
    from system.zh_itn_bridge import DEFAULT_LIBRARY
    protocol = json.loads((OUT / 'protocol.json').read_text())
    rows = protocol['samples']
    if mode == 'streaming':
        rows = [r for r in rows if r['id'] in protocol['streaming_smoke_ids']]
    for name, expected in json.loads((OUT / 'frozen_hashes.json').read_text()).items():
        if hashlib.sha256((OUT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen file changed: {name}')
    model = TransformersRefiner(MODEL, 'cuda:0', 256)
    calls = []
    class AuditedRefiner:
        def __getattr__(self, name):
            value = getattr(model, name)
            if not callable(value) or not name.startswith('refine'):
                return value
            def invoke(*args, **kwargs):
                started = time.perf_counter()
                result = value(*args, **kwargs)
                calls.append(dict(method=name, input=list(args), kwargs=kwargs,
                    output=result, wall_seconds=time.perf_counter() - started))
                return result
            return invoke
    audited = AuditedRefiner()
    web.TransformersRefiner = lambda *_args, **_kwargs: audited
    plausibility = RepetitionPlausibility('cuda:0')
    web.RepetitionPlausibility = lambda *_args, **_kwargs: plausibility
    apps = {}
    def application(condition, scene):
        key = (condition, scene)
        if key not in apps:
            db = None if condition == 'no_dictionary' else OUT / 'current_library.db' if condition == 'current_library' else OUT / f'{scene}.db'
            apps[key] = web.create_app(MODEL, 'cuda:0', 'http://127.0.0.1:8766',
                'Chinese', 256, None, entity_db=db,
                entity_fuzzy_mode='off' if db is None else 'auto',
                refinement_gate_mode='tri_state', final_refinement_mode='off',
                zh_itn_library=DEFAULT_LIBRARY, entity_asr_context=False)
        return apps[key]
    dest = OUT / f'live_{mode}.jsonl'
    complete = {(r['id'], r['variant'], r['condition']): r for r in read_jsonl(dest)}
    for index, row in enumerate(rows):
        for variant in ['clean', 'noise']:
            pcm, duration = audio_bytes(row, variant,
                SimpleNamespace(seed=row['seed'], noise_snr=row['noise_snr_db']))
            # Alternate order to reduce systematic warm-up/order effects.
            conditions = list(CONDITIONS) if (index + (variant == 'noise')) % 2 == 0 else list(reversed(CONDITIONS))
            for condition in conditions:
                key = (row['id'], variant, condition)
                if key in complete:
                    continue
                calls.clear()
                events = []
                app = application(condition, row['scene'])
                started = time.perf_counter()
                with TestClient(app) as client:
                    if condition != 'no_dictionary':
                        loaded = client.get('/api/entities').raise_for_status().json()['entities']
                    else:
                        loaded = []
                    query = urlencode(dict(mode=mode, domain='medical', language='Chinese',
                        asr_chunk_seconds='0.5', entity_asr_context='0'))
                    with client.websocket_connect('/ws/stream?' + query) as socket:
                        ready = socket.receive_json()
                        if ready.get('event') != 'ready' or ready.get('entity_asr_context_applied'):
                            raise ValueError(f'Unexpected ASR ready: {ready}')
                        for offset in range(0, len(pcm), 64000):
                            socket.send_bytes(pcm[offset:offset + 64000])
                            while True:
                                event = socket.receive_json()
                                events.append(event)
                                if event.get('event') == 'error':
                                    raise RuntimeError(event)
                                if event.get('event') == 'chunk_ack':
                                    break
                        socket.send_json(dict(event='finish'))
                        while True:
                            event = socket.receive_json()
                            events.append(event)
                            if event.get('event') == 'error':
                                raise RuntimeError(event)
                            if event.get('event') == 'final':
                                final = event
                                break
                record = dict(id=row['id'], variant=variant, condition=condition, mode=mode,
                    scene=row['scene'], cohort=row['cohort'], reference=row['reference'],
                    pcm_sha256=hashlib.sha256(pcm).hexdigest(), duration_seconds=duration,
                    loaded_entities=loaded, ready=ready, result=final, model_calls=list(calls),
                    elapsed_seconds=time.perf_counter() - started,
                    event_counts=dict(Counter(e.get('event') for e in events)),
                    intermediate_updates=[e for e in events if e.get('event') == 'update'])
                append_jsonl(dest, record)
                complete[key] = record
                print('LIVE', mode, index + 1, '/', len(rows), row['id'], variant,
                    condition, final['clean_text'], 'calls', len(calls), flush=True)


def report():
    from system.zh_itn_bridge import ZhITNNormalizer
    protocol = json.loads((OUT / 'protocol.json').read_text())
    manifest = {r['id']: r for r in protocol['samples']}
    itn = ZhITNNormalizer()
    summaries = {}
    changes = []
    mismatches = []
    attributable = {}
    for mode in ['offline', 'streaming']:
        rows = read_jsonl(OUT / f'live_{mode}.jsonl')
        if not rows:
            continue
        lookup = {(r['id'], r['variant'], r['condition']): r for r in rows}
        stats = {}
        timings = {}
        for r in rows:
            base = lookup.get((r['id'], r['variant'], 'no_dictionary'))
            if base is None:
                continue
            if base['pcm_sha256'] != r['pcm_sha256']:
                raise ValueError('PCM mismatch')
            raw_equal = base['result']['raw_text'] == r['result']['raw_text']
            if not raw_equal:
                mismatches.append(dict(mode=mode, id=r['id'], variant=r['variant'],
                    condition=r['condition'], baseline=base['result']['raw_text'], raw=r['result']['raw_text']))
            row = manifest[r['id']]
            terms = row['terms']
            ref = normalized(itn.normalize(row['reference']).text)
            a, b = base['result']['clean_text'], r['result']['clean_text']
            ae = Levenshtein.distance(ref, normalized(itn.normalize(a).text))
            be = Levenshtein.distance(ref, normalized(itn.normalize(b).text))
            key = (row['cohort'], row['scene'], r['condition'])
            s = stats.setdefault(key, Counter())
            s.update(samples=1, paired_raw_identical=int(raw_equal), reference_chars=len(ref),
                character_errors=be, improved_samples=int(be < ae), regressed_samples=int(be > ae),
                changed_samples=int(a != b), refiner_calls=r['result'].get('session_refiner_call_count', len(r['model_calls'])),
                normalizations=len(r['result'].get('entity_normalizations', [])),
                hinted_names=len(r['result'].get('entity_refinement_hints', [])))
            expected = Counter(e['text'] for e in occurrences(row['reference'], terms))
            before = Counter(e['text'] for e in occurrences(a, terms))
            after = Counter(e['text'] for e in occurrences(b, terms))
            for term, count in expected.items():
                bc, ac = min(count, before[term]), min(count, after[term])
                s['expected_mentions'] += count
                s['recovered_mentions'] += ac
                s['entity_gains'] += max(0, ac - bc)
                s['entity_losses'] += max(0, bc - ac)
            timings.setdefault(key, []).append(r['elapsed_seconds'])
            if a != b or r['result'].get('entity_normalizations'):
                changes.append(dict(mode=mode, id=r['id'], variant=r['variant'],
                    cohort=row['cohort'], scene=row['scene'], condition=r['condition'], raw_equal=raw_equal,
                    raw=r['result']['raw_text'], reference=row['reference'], baseline=a, output=b,
                    error_delta=be-ae, normalizations=r['result'].get('entity_normalizations', []),
                    hints=r['result'].get('entity_refinement_hints', []),
                    candidates=r['result'].get('entity_candidates', []),
                    reject_reasons=r['result'].get('refiner_reject_reasons', [])))
        summary = []
        for key, s in stats.items():
            cohort, scene, condition = key
            t = sorted(timings[key])
            summary.append(dict(cohort=cohort, scene=scene, condition=condition, **dict(s),
                elapsed_median_seconds=statistics.median(t), elapsed_p95_seconds=t[int(.95*(len(t)-1))]))
        summaries[mode] = summary
        # A common intersection gives all arms identical samples and exact
        # ASR inputs. Exclude unstable/incomplete pairs from causal reporting.
        paired = {}
        retained_pairs = []
        excluded_pairs = []
        for sample_id, variant in sorted({(r['id'], r['variant']) for r in rows}):
            triplet = [lookup.get((sample_id, variant, c)) for c in CONDITIONS]
            if any(r is None for r in triplet):
                continue
            if len({r['pcm_sha256'] for r in triplet}) != 1:
                raise ValueError('PCM mismatch in triplet')
            if len({r['result']['raw_text'] for r in triplet}) != 1:
                excluded_pairs.append(dict(id=sample_id, variant=variant))
                continue
            retained_pairs.append(dict(id=sample_id, variant=variant))
            row = manifest[sample_id]
            expected = Counter(e['text'] for e in occurrences(row['reference'], row['terms']))
            base_text = triplet[0]['result']['clean_text']
            before = Counter(e['text'] for e in occurrences(base_text, row['terms']))
            ref = normalized(itn.normalize(row['reference']).text)
            base_errors = Levenshtein.distance(ref, normalized(itn.normalize(base_text).text))
            for r in triplet:
                text = r['result']['clean_text']
                after = Counter(e['text'] for e in occurrences(text, row['terms']))
                errors = Levenshtein.distance(ref, normalized(itn.normalize(text).text))
                key = (row['cohort'], row['scene'], r['condition'])
                s = paired.setdefault(key, Counter())
                s.update(samples=1, reference_chars=len(ref), character_errors=errors,
                    improved_samples=int(errors < base_errors), regressed_samples=int(errors > base_errors),
                    changed_samples=int(text != base_text), actual_refiner_calls=len(r['model_calls']),
                    normalizations=len(r['result'].get('entity_normalizations', [])),
                    hinted_names=len(r['result'].get('entity_refinement_hints', [])))
                for term, n in expected.items():
                    bc, ac = min(n, before[term]), min(n, after[term])
                    s['expected_mentions'] += n
                    s['recovered_mentions'] += ac
                    s['entity_gains'] += max(0, ac - bc)
                    s['entity_losses'] += max(0, bc - ac)
        attributable[mode] = dict(
            rule='All three conditions share byte-identical PCM and exact raw ASR text',
            retained_pairs=retained_pairs, excluded_unstable_pairs=excluded_pairs,
            summary=[dict(cohort=k[0], scene=k[1], condition=k[2], **dict(v)) for k, v in paired.items()])
    write_json(OUT / 'summary.json', summaries)
    write_json(OUT / 'changes.json', changes)
    write_json(OUT / 'raw_mismatches.json', mismatches)
    write_json(OUT / 'attributable_summary.json', attributable)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    print('RAW MISMATCHES', len(mismatches), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'report'])
    parser.add_argument('--mode', choices=['offline', 'streaming'], default='offline')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'run':
        run(args.mode)
    else:
        report()
