"""Package matched native-skip windows for audio review, without assigning gold."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_detection_data import digest, review_basis
from experiments.scripts.benchmark_fixed_gate_pipeline import sha, write


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def packet_basis(row):
    # Only derived labels and the hash itself are excluded. Context, routing,
    # model proposals and exact audio all belong to the frozen review input.
    return digest({k: v for k, v in row.items() if k not in {'packet_basis_sha256', 'review_basis_sha256'}})


def lexical(text):
    return ''.join(c for c in text if unicodedata.category(c)[0] in {'L', 'N', 'M'})


def load_candidates(replay):
    sources = {row['case']: row for row in json.loads((replay / 'sources.json').read_text())}
    sessions = {}
    for row in read_rows(replay / 'sessions.jsonl'):
        if row['repetition'] != 0:
            continue
        key = (row['case'], row['policy'])
        if key in sessions:
            raise ValueError('Duplicate source/policy session')
        sessions[key] = row
    result = []
    for case, source in sorted(sources.items()):
        policies = {p: sessions[case, p] for p in ('native', 'confidence_supplement', 'all_refine')}
        profiles = {p['text']: p for p in source['profiles']}
        if len(profiles) != len(source['profiles']):
            raise ValueError('Ambiguous source profile')
        segments = {}
        for policy, session in policies.items():
            segments[policy] = {s['source']: s for s in session['window_segments']}
            if len(segments[policy]) != len(session['window_segments']):
                raise ValueError('Duplicate source window')
            if session['windows'] != policies['native']['windows'] or session['window_pipeline'] != policies['native']['window_pipeline']:
                raise ValueError('Mismatched paired source windows')
        if not all(set(s) == set(profiles) for s in segments.values()):
            raise ValueError('Incomplete paired window observations')
        provenance = source['audio_provenance']
        plan = provenance['source_plan_case']
        prepared = ''.join(s['baseline'] for s in policies['native']['window_segments'])
        for text, native in segments['native'].items():
            if native['outcome'] != 'gate_skip':
                continue
            profile = profiles[text]
            if source['raw_text'][profile['start']:profile['end']] != profile['raw_source_text']:
                raise ValueError('Stale source offsets')
            full = segments['all_refine'][text]
            supplement = segments['confidence_supplement'][text]
            stratum = ('accepted_changed' if full['outcome'] == 'accepted' and full['output'] != full['baseline']
                       else 'accepted_unchanged' if full['outcome'] == 'accepted'
                       else 'rejected' if full['outcome'] == 'rejected' else None)
            if stratum is None:
                raise ValueError('Incomplete all-refine outcome')
            confidence = profile['confidence']
            score = confidence['score']
            row = dict(id=f'safety-case-{case:02d}-window-{profile["index"]:02d}', case=case,
                window_index=profile['index'], raw_asr=text, baseline_text=native['baseline'],
                raw_recording_text=source['raw_text'], prepared_recording_text=prepared,
                left_context=source['raw_text'][:profile['start']][-160:],
                right_context=source['raw_text'][profile['end']:][:160], available_context='same isolated audio case',
                raw_source_start=profile['start'], raw_source_end=profile['end'],
                source_audio_path=provenance['audio_path'], source_audio_sha256=provenance['audio_sha256'],
                source_audio_used_samples=provenance['used_samples'],
                source_used_float32_pcm_sha256=provenance['used_float32_pcm_sha256'],
                recording_group=plan['recording_group'], dependency_group=plan['dependency_group'],
                category=plan['category'], source_plan_case=plan,
                source_final_sha256=source['source_final_sha256'],
                selection_stratum=stratum, score_band='unknown' if score is None else 'high' if score >= .98 else 'other',
                change_kind=('punctuation_or_spacing_only' if lexical(full['output']) == lexical(full['baseline'])
                             else 'lexical_change') if stratum == 'accepted_changed' else 'no_accepted_change',
                native_gate=native['gate'], supplement_gate=supplement['gate'],
                supplement_reviewed=supplement['gate']['action'] == 'refine', confidence_profile=profile,
                all_refine_result={k: full.get(k) for k in ('outcome', 'baseline', 'output', 'reject_reasons', 'model_calls')},
                supplement_result={k: supplement.get(k) for k in ('outcome', 'output', 'reject_reasons')},
                numeric_rule=native.get('numeric_rule'), entity_candidates=[], entity_hints=[],
                protection=native.get('numeric_rule'), window_pipeline=policies['native']['window_pipeline'],
                split='regression', split_group=plan['dependency_group'],
                desired_text=None, edits=None, character_tags=None, insertion_gaps=None,
                label_status='awaiting_audio_review', human_audio_gold=False,
                training_eligible=False, independent_acceptance_eligible=False, production_eligible=False)
            result.append(row)
    return result


def select_candidates(rows, controls=12):
    selected = [r for r in rows if r['selection_stratum'] == 'accepted_changed']
    # Review lexical changes first, then punctuation; group siblings together.
    selected.sort(key=lambda r: (r['change_kind'] != 'lexical_change', r['case'], r['window_index']))
    for stratum in ('accepted_unchanged', 'rejected'):
        pool = [r for r in rows if r['selection_stratum'] == stratum]
        if len(pool) < controls:
            raise ValueError('Insufficient controls')
        chosen = [r for r in pool if r['supplement_reviewed']]
        if len(chosen) > controls:
            raise ValueError('Control quota smaller than supplemental reviews')
        buckets = defaultdict(list)
        seen = {r['id'] for r in chosen}
        for row in pool:
            if row['id'] not in seen:
                buckets[row['dependency_group'], row['score_band']].append(row)
        queues = {k: deque(sorted(v, key=lambda r: digest(['safety-review-control-v1', r['id']]))) for k, v in buckets.items()}
        # Published read speech is an auxiliary control, not the main cleanup
        # domain. At most two public controls in either control stratum.
        while len(chosen) < controls:
            advanced = False
            for key in sorted(queues):
                queue = queues[key]
                if not queue:
                    continue
                if (queue[0]['category'] == 'public_read_speech_reference'
                    and sum(r['category'] == 'public_read_speech_reference' for r in chosen) >= 2):
                    continue
                chosen.append(queue.popleft()); advanced = True
                if len(chosen) == controls:
                    break
            if not advanced:
                raise ValueError('Cannot fill source-balanced control quota')
        selected.extend(chosen)
    if len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate selected review')
    return deepcopy(selected)


def build(args):
    import soundfile as sf

    rows = load_candidates(args.replay)
    selected = select_candidates(rows, args.controls)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audio').mkdir()
    used_audio = {}; verified_originals = {}
    for row in selected:
        case = row['case']
        if case not in used_audio:
            original = Path(row['source_audio_path'])
            if sha(original) != row['source_audio_sha256']:
                raise ValueError('Source audio changed')
            provenance = row['source_plan_case']['audio_provenance']
            original_recording = Path(provenance['original_path'])
            if original_recording not in verified_originals:
                verified_originals[original_recording] = sha(original_recording)
            if verified_originals[original_recording] != provenance['original_sha256']:
                raise ValueError('Original recording changed')
            pcm, rate = sf.read(original, dtype='float32', frames=row['source_audio_used_samples'])
            if rate != 16000 or pcm.ndim != 1 or len(pcm) != row['source_audio_used_samples']:
                raise ValueError('Unexpected source PCM format')
            pcm_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
            if pcm_hash != row['source_used_float32_pcm_sha256']:
                raise ValueError('Packaged audio must match exact ASR input samples')
            name = f'audio/case_{case:02d}.wav'
            sf.write(output / name, pcm, rate, subtype='FLOAT')
            packaged_pcm, _ = sf.read(output / name, dtype='float32')
            if hashlib.sha256(packaged_pcm.tobytes()).hexdigest() != pcm_hash:
                raise ValueError('Packaged PCM changed')
            used_audio[case] = dict(packaged_audio=name, audio_path=str(output / name),
                audio_sha256=sha(output / name), audio_provenance={**provenance,
                    'duration_seconds': len(pcm)/rate,
                    'clip_start_seconds': row['source_plan_case']['original_used_start_seconds'],
                    'clip_end_seconds': row['source_plan_case']['original_used_end_seconds'],
                    'packaged_float32_pcm_sha256': pcm_hash,
                    'timing_granularity': 'whole_used_audio_case_not_word_or_window_alignment'})
        row.update(deepcopy(used_audio[case]))
        row['review_basis_sha256'] = review_basis(row)
        row['packet_basis_sha256'] = packet_basis(row)
    write_rows(output / 'all_candidates.jsonl', rows)
    write_rows(output / 'review_queue.jsonl', selected)
    write_rows(output / 'review_drafts.jsonl', [dict(id=r['id'], packet_basis_sha256=r['packet_basis_sha256'],
        review_basis_sha256=r['review_basis_sha256'], reviewer='', audio_checked=False, review_kind=None,
        decision='uncertain', desired_text=None, edit_tasks=[], asr_fidelity='uncertain',
        refiner_effect='unassessable', note='') for r in selected])
    queue_hash = sha(output / 'review_queue.jsonl')
    template = ROOT / 'experiments/fixtures/confidence_safety_review.html'
    payload = json.dumps(dict(queue=selected, queue_sha256=queue_hash), ensure_ascii=False).replace('<', '\\u003c')
    (output / 'review.html').write_text(template.read_text().replace('__REVIEW_DATA__', payload))
    manifest = dict(scope='native_skip_recall_review_consumed_development_not_independent_acceptance',
        candidate_windows=len(rows), selected_windows=len(selected), selected_audio_cases=len(used_audio),
        selection_counts=dict(Counter(r['selection_stratum'] for r in selected)),
        change_kinds=dict(Counter(r['change_kind'] for r in selected)),
        sources=dict(Counter(r['category'] for r in selected)),
        original_recording_groups=len({r['recording_group'] for r in selected}),
        dependency_groups=len({r['dependency_group'] for r in selected}),
        supplemental_reviews_included=sum(r['supplement_reviewed'] for r in selected),
        unique_audio_seconds=sum(v['audio_provenance']['duration_seconds'] for v in used_audio.values()),
        human_audio_gold=0, training_eligible=0, no_new_ASR_or_refiner_calls=True,
        production_services_changed=False, score_calibrated=False,
        selection_policy='All native-skipped accepted changes + 12 unchanged/12 rejected controls; include all supplemental reviews; controls balanced by dependency group/score band with a public-domain cap',
        source_sha256={str((args.replay / name).resolve()): sha(args.replay / name)
                       for name in ('sources.json', 'sessions.jsonl', 'configuration.json')},
        output_sha256={name: sha(output / name) for name in ('all_candidates.jsonl', 'review_queue.jsonl', 'review_drafts.jsonl', 'review.html')},
        packaged_audio_sha256={v['packaged_audio']: v['audio_sha256'] for v in used_audio.values()},
        limitations=['Model proposals are not correctness or necessity labels',
            'Selected control strata are biased; do not extrapolate population error rates',
            'Project recordings have been consumed; some original recordings may be related',
            'Published read speech does not provide spontaneous-speech cleanup gold',
            'No exact word/window audio alignment; play the full used input case'])
    write(output / 'manifest.json', manifest)
    (output / 'executed_code').mkdir()
    for path in (Path(__file__), template, ROOT / 'system/edit_detection_data.py'):
        (output / 'executed_code' / path.name).write_bytes(path.read_bytes())
    print(json.dumps({k: manifest[k] for k in ('candidate_windows', 'selected_windows', 'selected_audio_cases',
        'selection_counts', 'change_kinds', 'supplemental_reviews_included', 'unique_audio_seconds')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, default=ROOT / 'results/refiner_audit_expansion_20261013_replay')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls', type=int, choices=[12], default=12)
    build(parser.parse_args())
