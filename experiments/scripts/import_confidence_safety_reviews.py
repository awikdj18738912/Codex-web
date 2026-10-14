"""Validate safety-net audio reviews and report diagnostic recall, never train."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.edit_detection_data import align_edits, attach_review, review_basis
from experiments.scripts.prepare_confidence_safety_review import packet_basis, read_rows, write_rows
from experiments.scripts.benchmark_fixed_gate_pipeline import sha, write


def validate_review(row, review, directory):
    if review.get('id') != row['id'] or review.get('packet_basis_sha256') != packet_basis(row):
        raise ValueError('Unknown or stale packet')
    if review.get('review_basis_sha256') != review_basis(row):
        raise ValueError('Stale input basis')
    if review.get('review_kind') != 'human_audio_cleaned' or review.get('audio_checked') is not True:
        raise ValueError('Explicit human audio review required')
    if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        raise ValueError('Reviewer required')
    if not isinstance(review.get('note', ''), str):
        raise ValueError('Review note must be text')
    if review.get('asr_fidelity') not in {'correct', 'incorrect', 'uncertain'}:
        raise ValueError('Explicit ASR fidelity assessment required')
    effect = review.get('refiner_effect')
    if effect not in {'beneficial', 'neutral', 'harmful', 'mixed', 'unassessable'}:
        raise ValueError('Explicit Refiner effect assessment required')
    changed = row['all_refine_result']['output'] != row['baseline_text']
    if not changed and effect in {'beneficial', 'harmful', 'mixed'}:
        raise ValueError('Unchanged final output cannot be an assessed change')
    for path, expected in ((Path(row['source_audio_path']), row['source_audio_sha256']),
            (directory / row['packaged_audio'], row['audio_sha256']),
            (Path(row['audio_provenance']['original_path']), row['audio_provenance']['original_sha256'])):
        if sha(path) != expected:
            raise ValueError('Audio provenance changed')
    decision = review.get('decision')
    if decision not in {'keep', 'edit', 'uncertain'}:
        raise ValueError('Explicit refinement decision required')
    if decision == 'uncertain':
        if review.get('desired_text') is not None or review.get('edit_tasks') != [] or not review.get('note', '').strip():
            raise ValueError('Uncertain review needs a note and no target/edit labels')
        result = {**row, 'review': review, 'label_status': 'audio_review_uncertain',
                  'necessary_refinement': None, 'human_audio_gold': False}
    else:
        target = review.get('desired_text')
        tasks = review.get('edit_tasks')
        if not isinstance(target, str) or not target.strip() or not isinstance(tasks, list):
            raise ValueError('Explicit nonempty target and edit tasks required')
        if decision == 'keep' and (target != row['baseline_text'] or tasks):
            raise ValueError('KEEP must preserve baseline and have no edits')
        if decision == 'edit' and target == row['baseline_text']:
            raise ValueError('EDIT target must differ from baseline')
        if decision == 'keep' and effect == 'beneficial':
            raise ValueError('Necessary change and KEEP assessment conflict')
        normalized = dict(review)
        if review.get('single_task_for_all_edits') is True and len(tasks) == 1:
            normalized['edit_tasks'] = tasks * len(align_edits(row['baseline_text'], target)['edits'])
        result = {**attach_review(row, normalized), 'necessary_refinement': decision == 'edit'}
    result.update(training_eligible=False, independent_acceptance_eligible=False, production_eligible=False,
        allowed_usage='verified_consumed_native_skip_diagnostic_only', refiner_effect=effect,
        asr_fidelity=review['asr_fidelity'])
    return result


def summarize_verified(rows):
    known = [r for r in rows if r['necessary_refinement'] is not None]
    needed = [r for r in known if r['necessary_refinement']]
    covered = [r for r in needed if r['supplement_reviewed']]
    beneficial = [r for r in needed if r['refiner_effect'] == 'beneficial']
    return dict(reviewed_records=len(rows), decided_records=len(known), uncertain_records=len(rows)-len(known),
        necessary_refinement_windows=len(needed), supplement_reviewed_necessary_windows=len(covered),
        still_skipped_necessary_windows=len(needed)-len(covered),
        necessary_refinement_coverage_on_reviewed_native_skips=len(covered)/len(needed) if needed else None,
        all_refine_beneficial_windows=len(beneficial),
        supplement_reviewed_beneficial_windows=sum(r['supplement_reviewed'] for r in beneficial),
        supplement_exact_target_matches=sum(r['supplement_reviewed'] and r['supplement_result']['output'] == r['desired_text'] for r in needed),
        full_refiner_effects=dict(Counter(r['refiner_effect'] for r in rows)),
        asr_fidelity=dict(Counter(r['asr_fidelity'] for r in rows)),
        scope='partial_source_biased_consumed_native_skip_diagnostic_not_population_or_independent_recall',
        training_eligible=0, independent_acceptance_eligible=0, production_eligible=False)


def run(args):
    manifest = json.loads((args.queue / 'manifest.json').read_text())
    for name in ('review_queue.jsonl',):
        if sha(args.queue / name) != manifest['output_sha256'][name]:
            raise ValueError('Frozen review queue changed')
    queue = read_rows(args.queue / 'review_queue.jsonl')
    rows = {r['id']: r for r in queue}
    if len(rows) != len(queue):
        raise ValueError('Duplicate queue identity')
    reviews = read_rows(args.reviews)
    counts = Counter(r.get('id') for r in reviews)
    verified = []; errors = []
    for review in reviews:
        identity = review.get('id')
        try:
            if identity not in rows or counts[identity] != 1:
                raise ValueError('Unknown or duplicate review identity')
            verified.append(validate_review(rows[identity], review, args.queue))
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(dict(id=identity, error=str(error)))
    args.output.mkdir(parents=True, exist_ok=False)
    write_rows(args.output / 'verified_diagnostics.jsonl', verified)
    write_rows(args.output / 'rejected_reviews.jsonl', errors)
    summary = {**summarize_verified(verified), 'rejected_reviews': len(errors),
        'unreviewed_windows': len(rows)-len(verified), 'queue_windows': len(rows),
        'queue_sha256': sha(args.queue / 'review_queue.jsonl'), 'reviews_sha256': sha(args.reviews)}
    write(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
