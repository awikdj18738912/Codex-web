"""Compare identical candidate slots; re-deduplicate each 300-slot subset."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parent / 'refiner_teacher_scale1000_20261017'
sys.path.insert(0, str(OUT.parents[1] / 'pipeline'))
from configs import settings
from src.pipeline.dedup import deduplicate


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def structures(directory, ids):
    found = {}
    for row in rows(directory / 'api_calls.jsonl'):
        if row.get('stage') != 'oral' or row['id'] not in ids or row['id'] in found:
            continue
        prompt = row['request']['user_prompt']
        schema = re.search(r'correction_values结构必须为：(\[[^\]]*\])', prompt).group(1)
        found[row['id']] = {'[]': 'none_or_passthrough', '[A,B]': 'single',
                            '[A,B,A]': 'rollback', '[A,B,C]': 'multiple'}[schema]
    assert set(found) == ids
    return found


def summarize(directory, ids):
    attempts = [r for r in rows(directory / 'attempts.jsonl') if r['id'] in ids]
    assert len(attempts) == len(ids)
    accepted = sorted([r for r in rows(directory / 'accepted_candidates.jsonl') if r['id'] in ids], key=lambda r: r['id'])
    unique = deduplicate(accepted, threshold=settings.DEDUP_THRESHOLD)
    control = structures(directory, ids)
    breakdown = defaultdict(Counter)
    for row in attempts:
        breakdown[control[row['id']]]['attempted'] += 1
        breakdown[control[row['id']]][row['status']] += 1
    accepted_ids = {r['id'] for r in accepted}
    return dict(attempted=len(attempts), accepted=len(accepted), retained=len(unique),
                retained_rate=len(unique)/len(attempts), acceptance_rate=len(accepted)/len(attempts),
                statuses=dict(Counter(r['status'] for r in attempts)),
                by_structure={k: dict(v) for k, v in sorted(breakdown.items())},
                unique_targets=len({r['output']['refined_text'] for r in unique}),
                real_fragments=sum(bool(r['meta'].get('is_fragment')) for r in unique),
                asr_no_boundary_full_fallbacks=sum(bool(r.get('asr_selected')) and not r.get('natural_fragment_eligible')
                    and bool(r.get('asr')) and not r['asr'].get('is_fragment', r['asr'].get('meta', {}).get('is_fragment', False)) for r in attempts),
                asr_no_boundary_retained=sum(bool(r.get('asr_selected')) and not r.get('natural_fragment_eligible')
                    and r['id'] in accepted_ids for r in attempts)), control, accepted_ids


def main():
    attempts = rows(OUT / 'attempts.jsonl')
    ids = {r['id'] for r in attempts}
    assert len(ids) == len(attempts) == 300
    old, old_controls, old_ids = summarize(BASE, ids)
    new, new_controls, new_ids = summarize(OUT, ids)
    assert old_controls == new_controls
    seed_files = [p.relative_to(OUT / 'seeds') for p in (OUT / 'seeds').rglob('*.json')]
    seed_equal = {str(p): hashlib.sha256((BASE/'seeds'/p).read_bytes()).hexdigest() ==
                         hashlib.sha256((OUT/'seeds'/p).read_bytes()).hexdigest() for p in seed_files}
    assert seed_equal and all(seed_equal.values())
    record = dict(method='Identical 300 candidate IDs, seed pools, model and control RNG; stochastic teacher outputs may differ. Each subset is re-deduplicated independently in ID order.',
                  baseline=old, current=new, same_control_assignment=True,
                  seed_files_identical=seed_equal,
                  retained_percentage_point_gain=100*(new['retained_rate']-old['retained_rate']),
                  transitions=dict(both_accept=len(old_ids&new_ids), old_only=len(old_ids-new_ids),
                                   new_only=len(new_ids-old_ids), both_reject=len(ids-(old_ids|new_ids))))
    (OUT / 'matched_comparison.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
