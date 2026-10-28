"""Export only the completed catalogue-screened ledger; keep old splits and lineage."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from run_error_loop import load_source, digest

P = Path(__file__).resolve().parent
STORE = Path('/home/aim0/data/ASR/dataset_error_loop_20261024')
if not (STORE / 'all_complete.json').exists():
    raise RuntimeError('Error catalogue pass is incomplete; training export blocked')
revisions = [json.loads(line) for line in (STORE / 'results.jsonl').open()]
rows = list({r['record_id']: r for r in revisions}.values())
assert len(rows) == 99852 and len({r['record_id'] for r in rows}) == 99852
expected_ids = {r['record_id'] for r in load_source()}
assert {r['record_id'] for r in rows} == expected_ids, 'result/source ID mismatch'
assert json.loads((STORE / 'all_complete.json').read_text())['source_ids_sha256'] == digest(sorted(expected_ids))
dest = STORE / 'candidate_sft'
dest.mkdir(exist_ok=True)
old_manifest = json.loads((P.parent / 'refiner_dataset_repair_20261023/manifest.json').read_text())
old_data = Path(old_manifest['original_data'])
original = json.loads((old_data / 'train_sft.json').read_text())[0]
system = original.get('system', '')
version = json.loads((STORE / 'all_complete.json').read_text())['catalog_version']
override_path = STORE / 'primary_review_overrides.jsonl'
overrides = [json.loads(line) for line in override_path.open()] if override_path.exists() else []
blocked = {(r['record_id'], r['candidate_pair_sha256']) for r in overrides}
accepted = [r for r in rows if r['decision'] == 'catalog_screened_candidate'
            and r.get('checked_catalog_version', 0) == version
            and (r['record_id'], digest(r['candidate_pair'])) not in blocked]
val = {hashlib.sha256(r['candidate_pair'][k].encode()).hexdigest()
       for r in accepted if r['original_split'] in ['val', 'validation'] for k in ['input', 'target']}
data = {'train': [], 'val': []}
lineage, excluded = [], []
seen = {'train': set(), 'val': set()}
for r in accepted:
    pair = r['candidate_pair']
    split = 'val' if r['original_split'] in ['val', 'validation'] else 'train'
    ih = hashlib.sha256(pair['input'].encode()).hexdigest()
    th = hashlib.sha256(pair['target'].encode()).hexdigest()
    if (split == 'train' and (ih in val or th in val)) or ih in seen[split]:
        excluded.append({'record_id': r['record_id'], 'reason': 'cross_split_collision_or_duplicate_input'})
        continue
    seen[split].add(ih)
    item = {'conversations': [{'from': 'human', 'value': pair['input']}, {'from': 'gpt', 'value': pair['target']}]}
    if system:
        item['system'] = system
    data[split].append(item)
    lineage.append({'record_id': r['record_id'], 'replacement_id': r.get('replacement_id'),
                    'split': split, 'origin': r['origin'], 'input_sha256': ih, 'target_sha256': th,
                    'original_input_sha256': r['original_input_sha256'],
                    'original_target_sha256': r['original_target_sha256']})
for split in ['train', 'val']:
    (dest / f'{split}_sft.json').write_text(json.dumps(data[split], ensure_ascii=False))
for name, values in [('lineage', lineage), ('excluded_duplicates', excluded),
                     ('pending_review', [r for r in rows if r['decision'] != 'catalog_screened_candidate'
                                         or (r['record_id'], digest(r.get('candidate_pair'))) in blocked])]:
    with (dest / (name + '.jsonl')).open('w') as f:
        for r in values:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
(dest / 'dataset_info.json').write_bytes((old_data / 'dataset_info.json').read_bytes())
report = {'status': 'teacher_screened_candidates_pending_independent_audit_and_training_ablation',
          'primary_review_overrides': len(blocked),
          'by_split': {k: len(v) for k, v in data.items()}, 'origins': dict(Counter(r['origin'] for r in accepted)),
          'formal_quality_guarantee': False, 'old_training_files_overwritten': False}
(dest / 'quality_status.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False))
