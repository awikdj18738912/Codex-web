#!/usr/bin/env python3
"""Prepare a deterministic, heuristic fresh sample for prompt scope validation."""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/aim0/桌面/tan/ASR/Codex-web')
OUT = ROOT / 'results/qwen38_prompt_scope_validation_20261022'
SFT = Path('/home/aim0/data/datasets/ASR/refiner_zh_100k_finalaudit_20261018/sft_reviewed_v5_20261021/train_sft.json')
SPLIT = SFT.with_name('split_manifest.jsonl')
EXCLUDE_FILES = [
    ROOT / 'results/teacher_model_comparison_1000_20261022/sample_1000.jsonl',
    ROOT / 'results/targeted_repair_pilot_20261022/pilot_sample_100.jsonl',
    ROOT / 'results/refiner_training_pair_audit_200_20261022/gpt_review_200.jsonl',
]
SEED = 20261022

def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def get_triplet(row):
    conv = row['conversations']
    human = next(x['value'] for x in conv if x['from'] == 'human')
    target = next(x['value'] for x in conv if x['from'] == 'gpt')
    return human, target

def heuristic_categories(text, target):
    cats = []
    # Candidate cues only; these do not establish that the label is correct.
    correction = re.compile(r'(?:不是|不对|应该是|准确地说|我的意思是|改成|更准确(?:地)?(?:说)?|不，|啊不|等等.{0,12}(?:不是|应该))')
    cues = len(correction.findall(text))
    if cues >= 2 or len(re.findall(r'(?:不是|不对|改成|应该是)', text)) >= 2:
        cats.append('multiple_revisions_candidate')
    elif cues == 1 or re.search(r'(?:不是|不对|改成|应该是)', text):
        cats.append('single_revision_candidate')
    if re.search(r'[A-Za-z]{2,}', text) and re.search(r'[\u3400-\u9fff]', text):
        cats.append('code_switch_candidate')
    # Mid-clause false starts followed by a restart/correction are a weak proxy
    # for accidental sentence breaks; punctuation and lexical cues are noisy.
    if re.search(r'[，；。！？,.!?…]{1,}.{0,16}(?:不是|不对|啊，不|应该是|等等)', text) or re.search(r'(?:啊|嗯|那个)[，,。.]{1,}\s*(?:但|可是|不过)', text):
        cats.append('unexpected_break_candidate')
    if text == target:
        cats.append('source_equals_target_control')
    return cats

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.load(SFT.open(encoding='utf-8'))
    split_by_index = {}
    with SPLIT.open(encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            if m.get('split') == 'train':
                split_by_index[int(m['output_index'])] = m
    if len(rows) != 84875 or len(split_by_index) != 84875:
        raise ValueError(f'Unexpected train alignment sizes: SFT={len(rows)}, manifest={len(split_by_index)}')
    exclusions = set()
    exclude_counts = {}
    for path in EXCLUDE_FILES:
        ids = {str(r['id']) for r in read_jsonl(path) if r.get('id') is not None}
        exclude_counts[str(path.relative_to(ROOT))] = len(ids)
        exclusions.update(ids)
    candidates = []
    source_hashes = {}
    for index, row in enumerate(rows):
        m = split_by_index[index]
        source, target = get_triplet(row)
        sid = str(m['source_id'])
        if sid in exclusions:
            continue
        ih, th = sha(source), sha(target)
        if ih != m['input_sha256'] or th != m['target_sha256']:
            raise ValueError(f'hash mismatch at output_index={index}, id={sid}')
        candidates.append({'id': sid, 'scene': m['scene'], 'input': source,
                           'existing_target': target, '_index': index,
                           '_cats': heuristic_categories(source, target),
                           '_ih': ih, '_th': th})
        source_hashes.setdefault('train_sft_sha256', None)
    rng = random.Random(SEED)
    random_pool = candidates[:]
    rng.shuffle(random_pool)
    random_selected = random_pool[:100]
    used = {x['id'] for x in random_selected}
    remaining = [x for x in candidates if x['id'] not in used]

    # Select each candidate stratum by scene-balanced round-robin over
    # deterministically shuffled per-scene queues. A record can occur once only.
    strata = ['multiple_revisions_candidate', 'single_revision_candidate',
              'code_switch_candidate', 'unexpected_break_candidate',
              'source_equals_target_control']
    selected_special = []
    deficits = {}
    for category in strata:
        queues = defaultdict(list)
        for item in remaining:
            if item['id'] not in used and category in item['_cats']:
                queues[item['scene']].append(item)
        scenes = sorted(queues)
        for scene in scenes:
            queues[scene].sort(key=lambda x: (x['_index'], x['id']))
            rng.shuffle(queues[scene])
        picked = []
        # Round robin scene coverage, then continue until 20 or exhausted.
        while len(picked) < 20 and any(queues[s] for s in scenes):
            for scene in scenes:
                if queues[scene] and len(picked) < 20:
                    item = queues[scene].pop()
                    if item['id'] not in used:
                        picked.append(item)
                        used.add(item['id'])
        deficits[category] = max(0, 20 - len(picked))
        for item in picked:
            item['_stratum'] = category
            selected_special.append(item)
    # Fill shortages from remaining eligible rows, retaining clear provenance.
    if len(selected_special) < 100:
        filler = [x for x in remaining if x['id'] not in used]
        rng.shuffle(filler)
        for item in filler[:100-len(selected_special)]:
            item['_stratum'] = 'shortage_fill_random'
            used.add(item['id'])
            selected_special.append(item)
    sample = []
    for i, item in enumerate(random_selected + selected_special, 1):
        sample.append({'case_no': i, 'id': item['id'], 'scene': item['scene'],
                       'input': item['input'], 'existing_target': item['existing_target'],
                       'selection_stratum': 'remaining_train_random' if i <= 100 else item['_stratum']})
    if len(sample) != 200 or len({x['id'] for x in sample}) != 200:
        raise ValueError(f'Expected 200 unique rows; got {len(sample)}')
    with (OUT / 'fresh_sample_200.jsonl').open('w', encoding='utf-8') as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    # Hash whole source files via streaming to avoid a second in-memory copy.
    file_hashes = {}
    for label, path in [('train_sft_sha256', SFT), ('split_manifest_sha256', SPLIT)]:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        file_hashes[label] = h.hexdigest()
    scene_dist = Counter(x['scene'] for x in sample)
    stratum_dist = Counter(x['selection_stratum'] for x in sample)
    manifest = {
        'purpose': 'fresh prompt-scope development/validation sample; not an independent final test set',
        'seed': SEED, 'sample_size': len(sample), 'train_rows': len(rows),
        'excluded_unique_id_count': len(exclusions), 'excluded_id_count_by_file': exclude_counts,
        'excluded_ids_union_sha256': sha('\n'.join(sorted(exclusions))),
        'source_files': {'train_sft': str(SFT), 'split_manifest': str(SPLIT), **file_hashes},
        'alignment': 'Every included output_index was joined to split_manifest train and input/target SHA-256 values were checked against the exact human/gpt conversation strings before sampling.',
        'selected_input_target_hashes': [{'case_no': i+1, 'id': x['id'], 'output_index': x['_index'],
                                          'input_sha256': x['_ih'], 'target_sha256': x['_th']} for i,x in enumerate(random_selected + selected_special)],
        'category_counts': dict(Counter(cat for x in sample for cat in next((z['_cats'] for z in random_selected + selected_special if z['id']==x['id']), []))),
        'special_stratum_counts': {k: sum(1 for x in sample if x['selection_stratum']==k) for k in strata},
        'category_shortfalls': deficits,
        'selection_stratum_counts': dict(stratum_dist), 'scene_counts': dict(scene_dist),
        'heuristics_note': 'Category cues are noisy selection heuristics only; no target was used to judge or revise any example. existing_target is retained solely as evidence and must not be sent to the teacher.',
        'independence_note': 'This sample is for development/prompt validation and should not be reported as an independent final test set. Shared source dataset, prior project exposure, and heuristic selection can create leakage or selection bias.'
    }
    with (OUT / 'fresh_sample_manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write('\n')

if __name__ == '__main__':
    main()
