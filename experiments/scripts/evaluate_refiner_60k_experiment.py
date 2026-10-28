"""Compare frozen 60k/full checkpoints; candidate references are never human gold.

This command runs Refiner inference on CUDA only when explicitly invoked. The
RBT3 gate is loaded read-only on CPU and uses its frozen development threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import json
from pathlib import Path
import re

from evaluate_refiner13_qwen import run_model, metric_text, levenshtein, BASE

ROOT = Path(__file__).resolve().parents[2]
OLD_DATA = Path('/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02')
OLD_MODEL = Path('/home/aim0/data/ASR/training/qwen25_05b_refiner_noC02_full_20261027')
OLD_GATE = Path('/home/aim0/data/ASR/training/refiner_noC02_gate_rbt3_20261027')
OLD_CACHE = ROOT / 'results/refiner13_no_C02_20261027/evaluation/no_C02_full_test.jsonl'
AUDIO_RAW = ROOT / 'results/audio1_refiner_comparison_20261027/asr_raw.json'
ENTITY = re.compile(r'__ENTITY_\d+__|[A-Za-z][A-Za-z0-9._/+:#-]*|\d+(?:[.,:/%-]\d+)*')


def read(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def normalize(row):
    out = dict(row)
    out['id'] = str(row['id'])
    out['input'] = row.get('input', row.get('text'))
    if not isinstance(out['input'], str):
        raise ValueError('Missing input/text: ' + out['id'])
    out.setdefault('category', row.get('original_category', 'unknown'))
    out.setdefault('source', 'unknown')
    return out


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(path):
    files = sorted({*path.glob('*.safetensors'), *path.glob('*.bin'),
                    *path.glob('*index.json'), *path.glob('*config.json'),
                    *path.glob('tokenizer*'), *path.glob('vocab*'), *path.glob('merges*')})
    if not any(p.suffix in ('.safetensors', '.bin') for p in files):
        raise ValueError('No full checkpoint weights: ' + str(path))
    return {p.name: file_hash(p) for p in files if p.is_file()}


def inference_identity(model, adapter=None):
    identity = {'model': str(model.resolve()), 'model_files_sha256': model_fingerprint(model),
                'adapter': str(adapter.resolve()) if adapter else None,
                'adapter_files_sha256': None}
    if adapter:
        if not (adapter / 'adapter_config.json').exists() or not any(
                (adapter / filename).exists() for filename in ('adapter_model.safetensors', 'adapter_model.bin')):
            raise ValueError('Missing LoRA adapter config/weights: ' + str(adapter))
        identity['adapter_files_sha256'] = model_fingerprint(adapter)
    return identity


def md_cell(value):
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', '<br>')


def descriptive(rows):
    return {'n': len(rows), 'changed_count': sum(r['prediction'] != r['input'] for r in rows),
            'label_status': 'seen-training diagnostic; no independent accuracy metrics'}


def verify_frozen(data):
    manifest = read(data / 'manifest.json')
    for name, digest in manifest.get('files_sha256', {}).items():
        path = data / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen dataset checksum mismatch: ' + str(path))
    rows = [normalize(r) for r in read(data / 'split_audit.jsonl')]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate dataset IDs')
    return manifest, rows


def no_train_leak(train, tests):
    # Cross-field and normalized matches: train input OR reference against test
    # input OR reference. The original pair is retained in the audit on failure.
    seen = {}
    for r in train:
        for field in ('input', 'reference'):
            if r.get(field):
                seen.setdefault(metric_text(r[field]), (r['id'], field))
    overlaps = []
    for name, rows in tests.items():
        for r in rows:
            for field in ('input', 'reference'):
                value = r.get(field)
                if value and metric_text(value) in seen:
                    overlaps.append({'test': name, 'id': r['id'], 'field': field,
                                     'train': seen[metric_text(value)]})
    return overlaps


def balanced(source):
    """Same source-split clean derivation/conflict removal as pregate.prepare."""
    candidates = []
    for r in source:
        candidates.append(dict(r, label=int(r['input'] != r['reference']), derived_clean=False))
        if r['input'] != r['reference']:
            candidates.append(dict(r, id=r['id'] + ':clean_reference', input=r['reference'],
                                   label=0, derived_clean=True))
    groups = defaultdict(list)
    for r in candidates:
        groups[r['input']].append(r)
    accepted = []
    rejected = 0
    for group in groups.values():
        if len({r['split'] for r in group}) > 1 or len({r['label'] for r in group}) > 1:
            rejected += len(group)
        else:
            accepted.append(min(group, key=lambda r: r['derived_clean']))
    test = [r for r in accepted if r['split'] == 'test']
    pos = [r for r in test if r['label']]
    clean = [r for r in test if not r['label'] and not r['derived_clean']]
    derived = sorted([r for r in test if r['derived_clean']],
                     key=lambda r: hashlib.sha256(('balanced20261027:' + r['id']).encode()).hexdigest())
    needed = len(pos) - len(clean)
    if needed < 0 or needed > len(derived):
        raise ValueError('Cannot create equal positive/clean test retaining original inputs')
    return pos + clean + derived[:needed], {'rejected_conflicting_or_cross_split_rows': rejected}


def verified_cache(path, rows):
    if not path.exists():
        return []
    cached = read(path)
    by_id = {r['id']: r for r in cached}
    if len(by_id) != len(cached) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Cache IDs differ from frozen test: ' + str(path))
    for r in rows:
        if any(by_id[r['id']][k] != r[k] for k in ('input', 'reference')):
            raise ValueError('Cache text mismatch: ' + r['id'])
    return [by_id[r['id']] for r in rows]


def score(rows):
    if not rows:
        return {'n': 0}
    def group_score(group):
        n = len(group)
        identity = [r for r in group if r['input'] == r['reference']]
        return {'n': n, 'exact_match_rate': sum(r['prediction'] == r['reference'] for r in group) / n,
            'normalized_char_difference_rate': sum(levenshtein(metric_text(r['prediction']), metric_text(r['reference'])) for r in group) / max(1, sum(len(metric_text(r['reference'])) for r in group)),
            'changed_count': sum(r['prediction'] != r['input'] for r in group),
            'identity_count': len(identity), 'identity_changed_count': sum(r['prediction'] != r['input'] for r in identity),
            'entity_sequence_mismatch_to_input': sum(ENTITY.findall(r['prediction']) != ENTITY.findall(r['input']) for r in group),
            'entity_sequence_mismatch_to_reference': sum(ENTITY.findall(r['prediction']) != ENTITY.findall(r['reference']) for r in group),
            'english_sequence_mismatch_to_input': sum(re.findall(r'[A-Za-z][A-Za-z0-9._/+:#-]*', r['prediction']) != re.findall(r'[A-Za-z][A-Za-z0-9._/+:#-]*', r['input']) for r in group),
            'number_sequence_mismatch_to_input': sum(re.findall(r'\d+(?:[.,:/%-]\d+)*', r['prediction']) != re.findall(r'\d+(?:[.,:/%-]\d+)*', r['input']) for r in group)}
    summary = group_score(rows)
    for field in ('category', 'source', 'scene'):
        summary['by_' + field] = {str(c): group_score([r for r in rows if r.get(field, 'unknown') == c]) for c in sorted({r.get(field, 'unknown') for r in rows}, key=str)}
    return summary


def gate_predict(directory, rows):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    summary = read(directory / 'summary.json')
    threshold = float(summary['threshold_selected_on_dev'])
    weights = next((directory / n for n in ('gate.pt', 'model.pt') if (directory / n).exists()), None)
    if weights is None:
        raise FileNotFoundError('No gate.pt/model.pt in ' + str(directory))
    tokenizer = AutoTokenizer.from_pretrained(directory / 'tokenizer', local_files_only=True)
    config = AutoConfig.from_pretrained(directory / 'encoder_config', local_files_only=True)
    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
            self.head = torch.nn.Sequential(torch.nn.Dropout(.1), torch.nn.Linear(config.hidden_size, 2))
        def forward(self, **inputs):
            x = self.encoder(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).to(x.dtype)
            return self.head((x * mask).sum(1) / mask.sum(1).clamp_min(1))
    torch.set_num_threads(8)
    model = Gate().eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    result = []
    max_len = min(int(config.max_position_embeddings), 512)
    for start in range(0, len(rows), 64):
        batch = rows[start:start + 64]
        lengths = [len(tokenizer(r['input'], truncation=False)['input_ids']) for r in batch]
        encoded = tokenizer([r['input'] for r in batch], padding=True, truncation=True,
                            max_length=max_len, return_tensors='pt')
        with torch.inference_mode():
            probs = model(**encoded).softmax(-1)[:, 1].tolist()
        result.extend({'id': r['id'], 'probability': p if length <= max_len else 1.,
                       'route_to_refiner': bool(p >= threshold or length > max_len),
                       'overflow_forced_route': length > max_len} for r, p, length in zip(batch, probs, lengths))
    return result, {'weights': str(weights), 'threshold_selected_on_dev': threshold, 'device': 'cpu'}


def edits(before, after):
    return [{'operation': op, 'before': before[a:b], 'after': after[c:d],
             'long_deletion_suspicion_heuristic_only': op in ('delete', 'replace') and b - a >= 12 and d - c < (b - a) / 2}
            for op, a, b, c, d in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if op != 'equal']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lora-model', type=Path, help='Optional same-data LoRA adapter over the frozen Qwen2.5-0.5B base')
    p.add_argument('--review-sample', type=Path)
    p.add_argument('--gate-model', type=Path, default=OLD_GATE)
    p.add_argument('--inference-only', action='store_true', help='Write all suite predictions without gate evaluation')
    p.add_argument('--reuse-inference', action='store_true', help='Require and verify completed inference cache; never run Refiner')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, source = verify_frozen(args.data)
    _, old_source = verify_frozen(OLD_DATA)
    old_test = [r for r in old_source if r['split'] == 'test']
    if len(old_test) != 1757:
        raise ValueError('Expected identical frozen old 1757 test')
    new_balanced, balance_audit = balanced(source)
    # Read the exact prior 3056 mix; join original source categories and references.
    old_by_id = {r['id']: r for r in old_source}
    old_balanced = []
    for row in read(OLD_GATE / 'balanced_test_predictions.jsonl'):
        base_id = row['id'].removesuffix(':clean_reference')
        origin = old_by_id[base_id]
        expected = origin['reference'] if row.get('derived_clean') else origin['input']
        if row['text'] != expected or origin['split'] != 'test':
            raise ValueError('Old gate mix/source mismatch: ' + row['id'])
        old_balanced.append(dict(origin, id=row['id'], input=row['text'], derived_clean=row.get('derived_clean', False), label=row['label']))
    if len(old_balanced) != 3056:
        raise ValueError('Expected old balanced3056')
    suites = {'old1757': old_test, 'new_test': [r for r in source if r['split'] == 'test'],
              'old_balanced3056': old_balanced, 'new_balanced_50_50': new_balanced}
    previous_diagnostics=ROOT/'results/refiner13_no_C02_20261027/evaluation/prior_71_full_comparison.jsonl'
    if previous_diagnostics.exists():
        historical=read(previous_diagnostics)
        if len(historical)!=71:raise ValueError('Prior diagnostic set count changed')
        suites['prior71_seen_clean_diagnostics']=[dict(id='prior71:'+r['id'],
            category=r.get('category','historical_diagnostic'),source=r.get('source','historical_diagnostic'),
            input=r['input'],reference=r['input'],previously_inspected=True) for r in historical]
    review = args.review_sample
    if review is None:
        review = next((q for q in (args.data / 'review_reserved1200/sample.jsonl', args.data / 'review_reserved1200/sample.json') if q.exists()), None)
    if review:
        value = read(review)
        review_rows = [normalize(r) for r in (value if isinstance(value, list) else value['samples'])]
        new_train = [r for r in source if r['split'] == 'train']
        old_train = [r for r in old_source if r['split'] == 'train']
        training_texts = {metric_text(r[field]) for r in old_train + new_train
                          for field in ('input', 'reference') if r.get(field)}
        old_split_by_id = {r['id']: r['split'] for r in old_source}
        independent, diagnostic = [], []
        for r in review_rows:
            r['prior_old_split'] = old_split_by_id.get(r['id'], 'not_in_old_frozen_split')
            r['independent_blind_test'] = False  # Review selection/old use already inspected.
            if any(metric_text(r[field]) in training_texts for field in ('input', 'reference') if r.get(field)):
                diagnostic.append(r)
            else:
                independent.append(r)
        suites['review_new_holdout'] = independent
        suites['review_seen_training_diagnostic'] = diagnostic
    if AUDIO_RAW.exists():
        suites['audio1'] = [dict(id='audio1', category='real_audio_no_human_reference', source=str(AUDIO_RAW), input=read(AUDIO_RAW)['text'])]
    leaks = no_train_leak([r for r in source if r['split'] == 'train'], {k: v for k, v in suites.items() if k != 'review_seen_training_diagnostic'})
    write(args.output / 'leak_audit.json', {'overlaps': leaks, 'normalization': 'NFKC, remove punctuation/whitespace; both input/reference fields'})
    if leaks:
        raise ValueError(f'New train overlaps evaluation text ({len(leaks)}); see leak_audit.json')
    # Deduplicate identical inference inputs across suites and aliases, one load/model.
    unique = {}
    lookup = {}
    for suite, rows in suites.items():
        for r in rows:
            key = r['input']
            if key not in unique:
                unique[key] = dict(r, id='eval:' + str(len(unique)), reference=r.get('reference', r['input']))
            lookup[suite, r['id']] = unique[key]['id']
    inputs_path = args.output / 'inference_inputs.jsonl'
    expected_inputs = list(unique.values())
    input_hash = hashlib.sha256(json.dumps(expected_inputs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    prompt_hash = hashlib.sha256(manifest['system_prompt'].encode()).hexdigest()
    cached = verified_cache(OLD_CACHE, old_test)
    old_cached_by_input = {r['input']: r for r in cached}
    predictions = {}
    specs = [('old_full', OLD_MODEL, None), ('new_full', args.model, None)]
    if args.lora_model:
        specs.append(('new_lora', BASE, args.lora_model))
    request_path = args.output / 'inference_request.json'
    request = {'cache_schema': 2, 'models': [name for name, _, _ in specs],
               'lora_adapter': str(args.lora_model.resolve()) if args.lora_model else None,
               'input_content_sha256': input_hash, 'system_prompt_sha256': prompt_hash}
    if args.reuse_inference:
        previous_request = read(request_path)
        if previous_request.get('status') != 'complete' or any(previous_request.get(k) != v for k, v in request.items()):
            raise ValueError('Inference request mismatch (including explicit LoRA presence/absence) or unfinished cache')
        if read(inputs_path) != expected_inputs:
            raise ValueError('Inference input cache differs from requested suites')
    else:
        write(request_path, dict(request, status='running'))
        jsonl(inputs_path, expected_inputs)
    for name, model, adapter in specs:
        model_identity = inference_identity(model, adapter)
        path = args.output / (name + '_all_inference.jsonl')
        completion_path = args.output / (name + '_inference_complete.json')
        identity = dict(model_identity, input_content_sha256=input_hash, system_prompt_sha256=prompt_hash)
        if args.reuse_inference:
            complete = read(completion_path)
            if complete.get('status') != 'complete' or any(complete.get(k) != v for k, v in identity.items()):
                raise ValueError('Unfinished or stale model/input inference cache: ' + name)
            if complete['predictions_sha256'] != file_hash(path) or complete['inputs_file_sha256'] != file_hash(inputs_path):
                raise ValueError('Inference cache file corrupted: ' + name)
            records = read(path)
            if len(records) != len(expected_inputs) or any(
                    any(got.get(k) != expected.get(k) for k in ('id', 'input', 'reference'))
                    or not isinstance(got.get('prediction'), str)
                    for got, expected in zip(records, expected_inputs)):
                raise ValueError('Inference cache rows misaligned: ' + name)
            predictions[name] = {r['id']: r['prediction'] for r in records}
            continue
        write(completion_path, dict(identity, status='running'))
        pending = [r for text, r in unique.items() if name != 'old_full' or text not in old_cached_by_input]
        if pending:
            run_model(name, model, adapter, pending, manifest['system_prompt'], path)
            inferred = {r['id']: r['prediction'] for r in read(path)}
        else:
            inferred = {}
        predictions[name] = {r['id']: old_cached_by_input[text]['prediction'] if name == 'old_full' and text in old_cached_by_input else inferred[r['id']] for text, r in unique.items()}
        # Publish a full aligned cache only after every actual inference completed.
        jsonl(path, [dict(r, prediction=predictions[name][r['id']]) for r in expected_inputs])
        write(path.with_suffix('.summary.json'), {'model': name, 'sample_count': len(expected_inputs),
            'status': 'inference log only; see scores.json for valid suite metrics',
            'warning': 'Mixed suites contain unlabelled/diagnostic rows; no aggregate accuracy'})
        write(completion_path, dict(identity, status='complete', predictions_sha256=file_hash(path),
                                   inputs_file_sha256=file_hash(inputs_path)))
    if not args.reuse_inference:
        write(request_path, dict(request, status='complete'))
    scores = {'requested_models': request['models'], 'lora_adapter': request['lora_adapter'], 'reference_status': 'machine candidate references, not human gold; no audio accuracy claim',
              'balance_audit': balance_audit, 'old_cache': str(OLD_CACHE) if cached else None, 'suites': {}, 'gates': {}, 'audio1_reused': str(AUDIO_RAW) if AUDIO_RAW.exists() else None,
              'review_sample': str(review) if review else None, 'human_review_status': 'not completed by this command'}
    for suite, rows in suites.items():
        comparisons = []
        diagnostic = suite == 'review_seen_training_diagnostic'
        scores['suites'][suite] = {'original_asr': score([dict(r, prediction=r['input']) for r in rows if 'reference' in r])} if suite != 'audio1' and not diagnostic else {}
        for name, pred in predictions.items():
            result = [dict(r, prediction=pred[lookup[suite, r['id']]]) for r in rows]
            jsonl(args.output / (suite + '_' + name + '.jsonl'), result)
            labeled = [r for r in result if 'reference' in r]
            scores['suites'][suite][name] = descriptive(result) if diagnostic else score(labeled) if suite != 'audio1' else {'n': len(rows), 'label_status': 'no human reference; descriptive comparison only'}
        for r in rows:
            outputs = {name: pred[lookup[suite, r['id']]] for name, pred in predictions.items()}
            changes = {name: edits(r['input'], value) for name, value in outputs.items()}
            comparisons.append(dict(r, **outputs, changes=changes,
                                    old_changes=changes['old_full'], new_changes=changes['new_full']))
        jsonl(args.output / (suite + '_comparison.jsonl'), comparisons)
        md = ['# ' + suite + ' 逐句对照', '', '参考为机器候选，需人审；删除长span仅启发式疑点，不是确定错误。', '']
        for r in comparisons:
            if all(r[name] == r['input'] for name in predictions) and suite != 'audio1':
                continue
            md += ['## ' + r['id'], '', f"类别：{r['category']}；来源：{r['source']}", '',
                   '| 版本 | 文本 |', '| --- | --- |', '| 原ASR | ' + md_cell(r['input']) + ' |',
                   '| 候选参考 | ' + md_cell(r.get('reference', '无')) + ' |']
            names = {'old_full': '旧全量', 'new_full': '新全量', 'new_lora': '新LoRA'}
            md += ['| ' + names[name] + ' | ' + md_cell(r[name]) + ' |' for name in predictions]
            md += ['', '改动列表：', '', '```json', json.dumps(r['changes'], ensure_ascii=False, indent=2), '```', '']
        (args.output / (suite + '_comparison.md')).write_text('\n'.join(md))
    gate_directories = {} if args.inference_only else {'old_gate': OLD_GATE}
    if not args.inference_only and args.gate_model.resolve() != OLD_GATE.resolve():
        gate_directories['selected_gate'] = args.gate_model
    for gate_name, directory in gate_directories.items():
        scores['gates'][gate_name] = {'directory': str(directory), 'suites': {}}
        for suite in ('old_balanced3056', 'new_balanced_50_50'):
            rows = suites[suite]
            gate_train = [normalize(r) for r in read(directory / 'train.jsonl')]
            gate_overlap = no_train_leak(gate_train, {suite: rows})
            write(args.output / (gate_name + '_' + suite + '_train_leak_audit.json'), {'overlaps': gate_overlap})
            if gate_overlap:
                raise ValueError('Gate train overlaps holdout: ' + gate_name + '/' + suite)
            routes, metadata = gate_predict(directory, rows)
            route_by_id = {r['id']: r for r in routes}
            scores['gates'][gate_name].update(metadata)
            stats = {'n': len(rows), 'refiner_calls': sum(r['route_to_refiner'] for r in routes), 'by_refiner': {}}
            positives = [r for r in rows if r['input'] != r['reference']]
            negatives = [r for r in rows if r['input'] == r['reference']]
            stats['need_refine_recall_percent'] = 100 * sum(route_by_id[r['id']]['route_to_refiner'] for r in positives) / len(positives) if positives else None
            stats['clean_skip_rate_percent'] = 100 * sum(not route_by_id[r['id']]['route_to_refiner'] for r in negatives) / len(negatives) if negatives else None
            for name, pred in predictions.items():
                outputs = []
                for r in rows:
                    route = route_by_id[r['id']]
                    actual = pred[lookup[suite, r['id']]]
                    outputs.append(dict(r, **{k: v for k, v in route.items() if k != 'id'}, refiner_prediction=actual,
                                        prediction=actual if route['route_to_refiner'] else r['input']))
                stats['by_refiner'][name] = dict(score(outputs),
                    could_match_reference_but_skipped=sum(not r['route_to_refiner'] and r['input'] != r['reference'] and r['refiner_prediction'] == r['reference'] for r in outputs),
                    weak_positive_missed=sum(not r['route_to_refiner'] and r['input'] != r['reference'] for r in outputs))
                jsonl(args.output / (gate_name + '_' + suite + '_' + name + '.jsonl'), outputs)
            scores['gates'][gate_name]['suites'][suite] = stats
    scores['status'] = 'inference_complete_gate_pending' if args.inference_only else 'complete'
    write(args.output / 'scores.json', scores)
    summary_md = ['# 同输入精修指标对照', '', '机器候选参考；已见训练诊断和无参考音频不报告独立匹配率。', '',
                  '| 数据集 | 版本 | 样本数 | 候选精确匹配率 | 归一字符差异率 | 改动数 |',
                  '| --- | --- | ---: | ---: | ---: | ---: |']
    for suite, models in scores['suites'].items():
        for name, values in models.items():
            exact = values.get('exact_match_rate')
            difference = values.get('normalized_char_difference_rate')
            summary_md.append('| ' + ' | '.join([suite, name, str(values.get('n', 0)),
                f'{exact:.2%}' if exact is not None else '不适用',
                f'{difference:.2%}' if difference is not None else '不适用', str(values.get('changed_count', '—'))]) + ' |')
    (args.output / 'scores_comparison.md').write_text('\n'.join(summary_md) + '\n')
    (args.output / 'README.md').write_text('# 60k精修评估\n\n' + ('待完成：门控评估，本次只完成推理。\n\n' if args.inference_only else '完成：冻结dev阈值的input-only RBT3路由与实际精修组合。\n\n') + '完成：同旧1757/新test、旧3056/新五五均衡的同输入旧新全量及可选新LoRA推理；数字/英文/占位符检查；原始audio1转写复用（若文件存在）；逐句改动对照。\n\n缺少：机器候选参考的人审、真实音频人工逐字参考、真实ASR外部盲测及端到端时延。候选匹配率和归一字符差异率不是人工gold准确率。删除长span标记仅供人工排查。未训练/修改门控，未启动ASR。review中接触过旧/新train的样本仅作seen-training诊断，不计独立准确率；review_new_holdout排除旧/新train。旧dev/test保持已有用途，不声称盲测。未提供不代表已完成1200条审核。\n')
    print(json.dumps({'status': scores['status'], 'output': str(args.output), 'suite_counts': {k: len(v) for k, v in suites.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
