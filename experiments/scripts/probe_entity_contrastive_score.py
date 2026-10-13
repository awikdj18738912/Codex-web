"""Offline entity-only contrastive Refiner pilot. Never installs a rule.

No target reference is opened by prepare/infer. Both hypothetical transcripts
are scored under both source prompts to expose copying bias. Scores are not
calibrated correctness probabilities; all threshold exploration is development.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / 'results/entity_context_confirmation_20261012'
OUT = ROOT / 'results/entity_conflict_veto_20261012'
sys.path.insert(0, str(ROOT))
PROMPT = '你是ASR文本纠错助手。保留原意，最小修改，核对实体字形。不要总结、扩写或解释。只返回转录文本。'
THRESHOLDS = [0.1, 0.25, 0.5, 1.0, 2.0]


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'score_plan.json').exists():
        raise FileExistsError('Score pilot is already frozen')
    profiles = json.loads((PREVIOUS / 'verified_backgrounds.json').read_text())['profiles']
    rows = []
    for row in read(PREVIOUS / 'pilot_inputs.jsonl'):
        start = row['context_with_slot'].index('【待核对】')
        source = row['source_text']
        end = start + len(row['observed'])
        assert source[start:end] == row['observed']
        changed = source[:start] + row['target_candidate'] + source[end:]
        backgrounds = [dict(name=p['canonical'], description=profiles[p['canonical']]['description'])
                       for p in row['profiles'] if p['canonical'] in profiles]
        rows.append(dict(id=row['id'], sample_id=row['sample_id'], source=source,
                         changed=changed, start=start, end=end,
                         observed=row['observed'], canonical=row['target_candidate'],
                         backgrounds=backgrounds))
    inputs = OUT / 'score_inputs.jsonl'
    inputs.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write(OUT / 'score_plan.json', dict(cases=len(rows), development_only=True,
          target_reference_at_inference=False, inputs_sha256=sha(inputs),
          background_sha256=sha(PREVIOUS / 'verified_backgrounds.json'),
          entity_module_sha256=sha(ROOT / 'system/entity_audio_repair.py'),
          model='/home/aim0/data/models/ASR/AgenticASR-Refiner', prompt=PROMPT,
          conditions=['context_only', 'official_background'], thresholds=THRESHOLDS,
          rule='Veto only if target mean log probability minus source mean log probability '
               'is below -threshold under BOTH input hypotheses. No new replacements.',
          acceptance='No target hit loss, no total CER increase, at least one fewer '
                     'reference-absent replacement. Explore on development only.'))
    print('PREPARED', len(rows), flush=True)


def infer(device):
    from system.live_qwen_refiner import TransformersRefiner
    plan = json.loads((OUT / 'score_plan.json').read_text())
    assert sha(OUT / 'score_inputs.jsonl') == plan['inputs_sha256']
    assert sha(PREVIOUS / 'verified_backgrounds.json') == plan['background_sha256']
    assert sha(ROOT / 'system/entity_audio_repair.py') == plan['entity_module_sha256']
    model = TransformersRefiner(Path(plan['model']), device, 32)
    model._torch.set_num_threads(4)
    torch = model._torch
    tokenizer = model._tokenizer
    embedding_device = model._model.get_input_embeddings().weight.device
    destination = OUT / 'scores.jsonl'
    done = {(r['id'], r['condition']) for r in read(destination)} if destination.exists() else set()
    successful_forwards = 0

    def score(prompt_source, completion, system):
        nonlocal successful_forwards
        encoded = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt_source}],
            tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors='pt', **model._template_kwargs)
        prompt_ids = encoded['input_ids'][0].to(embedding_device)
        completion_ids = tokenizer(completion, add_special_tokens=False,
                                   return_tensors='pt')['input_ids'][0].to(embedding_device)
        combined = torch.cat((prompt_ids, completion_ids)).unsqueeze(0)
        with torch.inference_mode():
            logits = model._model(input_ids=combined).logits[0, len(prompt_ids)-1:-1].float()
            logs = torch.log_softmax(logits, dim=-1).gather(1, completion_ids.unsqueeze(1))
        successful_forwards += 1
        return dict(mean=logs.mean().item(), total=logs.sum().item(), tokens=len(completion_ids))

    for row in read(OUT / 'score_inputs.jsonl'):
        for condition in plan['conditions']:
            if (row['id'], condition) in done:
                continue
            started = time.perf_counter()
            forwards_before = successful_forwards
            system = plan['prompt']
            if condition == 'official_background' and row['backgrounds']:
                system += '\n资料仅供核对，同名可能有不同身份，资料未提及不构成反证：' + json.dumps(row['backgrounds'], ensure_ascii=False)
            matrix = {}
            for prompt_key in ('source', 'changed'):
                matrix[prompt_key] = {key: score(row[prompt_key], row[key], system)
                                      for key in ('source', 'changed')}
            margins = {key: matrix[key]['changed']['mean'] - matrix[key]['source']['mean']
                       for key in matrix}
            record = dict(id=row['id'], condition=condition, matrix=matrix, margins=margins,
                          forward_calls=successful_forwards-forwards_before, wall_seconds=time.perf_counter()-started)
            with destination.open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print('SCORED', row['id'], condition, margins, flush=True)


def report(*, average_diagnostic=False):
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.entity_audio_repair import normalized
    from system.zh_itn_bridge import ZhITNNormalizer
    itn = ZhITNNormalizer(ROOT / 'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    cc = opencc.OpenCC('t2s')
    def norm(text):
        return normalized(cc.convert(itn.normalize(text).text))
    plan = json.loads((OUT / 'score_plan.json').read_text())
    assert sha(OUT / 'score_inputs.jsonl') == plan['inputs_sha256']
    inputs = {r['id']: r for r in read(OUT / 'score_inputs.jsonl')}
    labels = {r['id']: r for r in read(PREVIOUS / 'pilot_labels.jsonl')}
    scores = read(OUT / 'scores.jsonl')
    assert len(scores) == len({(r['id'], r['condition']) for r in scores}) == len(inputs)*2
    assert {(r['id'], r['condition']) for r in scores} == {(key, arm) for key in inputs for arm in plan['conditions']}
    assert all(set(r['margins']) == {'source', 'changed'} and
               all(math.isfinite(m) for m in r['margins'].values()) for r in scores)
    if average_diagnostic:
        exploration = json.loads((OUT / 'score_average_plan.json').read_text())
        assert exploration['post_score_exploration'] and sha(OUT / 'scores.jsonl') == exploration['scores_sha256']
    baseline = read(PREVIOUS / 'background_asr_outputs.jsonl')
    assert len(baseline) == len({r['id'] for r in baseline}) == 63
    baseline_by_id = {r['id']: r for r in baseline}
    assert set(baseline_by_id) == {r['sample_id'] for r in inputs.values()}
    assert all(r['source'] == baseline_by_id[r['sample_id']]['initial']['text'] for r in inputs.values())
    baseline_metrics = json.loads((PREVIOUS / 'background_asr_summary.json').read_text())['baseline']
    base = ROOT / 'results/large_entity_benchmark_v1_20261010/dataset'
    samples = {r['id']: r for r in read(base / 'samples.jsonl')}
    for folder in ('holdout', 'holdout_news', 'holdout_v2'):
        samples.update({r['id']: r for r in read(ROOT / 'results/entity_confirmation_20261011' / folder / 'samples.jsonl')})
    summaries = []
    details = []
    for condition in plan['conditions']:
        for threshold in plan['thresholds']:
            veto = {}
            for result in scores:
                margins = list(result['margins'].values())
                reject = (sum(margins)/len(margins) <= -threshold if average_diagnostic
                          else all(m <= -threshold for m in margins))
                if result['condition'] != condition or not reject:
                    continue
                r = inputs[result['id']]
                veto.setdefault(r['sample_id'], set()).add((r['start'], r['end'], r['canonical']))
            counter = Counter()
            for r in baseline:
                status = r['baseline']['entity_audio_repair']
                current_changes = status['changes']
                rejected = [c for c in current_changes if (c['start'], c['end'], c['canonical']) in veto.get(r['id'], set())]
                kept = [c for c in current_changes if c not in rejected]
                text = r['initial']['text']
                for c in sorted(kept, key=lambda c:c['start'], reverse=True):
                    assert text[c['start']:c['end']] == c['observed']
                    text = text[:c['start']] + c['canonical'] + text[c['end']:]
                assert (not rejected) == (text == r['baseline']['text'])
                s = samples[r['id']]
                ref = norm(s['reference'])
                expected = Counter(norm(e['canonical']) for e in s['gold_entities'])
                actual = norm(text)
                old = norm(r['baseline']['text'])
                counter.update(samples=1, hits=sum(min(n, actual.count(w)) for w,n in expected.items()),
                               expected_mentions=sum(expected.values()), errors=Levenshtein.distance(ref, actual),
                               reference_chars=len(ref), absent_changes=sum(norm(c['canonical']) not in ref for c in kept),
                               vetoed_changes=len(rejected),
                               lost_reference_present_changes=sum(norm(c['canonical']) in ref for c in rejected),
                               improved=Levenshtein.distance(ref, actual)<Levenshtein.distance(ref, old),
                               regressed=Levenshtein.distance(ref, actual)>Levenshtein.distance(ref, old))
                if rejected:
                    details.append(dict(condition=condition, threshold=threshold, sample_id=r['id'],
                                        reference=s['reference'], before=r['baseline']['text'], after=text,
                                        rejected=rejected))
            counter['CER'] = counter['errors']/counter['reference_chars']
            qualifies = (counter['hits'] >= baseline_metrics['hits']
                         and counter['errors'] <= baseline_metrics['errors']
                         and counter['absent_changes'] < baseline_metrics['absent_changes']
                         and counter['lost_reference_present_changes'] == 0)
            summaries.append(dict(condition=condition, threshold=threshold, metrics=counter,
                                  aggregation='average_diagnostic' if average_diagnostic else 'both_prompts',
                                  qualifies_on_development=qualifies))
    prefix = 'score_average' if average_diagnostic else 'score'
    write(OUT / f'{prefix}_summary.json', summaries)
    write(OUT / f'{prefix}_changes.json', details)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'infer', 'report'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--average-diagnostic', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'infer':
        infer(args.device)
    else:
        report(average_diagnostic=args.average_diagnostic)
