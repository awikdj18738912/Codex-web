"""Development-only confirmation sweep with unchanged real ASR responses.

Does not change the glossary, prompt, production modules or inference output
files. References are used exclusively for scoring after candidate selection.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/large_entity_benchmark_v1_20261010'
OUT = ROOT / 'results/entity_confirmation_20261011'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare():
    sys.path.insert(0, str(BASE / 'full_evaluation/source_snapshot'))
    from system.entity_audio_repair import GenericEntitySelector, verify_entity_audio_changes
    from system.entity_store import EntityStore
    destination = OUT / 'development_replays.jsonl'
    if destination.exists():
        raise FileExistsError('Development replay inputs already frozen')
    definitions = EntityStore(BASE / 'dataset/library/entities.db').list_entities(domain='general')
    selector = GenericEntitySelector(definitions)
    records = [json.loads(line) for line in (BASE / 'full_evaluation/audio.jsonl').read_text().splitlines()]
    assert len(records) == 2410
    OUT.mkdir(parents=True, exist_ok=True)
    for row in records:
        status = row['production_final']['entity_audio_repair']
        if not status['changes']:
            continue
        source = row['initial']['text']
        candidates, terms = selector.select(source)
        assert terms == status['prompt_terms'], row['id']
        prompted = {item['result']['session_id'] for item in row['trace']
                    if item['endpoint'] == '/stream/start' and item['context']}
        replay = next(item['result']['text'] for item in reversed(row['trace'])
                      if item['endpoint'] == '/stream/finish' and item['session_id'] in prompted)
        revised, changes = verify_entity_audio_changes(source, replay, candidates)
        assert revised == row['production_final']['text'], row['id']
        assert {(c['start'], c['end'], c['canonical']) for c in changes} == {
            (c['start'], c['end'], c['canonical']) for c in status['changes']}, row['id']
        with destination.open('a') as stream:
            stream.write(json.dumps(dict(id=row['id'], source=source, replay=replay,
                                        candidates=candidates, prompt_terms=terms), ensure_ascii=False) + '\n')
        print('PREPARED', row['id'], flush=True)
    write(OUT / 'development_manifest.json', dict(
        development_only=True, reference_free_replay_inputs=True,
        original_rows=2410, applied_rows=172, applied_changes=178,
        input_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        selection='All previously applied changes, plus unchanged rows scored from full original cohort',
        protocol='Only reject candidate confirmations; actual ASR prompt and response remain unchanged'))


def permitted(candidate, candidates, mode, margin):
    if mode == 'baseline' or max(len(candidate['observed']), len(candidate['canonical'])) > 2:
        return True
    rivals = [c for c in candidates if (c['start'], c['end']) == (candidate['start'], candidate['end'])
              and c['canonical'] != candidate['canonical']]
    if mode == 'exact_sound':
        return candidate['phonetic_score'] >= 1.0
    if mode == 'shared_character':
        return candidate['character_score'] > 0
    if mode == 'any_rival':
        return not rivals
    if mode == 'no_character_margin' and candidate['character_score'] > 0:
        return True
    return not rivals or candidate['retrieval_score'] - max(c['retrieval_score'] for c in rivals) > margin


def score():
    sys.path.insert(0, str(BASE / 'full_evaluation/source_snapshot'))
    import opencc
    from rapidfuzz.distance import Levenshtein
    from system.entity_audio_repair import normalized, verify_entity_audio_changes
    from system.zh_itn_bridge import ZhITNNormalizer
    itn = ZhITNNormalizer(ROOT / 'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    cc = opencc.OpenCC('t2s')
    cache = {}

    def norm(value):
        if value not in cache:
            cache[value] = normalized(cc.convert(itn.normalize(value).text))
        return cache[value]

    replay_file = OUT / 'development_replays.jsonl'
    manifest = json.loads((OUT / 'development_manifest.json').read_text())
    assert hashlib.sha256(replay_file.read_bytes()).hexdigest() == manifest['input_sha256']
    replays = {r['id']: r for r in (json.loads(line) for line in replay_file.read_text().splitlines())}
    records = [json.loads(line) for line in (BASE / 'full_evaluation/audio.jsonl').read_text().splitlines()]
    refs = {r['sample_id']: r['reference'] for r in json.loads((BASE / 'dataset/references.json').read_text())}
    gold = {r['sample_id']: r for r in json.loads((BASE / 'dataset/sample_entities.json').read_text())}
    variants = [('baseline', 'baseline', 0), ('short_exact_sound', 'exact_sound', 0),
                ('short_shared_character', 'shared_character', 0), ('short_any_rival', 'any_rival', 0)]
    for mode in ('margin', 'no_character_margin'):
        variants.extend((f'short_{mode}_{m:.3f}', mode, m) for m in (0, .025, .05, .075, .10, .15, .20))
    summary = {name: Counter() for name, _, _ in variants}
    interventions = {name: [] for name, _, _ in variants}
    for record in records:
        sid = record['id']; reference = norm(refs[sid])
        expected = Counter(norm(e['canonical']) for e in gold[sid]['gold_entities'])
        prior = record['production_final']['text']; prior_norm = norm(prior)
        prior_hits = sum(min(count, prior_norm.count(name)) for name, count in expected.items())
        prior_errors = Levenshtein.distance(reference, prior_norm)
        for name, mode, margin in variants:
            if sid in replays:
                row = replays[sid]
                _, prior_changes = verify_entity_audio_changes(row['source'], row['replay'], row['candidates'])
                changes = [c for c in prior_changes if permitted(c, row['candidates'], mode, margin)]
                value = row['source']
                for change in sorted(changes, key=lambda c: c['start'], reverse=True):
                    value = value[:change['start']] + change['canonical'] + value[change['end']:]
            else:
                value = record['initial']['text']; changes = []
            compact = norm(value); hits = sum(min(count, compact.count(word)) for word, count in expected.items())
            errors = Levenshtein.distance(reference, compact)
            absent = [c for c in changes if norm(c['canonical']) not in reference]
            summary[name].update(samples=1, hits=hits, expected_mentions=sum(expected.values()),
                errors=errors, reference_chars=len(reference), changes=len(changes),
                absent_changes=len(absent), absent_samples=bool(absent),
                gained_mentions=max(0, hits-prior_hits), lost_mentions=max(0, prior_hits-hits),
                improved_samples=errors < prior_errors, regressed_samples=errors > prior_errors,
                negative_absent_changes=len(absent) if gold[sid]['cohort']=='negative' else 0)
            if value != prior:
                interventions[name].append(dict(id=sid, reference=refs[sid], before=prior, after=value,
                    hits_delta=hits-prior_hits, errors_delta=errors-prior_errors))
    for name, value in summary.items():
        value['CER'] = value['errors'] / value['reference_chars']
        value['entity_recall'] = value['hits'] / value['expected_mentions']
    assert summary['baseline']['hits'] == 2604 and summary['baseline']['errors'] == 819
    write(OUT / 'development_summary.json', summary)
    write(OUT / 'development_changes.json', interventions)
    lines = ['# 短词确认规则开发回放', '',
             '已见2410条数据，仅改变确认过滤，词库、提示和真实ASR响应均未改变。不是独立验证或新请求节省测试。', '',
             '|规则|命中|CER|参考中不存在的修改|丢失命中|改善/退化样本（相对旧原音规则）|',
             '|---|---:|---:|---:|---:|---:|']
    for name, value in summary.items():
        lines.append(f"|{name}|{value['hits']}/2745|{value['CER']:.3%}|{value['absent_changes']}|{value['lost_mentions']}|{value['improved_samples']}/{value['regressed_samples']}|")
    (OUT / 'DEVELOPMENT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'score'])
    if parser.parse_args().stage == 'prepare': prepare()
    else: score()
