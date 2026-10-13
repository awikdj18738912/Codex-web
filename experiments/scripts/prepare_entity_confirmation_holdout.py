"""Freeze unused real audio before evaluating a confirmation rule.

Uses the existing 1,346-entry glossary, never reads ASR hypotheses. Annotation
is reference surface matching for CV and official source NER for AISHELL.
"""
import csv
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

import opencc
from prepare_large_entity_aishell import historical_ids

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/large_entity_benchmark_v1_20261010'
OUT = ROOT / 'results/entity_confirmation_20261011/holdout'
SEED = 'entity-confirmation-holdout-20261011-v1'
CV_ROOT = Path('/home/aim0/data/datasets/ASR/commonvoice_zh-CN/cv-corpus-26.0-2026-06-12/zh-CN')
CV_TSV = CV_ROOT / 'test.tsv'


def cv_exclusions():
    ids, refs = set(), set()
    for name in ('generic_entity_blind_20261010/metadata.json',
                 'generic_entity_optimization_20261010/holdout/metadata.json'):
        path = ROOT / 'results' / name
        for row in json.loads(path.read_text())['samples']:
            if row.get('source', '').startswith('commonvoice'):
                ids.add(row.get('location', {}).get('sentence_id', ''))
                refs.add(row['reference'])
    return ids, refs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order(row):
    return hashlib.sha256((SEED + '\0' + row['source_id']).encode()).hexdigest()


def main(news_extension=False, second_validation=False):
    global OUT
    if news_extension:
        OUT = ROOT / 'results/entity_confirmation_20261011/holdout_news'
    if second_validation:
        OUT = ROOT / 'results/entity_confirmation_20261011/holdout_v2'
    if OUT.exists():
        raise FileExistsError('Refuse to overwrite frozen holdout')
    OUT.mkdir(parents=True); (OUT / 'pcm').mkdir()
    converter = opencc.OpenCC('t2s')
    entities = {r['canonical_text']: r for r in json.loads((BASE / 'dataset/entities.json').read_text())}
    expression = re.compile('|'.join(re.escape(w) for w in sorted(entities, key=lambda w: (-len(w), w))))
    old = [json.loads(line) for line in (BASE / 'dataset/samples.jsonl').read_text().splitlines()]
    if news_extension or second_validation:
        old += [json.loads(line) for line in (ROOT/'results/entity_confirmation_20261011/holdout/samples.jsonl').read_text().splitlines()]
    if second_validation:
        old += [json.loads(line) for line in (ROOT/'results/entity_confirmation_20261011/holdout_news/samples.jsonl').read_text().splitlines()]
    used_paths = {r['audio_path'] for r in old}
    used_audio = {r.get('source_audio_sha256', r['audio_sha256']) for r in old}
    used_pcm = {r['pcm_sha256'] for r in old}
    used_refs = {converter.convert(r['reference']) for r in old}
    used_ids = {r.get('recording_id', r.get('source_id')) for r in old}
    old_aishell = historical_ids()
    cv_ids, cv_refs = cv_exclusions()
    pools = defaultdict(list)
    for line in (BASE / 'staging/aishell/samples.jsonl').read_text().splitlines():
        row = json.loads(line); ref = ''.join(row['reference'].split())
        if row['id'] in used_ids or row['id'] in old_aishell or str(Path(row['audio_path']).resolve()) in used_paths:
            continue
        if row['audio_sha256'] in used_audio or row['pcm_sha256'] in used_pcm or ref in used_refs:
            continue
        if not 3 <= row['duration_seconds'] <= 20:
            continue
        gold = [dict(canonical=e['text'], entity_type=entities[e['text']]['entity_type'],
                     start=len(''.join(row['reference'][:e['start']].split())),
                     end=len(''.join(row['reference'][:e['end']].split())),
                     annotation_method='official_aishell_ner') for e in row['entities'] if e['text'] in entities]
        has_surface = bool(expression.search(ref))
        if has_surface and not gold:
            continue
        sample = dict(source='AISHELL-1 test', source_id=row['id'], reference=ref,
                      reference_original=row['reference'], audio_path=row['audio_path'],
                      duration_seconds=row['duration_seconds'], gold_entities=gold,
                      topics=['新闻人物组织地理'], cohort='positive' if gold else 'negative')
        pools['新闻人物组织地理' if gold else 'negative'].append(sample)
    with (CV_ROOT / 'clip_durations.tsv').open() as f:
        durations = {r['clip']: float(r['duration[ms]']) / 1000 for r in csv.DictReader(f, delimiter='\t')}
    with CV_TSV.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            path = CV_ROOT / 'clips' / row['path']; ref = converter.convert(row['sentence'])
            if row['sentence_id'] in cv_ids or row['sentence'] in cv_refs or ref in used_refs or str(path.resolve()) in used_paths:
                continue
            duration = durations.get(row['path'], 0)
            if not path.exists() or not 3 <= duration <= 20 or len(ref) / duration > 8:
                continue
            gold = []
            topics = set()
            for match in expression.finditer(ref):
                e = entities[match.group()]
                gold.append(dict(canonical=match.group(), entity_type=e['entity_type'],
                                 start=match.start(), end=match.end(), annotation_method='reference_longest_surface_match'))
                topics.update(t for t in e['topics'] if t != '新闻人物组织地理')
            if (news_extension or second_validation) and any(e['entity_type'] in ('PERSON','ORG','LOC') for e in gold):
                topics.add('新闻人物组织地理')
            if not topics:
                continue
            sample = dict(source='Common Voice zh-CN test', source_id=row['sentence_id'],
                          reference=ref, reference_original=row['sentence'], audio_path=str(path.resolve()),
                          duration_seconds=duration, gold_entities=gold, topics=sorted(topics), cohort='positive')
            for topic in topics:
                pools[topic].append(sample)
    quotas = {topic: 12 for topic in sorted(pools) if topic not in ('negative', '新闻人物组织地理')}
    quotas.update({'新闻人物组织地理': 60, 'negative': 40})
    if news_extension:
        quotas = {'新闻人物组织地理': 67}
    if second_validation:
        quotas.update({'新闻人物组织地理': 112, 'negative': 50})
    selected = []; seen_paths = set(used_paths); seen_refs = set(used_refs)
    summary = dict(seed=SEED, pre_ASR_selection=True, fixed_library_size=len(entities),
                   pools={k: len(v) for k, v in pools.items()}, quotas=quotas, accepted={})
    print('POOLS', summary['pools'], flush=True)
    for topic, quota in quotas.items():
        count = 0
        def preference(row):
            if not news_extension and not second_validation: return (0, order(row))
            kinds = {e['entity_type'] for e in row['gold_entities']}
            return (0 if 'PERSON' in kinds else 1 if 'ORG' in kinds else 2, order(row))
        for row in sorted(pools[topic], key=preference):
            if count >= quota:
                break
            path = Path(row['audio_path'])
            if str(path) in seen_paths or row['reference'] in seen_refs:
                continue
            audio_hash = digest(path)
            if audio_hash in used_audio:
                continue
            sid = 'ec2_' + hashlib.sha256((SEED + audio_hash).encode()).hexdigest()[:20]
            pcm_path = OUT / 'pcm' / (sid + '.f32le')
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(path), '-map', '0:a:0',
                            '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_f32le', '-f', 'f32le', str(pcm_path)], check=True)
            pcm_hash = digest(pcm_path); duration = pcm_path.stat().st_size / 64000
            if pcm_hash in used_pcm or not 3 <= duration <= 20 or len(row['reference']) / duration > 8:
                pcm_path.unlink(); continue
            for e in row['gold_entities']:
                assert row['reference'][e['start']:e['end']] == e['canonical']
            row.update(id=sid, selection_topic=topic, pcm_path=str(pcm_path.resolve()),
                       audio_sha256=audio_hash, pcm_sha256=pcm_hash, duration_seconds=duration,
                       sample_rate=16000, channels=1, sample_format='f32le')
            selected.append(row); count += 1
            seen_paths.add(str(path)); seen_refs.add(row['reference']); used_audio.add(audio_hash); used_pcm.add(pcm_hash)
        summary['accepted'][topic] = count
    assert len({r['audio_sha256'] for r in selected}) == len(selected)
    assert len({r['pcm_sha256'] for r in selected}) == len(selected)
    summary.update(samples=len(selected), cohorts=dict(Counter(r['cohort'] for r in selected)),
                   limitations=['Reference literal annotations do not establish semantic NER identity.',
                                'Fixed known-word library, same source corpora; not speaker or model-pretraining independence.',
                                'Only AISHELL and Common Voice used; quotas may have source availability shortfalls.'])
    (OUT / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    manifest = dict(frozen=True, selection=summary, library_sha256=digest(BASE/'dataset/library/entities.db'),
                    samples_sha256=digest(OUT/'samples.jsonl'),
                    pcm_sha256={r['id']:r['pcm_sha256'] for r in selected},
                    original_benchmark_manifest_sha256=digest(BASE/'dataset/manifest.json'))
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (OUT/'READY').write_text('Frozen before ASR; no reference-guided inference\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--news-extension',action='store_true')
    parser.add_argument('--second-validation',action='store_true');args=parser.parse_args()
    main(args.news_extension,args.second_validation)
