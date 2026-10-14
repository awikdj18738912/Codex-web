"""Build a baseline-selected real-audio entity challenge without module selection.

Preparation/selection read scoring references. The raw ASR collector reads only
frozen inference inputs and sends unmodified PCM without context or a glossary.
Run with the default Python and the existing functional_test_deps on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
import wave

import httpx
import numpy as np
import opencc
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.entity_store import EntityStore
from system.zh_itn_bridge import ZhITNNormalizer

OUT = ROOT / 'results/entity_challenge_20261014'
AUDIO = Path('/home/aim0/data/datasets/ASR/entity_challenge_20261014/pcm')
OLD = ROOT / 'results/entity_modes_dataset_20261013'
SEED = 'entity-challenge-20261014-v1'
CC = opencc.OpenCC('t2s')
ITN = ZhITNNormalizer()


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l] if Path(path).exists() else []


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def lines(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w') as f:
        for r in values:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rank(text):
    return hashlib.sha256((SEED + text).encode()).hexdigest()


@lru_cache(maxsize=100000)
def norm(text):
    value = unicodedata.normalize('NFKC', CC.convert(ITN.normalize(text).text)).casefold()
    return ''.join(c for c in value if unicodedata.category(c)[0] not in {'P', 'Z', 'C'})


def score(text, label):
    t = norm(text)
    expected = Counter(norm(e['canonical']) for e in label['gold_entities'])
    assert all(w and norm(label['reference']).count(w) >= n for w, n in expected.items())
    hits = sum(min(n, t.count(w)) for w, n in expected.items())
    return dict(hits=hits, expected_mentions=sum(expected.values()),
                missed=[dict(canonical=w, count=n-min(n, t.count(w)))
                        for w, n in expected.items() if t.count(w) < n],
                errors=Levenshtein.distance(norm(label['reference']), t),
                reference_chars=len(norm(label['reference'])))


def scorable_label(sample, original):
    label = dict(original)
    if sample.get('source', '').startswith('AISHELL'):
        ref = original['reference']
        # Official AISHELL text is word/character-tokenized with spaces. The
        # original benchmark also compacts these before conservative ITN.
        label['reference'] = ''.join(c for c in ref if not c.isspace())
        if ref != label['reference']:
            label['reference_original'] = ref
            label['gold_entities'] = []
            offsets = [0]
            for c in ref: offsets.append(offsets[-1] + int(not c.isspace()))
            for e in original['gold_entities']:
                converted = dict(e)
                if 'start' in e and 'end' in e:
                    converted.update(start=offsets[e['start']], end=offsets[e['end']])
                    assert norm(label['reference'][converted['start']:converted['end']]) == norm(e['canonical'])
                label['gold_entities'].append(converted)
    return label


def cache():
    result = {}
    for r in read(OLD / 'pure_asr/results.jsonl'):
        ready = r['ready']
        assert not ready['entity_asr_context_requested'] and not ready['entity_audio_repair_requested']
        assert not ready['refiner_enabled'] and not ready['postprocessing_enabled']
        result[r['pcm_sha256']] = dict(raw_text=r['result']['raw_text'],
            source_result=str(OLD / 'pure_asr/results.jsonl'), source_id=r['id'],
            original_timing=r['timing'], reused=True)
    for p in sorted(OUT.glob('pool_*/baseline.jsonl')):
        for r in read(p):
            result.setdefault(r['pcm_sha256'], dict(raw_text=r['raw_text'],
                source_result=str(p), source_id=r['id'], reused=True))
    return result


def pcm_from_wave(data):
    from scipy.io.wavfile import read as read_wave
    rate, values = read_wave(io.BytesIO(data))
    assert rate == 16000 and values.ndim == 1
    if values.dtype == np.int16:
        values = values.astype('<f4') / 32768
    else:
        assert values.dtype == np.float32 and np.isfinite(values).all()
    return values.astype('<f4', copy=False).tobytes()


def save_pcm(data):
    h = hashlib.sha256(data).hexdigest()
    p = AUDIO / (h + '.f32le')
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(data)
    assert p.stat().st_size == len(data)
    return p, h


def freeze_pool(name, samples, vocabulary_paths=()):
    folder = OUT / name
    if (folder / 'manifest.json').exists():
        raise FileExistsError(folder / 'manifest.json')
    labels = {}
    inputs = []
    seen_pcm, seen_ref = set(), set()
    raw_cache = cache()
    protected_file = ROOT / 'results/entity_context_confirmation_20261012/unseen/summary.json'
    protected = set(json.loads(protected_file.read_text()).get('pcm_sha256', {}).values()) if protected_file.exists() else set()
    reused = []
    for s in samples:
        if s['pcm_sha256'] in protected or s['pcm_sha256'] in seen_pcm or norm(s['reference']) in seen_ref:
            continue
        seen_pcm.add(s['pcm_sha256']); seen_ref.add(norm(s['reference']))
        gold = s.pop('gold_entities'); reference = s.pop('reference')
        labels[s['id']] = dict(reference=reference, gold_entities=gold,
            annotation_method=s.pop('annotation_method', 'reference_curated'))
        if 'reference_original' in s:
            labels[s['id']]['reference_original'] = s.pop('reference_original')
        inputs.append(s)
        if s['pcm_sha256'] in raw_cache:
            reused.append(dict(id=s['id'], pcm_sha256=s['pcm_sha256'],
                **raw_cache[s['pcm_sha256']]))
    lines(folder / 'inference_inputs.jsonl', inputs)
    write(folder / 'references.json', dict(samples=labels))
    lines(folder / 'baseline.jsonl', reused)
    write(folder / 'manifest.json', dict(schema_version=1, samples=len(inputs),
        domains=dict(Counter(s['domain'] for s in inputs)), seed=SEED,
        inference_input_sha256=sha(folder / 'inference_inputs.jsonl'),
        scoring_reference_sha256=sha(folder / 'references.json'),
        vocabulary_hashes={str(p): sha(p) for p in vocabulary_paths},
        baseline_reused=len(reused), normalization='zh-ITN, OpenCC t2s, NFKC/casefold, strip Unicode P/Z/C',
        selection='reference-only candidate admission; raw baseline later determines difficulty; no module results used',
        annotation_limitation='Source references and dictionaries; no per-item audio listening claim'))
    print('FROZEN', name, len(inputs), 'REUSED', len(reused), flush=True)


def prepare_old():
    inputs = {s['id']: s for s in read(OLD / 'inference_inputs.jsonl')}
    refs = json.loads((OLD / 'references.json').read_text())['samples']
    samples = []
    for sid, original in inputs.items():
        if not refs[sid]['gold_entities']:
            continue
        s = dict(original)
        s['domain'] = {'automotive': '汽车交通', 'football': '足球'}.get(s['library_key'],
            re.sub(r'^topic_\d+_', '', s['library_key']))
        if s['domain'] == '新闻人物组织地理': s['domain'] = '人名机构地名'
        s.update(refs[sid]); s['origin'] = 'existing_benchmark'
        samples.append(s)
    freeze_pool('pool_existing', samples)


def prepare_aishell(limit, name):
    samples = []
    for r in read(OUT / 'aishell_pool.jsonl')[:limit]:
        data = Path(r['audio_path']).read_bytes()
        pcm = pcm_from_wave(data); p, h = save_pcm(pcm)
        gold = [dict(canonical=e['text'], entity_type=e['type'], start=e['start'], end=e['end'],
                     annotation_method='official_aishell_ner') for e in r['gold_entities']]
        samples.append(dict(id='chall_aishell_' + r['source_recording_id'],
            origin='additional_aishell', source='AISHELL-1 / AISHELL-NER',
            source_recording_id=r['source_recording_id'], speaker_id=r['speaker_id'], split=r['split'],
            audio_path=r['audio_path'], domain=r['domain'], pcm_path=str(p), pcm_sha256=h,
            duration_seconds=len(pcm)/64000, source_audio_sha256=hashlib.sha256(data).hexdigest(),
            reference=r['reference'], gold_entities=gold, annotation_method='official_aishell_ner'))
    freeze_pool(name, samples, [OUT / 'aishell_pool.jsonl'])


def find_entities(text, words):
    if not words: return []
    pattern = re.compile('|'.join(re.escape(w) for w in sorted(words, key=lambda w: (-len(w), w))))
    return [dict(canonical=m.group(), entity_type=words[m.group()], start=m.start(), end=m.end(),
                 annotation_method='agent_reference_curated') for m in pattern.finditer(text)]


def prepare_speechio(vocabulary_paths, limit, name):
    vocab = defaultdict(dict); domains = {}
    for p in vocabulary_paths:
        for w in json.loads(Path(p).read_text())['words']:
            if norm(w['canonical']) in {norm(x) for x in vocab[w['subset']]}:
                continue
            vocab[w['subset']][w['canonical']] = w['entity_type']; domains[w['subset']] = w['domain']
    grouped = defaultdict(list)
    for r in read(OUT / 'source_catalog.jsonl'):
        gold = find_entities(r['text'], vocab[r['subset']])
        if gold and 2 <= r['duration'] <= 20 and len(norm(r['text'])) >= 6:
            r['gold_entities'] = gold
            grouped[r['subset']].append(r)
    samples = []
    for subset, rows in sorted(grouped.items()):
        # Favor entity-bearing speech, never ASR errors, during pool admission.
        rows.sort(key=lambda r: (rank(subset + ':' + r['segment_id'])))
        selected = rows[:limit] if limit else rows
        table = pq.read_table(rows[0]['parquet_path'], columns=['audio'])
        for r in selected:
            audio = table['audio'][r['row_index']].as_py()['bytes']
            pcm = pcm_from_wave(audio); p, h = save_pcm(pcm)
            assert abs(len(pcm)/64000 - r['duration']) < .1
            samples.append(dict(id='chall_speechio_' + rank(subset + r['segment_id'])[:20],
                origin='speechio_expansion', source='SpeechIO', subset=subset,
                source_recording_id=r['segment_id'], speaker_id=r['speaker'],
                source_parquet=r['parquet_path'], source_row_index=r['row_index'], domain=domains[subset],
                pcm_path=str(p), pcm_sha256=h, duration_seconds=len(pcm)/64000,
                source_audio_sha256=hashlib.sha256(audio).hexdigest(),
                reference=r['text'], gold_entities=r['gold_entities']))
        print('PREPARED', subset, len(selected), 'ELIGIBLE', len(rows), flush=True)
    freeze_pool(name, samples, vocabulary_paths)


def prepare_medical(name):
    from prepare_large_entity_medical import cmeee_terms, build_trie, longest_nonoverlap
    manifest_path = ROOT / 'results/multimed_chinese_test_eval_20261008/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    terms = cmeee_terms()
    # Descriptive fragments from the text NER corpus are not canonical targets.
    terms = {w:k for w,k in terms.items() if w not in {'有阴影', '皮质下'}}
    trie = build_trie(terms)
    samples = []
    evidence = defaultdict(list)
    for r in manifest['samples']:
        if not 2 <= r['duration_seconds'] <= 20: continue
        gold = longest_nonoverlap(r['reference'], trie)
        if not gold: continue
        data = Path(r['audio_path']).read_bytes(); pcm = pcm_from_wave(data); p, h = save_pcm(pcm)
        for e in gold:
            e['entity_type'] = 'TERM'
            e['annotation_method'] = 'cmeee_dictionary_exact_reference_match'
            evidence[e['canonical']].append(dict(reference=r['reference'], source_id=r['id'],
                                                 start=e['start'], end=e['end']))
        samples.append(dict(id='chall_medical_' + r['id'], origin='multimed_expansion',
            source='MultiMed Chinese test', source_recording_id=r['id'],
            domain='临床医疗', audio_path=r['audio_path'], pcm_path=str(p), pcm_sha256=h,
            source_audio_sha256=hashlib.sha256(data).hexdigest(), duration_seconds=len(pcm)/64000,
            reference=r['reference'], gold_entities=gold,
            annotation_method='cmeee_dictionary_exact_reference_match'))
    evidence_path = OUT / (name + '_vocabulary_evidence.json')
    write(evidence_path, dict(words=evidence,
        annotation_limitation='CMeEE terms exact-matched to supplied MultiMed references; no native audio NER or per-item listening'))
    freeze_pool(name, samples, [manifest_path, evidence_path])


async def collect(name, limit, per_domain_limit=0, priority_domains=()):
    folder = OUT / name
    m = json.loads((folder / 'manifest.json').read_text())
    assert sha(folder / 'inference_inputs.jsonl') == m['inference_input_sha256']
    inputs = read(folder / 'inference_inputs.jsonl')
    assert all('reference' not in s and 'gold_entities' not in s for s in inputs)
    done = {r['id'] for r in read(folder / 'baseline.jsonl')}
    by_id = {s['id']:s for s in inputs}
    domain_done = Counter(by_id[sid]['domain'] for sid in done)
    with (folder / 'collection_runs.jsonl').open('a') as f:
        f.write(json.dumps(dict(started_unix=time.time(), limit=limit,
                               per_domain_limit=per_domain_limit, priority_domains=priority_domains,
                               already_completed=len(done)))+'\n')
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        health = (await client.get('http://127.0.0.1:8766/health')).raise_for_status().json()
        protocol = dict(asr_url='http://127.0.0.1:8766', health=health,
            chunk_seconds=2.0, transport_samples=16000, context='', language='Chinese',
            reference_at_inference=False, module_at_inference=False, audio_transform='original mono PCM16 to float32 only',
            inference_input_sha256=m['inference_input_sha256'])
        if (folder / 'protocol.json').exists():
            previous = json.loads((folder / 'protocol.json').read_text())
            order = previous.get('order', 'input_order')
            assert {k:v for k,v in previous.items() if k != 'order'} == protocol
        else:
            order = 'round_robin_domains_hash'
            write(folder / 'protocol.json', dict(protocol, order=order))
        if order == 'round_robin_domains_hash':
            groups = defaultdict(list)
            for s in inputs: groups[s['domain']].append(s)
            for g in groups.values(): g.sort(key=lambda s:rank(s['id']))
            inputs = [groups[d][i] for i in range(max(map(len,groups.values())))
                      for d in sorted(groups) if i < len(groups[d])]
        priority_domains = set(priority_domains)
        if priority_domains:
            inputs.sort(key=lambda s:s['domain'] not in priority_domains)
        count = 0
        for index, s in enumerate(inputs):
            if s['id'] in done: continue
            if per_domain_limit and s['domain'] not in priority_domains and domain_done[s['domain']] >= per_domain_limit: continue
            if limit and count >= limit: break
            pcm = Path(s['pcm_path']).read_bytes(); assert hashlib.sha256(pcm).hexdigest() == s['pcm_sha256']
            result = None
            for attempt in range(3):
                session = None
                try:
                    start_time = time.perf_counter()
                    ready = (await client.post(protocol['asr_url'] + '/stream/start',
                        params=dict(language='Chinese', chunk_seconds=2.0))).raise_for_status().json()
                    assert not ready.get('context_applied'); session = ready['session_id']
                    for offset in range(0, len(pcm), 64000):
                        response = (await client.post(protocol['asr_url'] + '/stream/chunk',
                            params=dict(session_id=session), content=pcm[offset:offset+64000])).raise_for_status().json()
                        assert not response.get('context_applied')
                    result = (await client.post(protocol['asr_url'] + '/stream/finish',
                        params=dict(session_id=session))).raise_for_status().json()
                    assert not result.get('context_applied')
                    elapsed = time.perf_counter() - start_time
                    break
                except Exception as e:
                    print('RETRY', s['id'], attempt+1, type(e).__name__, flush=True)
                    if attempt == 2: raise
                finally:
                    if session and result is None:
                        await client.post(protocol['asr_url'] + '/stream/cancel', params=dict(session_id=session))
            record = dict(id=s['id'], pcm_sha256=s['pcm_sha256'], raw_text=result['text'],
                asr_payload=result, ready=ready, elapsed_seconds=elapsed, reused=False,
                context_supplied=False, refiner_enabled=False, entity_module_enabled=False)
            with (folder / 'baseline.jsonl').open('a') as f: f.write(json.dumps(record, ensure_ascii=False) + '\n')
            done.add(s['id']); count += 1
            domain_done[s['domain']] += 1
            if count % 100 == 0:
                print('ASR', name, len(done), '/', len(inputs), flush=True)
                write(folder / 'progress.json', dict(completed=len(done), total=len(inputs), last_id=s['id']))
        write(folder / 'progress.json', dict(completed=len(done), total=len(inputs), complete=len(done)==len(inputs),
              staged_domain_cap=per_domain_limit, completed_by_domain=dict(domain_done)))
    print('COLLECTED', name, len(done), '/', len(inputs), flush=True)


def scored_pool():
    records = {}
    quarantine = []
    protected_file = ROOT / 'results/entity_context_confirmation_20261012/unseen/summary.json'
    protected = set(json.loads(protected_file.read_text()).get('pcm_sha256', {}).values()) if protected_file.exists() else set()
    for folder in sorted(OUT.glob('pool_*')):
        if not (folder / 'references.json').exists(): continue
        labels = json.loads((folder / 'references.json').read_text())['samples']
        inputs = {s['id']: s for s in read(folder / 'inference_inputs.jsonl')}
        for r in read(folder / 'baseline.jsonl'):
            s = inputs[r['id']]; label = scorable_label(s, labels[r['id']])
            if s['pcm_sha256'] in protected:
                quarantine.append(dict(id=r['id'], pool=str(folder), reason='protected_holdout_pcm',
                                       pcm_sha256=s['pcm_sha256']))
                continue
            expected = Counter(norm(e['canonical']) for e in label['gold_entities'])
            unsupported = [w for w,n in expected.items() if not w or norm(label['reference']).count(w)<n]
            if unsupported:
                quarantine.append(dict(id=r['id'], pool=str(folder), reason='normalized_gold_not_supported_by_reference',
                    unsupported=unsupported, reference=label['reference'], gold_entities=label['gold_entities']))
                continue
            data = dict(s, **label, baseline_raw_text=r['raw_text'], baseline_score=score(r['raw_text'], label),
                        baseline_source=str(folder / 'baseline.jsonl'), baseline_reused=r.get('reused', False))
            # Expanded frozen reference vocabulary can supersede a sparse old annotation.
            prior = records.get(s['pcm_sha256'])
            if prior is None or len(label['gold_entities']) > len(prior['gold_entities']):
                records[s['pcm_sha256']] = data
    # Transcript duplicates do not fill the recording quota.
    unique = {}
    for r in records.values():
        key = norm(r['reference'])
        if key not in unique or rank(r['id']) < rank(unique[key]['id']): unique[key] = r
    write(OUT / 'candidate_quarantine.json', dict(rule='reference/annotation-only consistency check; ASR text not used',
                                               samples=quarantine))
    return list(unique.values())


def pool_report():
    records = scored_pool(); summary = {}
    for d in sorted({r['domain'] for r in records}):
        rows = [r for r in records if r['domain'] == d]
        hits = sum(r['baseline_score']['hits'] for r in rows)
        total = sum(r['baseline_score']['expected_mentions'] for r in rows)
        summary[d] = dict(recordings=len(rows), hits=hits, mentions=total, recall=hits/total,
            error_recordings=sum(r['baseline_score']['hits'] < r['baseline_score']['expected_mentions'] for r in rows),
            zero_hit_recordings=sum(r['baseline_score']['hits'] == 0 for r in rows))
    write(OUT / 'candidate_baseline_summary.json', dict(recordings=len(records), domains=summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['existing', 'aishell', 'speechio', 'medical', 'collect', 'pool-report'])
    p.add_argument('--name', default='pool_aishell')
    p.add_argument('--limit', type=int, default=2000)
    p.add_argument('--vocabulary', action='append', default=[])
    p.add_argument('--per-domain-limit', type=int, default=0)
    p.add_argument('--priority-domain', action='append', default=[])
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'existing': prepare_old()
    elif a.stage == 'aishell': prepare_aishell(a.limit, a.name)
    elif a.stage == 'speechio': prepare_speechio(a.vocabulary, a.limit, a.name)
    elif a.stage == 'medical': prepare_medical(a.name)
    elif a.stage == 'collect': asyncio.run(collect(a.name, a.limit, a.per_domain_limit, a.priority_domain))
    elif a.stage == 'pool-report': pool_report()


if __name__ == '__main__': main()
