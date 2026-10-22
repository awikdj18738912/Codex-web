"""Freeze and evaluate the current entity controllers on all fixed challenge inputs.

Inference reads audio, dictionaries and saved ASR hypotheses only. Reference
labels are reserved for the separate score_current_entity_challenge.py process.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/entity_challenge_20261014/dataset'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def prepare(out):
    if (out / 'protocol.json').exists():
        raise FileExistsError('Already frozen; collect resumes the existing protocol')
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / 'source_snapshot'
    shutil.copytree(ROOT / 'system', snapshot / 'system', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(DATA / 'libraries', out / 'libraries')
    shutil.copy2(Path(__file__), out / 'evaluate_current_entity_challenge.py')
    inputs = rows(DATA / 'inference_inputs.jsonl')
    assert len(inputs) == 2000 and len({s['id'] for s in inputs}) == 2000
    assert all('reference' not in s and 'gold_entities' not in s for s in inputs)
    for sample in inputs:
        sample['original_library_path'] = sample['library_path']
        sample['library_path'] = str(out / 'libraries' / Path(sample['library_path']).name)
    inputs.sort(key=lambda s: (s['library_path'], hashlib.sha256(s['id'].encode()).hexdigest()))
    (out / 'inference_inputs.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in inputs))
    # Discard stored scores: the inference file contains hypotheses, not labels.
    baseline = {r['id']: r['raw_text'] for r in rows(DATA / 'baseline_raw.jsonl')}
    write(out / 'saved_asr_hypotheses.json', baseline)
    before = subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True, check=True)
    (out / 'workspace_status_before.txt').write_text(before.stdout)
    diff = subprocess.run(['git', 'diff', '--', 'system'], cwd=ROOT, capture_output=True, text=True, check=True)
    (out / 'system_before.patch').write_text(diff.stdout)
    hashes = {str(p.relative_to(out)): sha(p) for d in ('source_snapshot', 'libraries') for p in (out / d).rglob('*') if p.is_file()}
    for name in ('inference_inputs.jsonl', 'saved_asr_hypotheses.json', 'evaluate_current_entity_challenge.py'):
        hashes[name] = sha(out / name)
    model = DATA / 'asr_model_identity.json'
    shutil.copy2(model, out / model.name)
    write(out / 'protocol.json', dict(
        prepared_utc=datetime.now(timezone.utc).isoformat(), sample_count=2000,
        project_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        dataset=str(DATA), dataset_manifest_sha256=sha(DATA / 'manifest.json'),
        dataset_inputs_sha256=sha(DATA / 'inference_inputs.jsonl'), frozen_sha256=hashes,
        asr_url='http://127.0.0.1:8766',
        module='current public system.entity_audio_repair.EntityAudioRepair; frozen defaults',
        transport=dict(pcm='16000 Hz mono float32 little endian', frame_bytes=64000,
                       decode_chunk_seconds=2., post_ack_delay_seconds=.08, concurrent_samples=1),
        baseline='Same actual primary ASR stream before the entity guard edits its response; saved 50.49% baseline reported separately',
        text_rules='prepare_entity_segment(allow_auto=True), normal production matcher limits, no neural Refiner',
        full_web=False, neural_refiner=False, inference_uses_reference_or_gold=False,
        capacity_failure_policy='Retain full library; report rules errors without truncation or limit changes',
        latency_scope='Direct entity controller + actual backend HTTP; accelerated file streaming, not browser or live microphone latency'))
    print('PREPARED', out, flush=True)


async def collect(out, limit):
    import httpx
    protocol = json.loads((out / 'protocol.json').read_text())
    for name, digest in protocol['frozen_sha256'].items():
        assert sha(out / name) == digest, name
    sys.path.insert(0, str(out / 'source_snapshot'))
    from system.entity_audio_repair import EntityAudioRepair
    from system.entity_matcher import EntityCandidateMatcher
    from system.entity_pipeline import prepare_entity_segment
    from system.entity_store import EntityStore
    from system.protection import EntityProtector
    from system.refinement_guard import split_for_refinement, join_refined_segments
    from system.zh_itn_bridge import ZhITNNormalizer
    itn = ZhITNNormalizer(ROOT / 'zh-itn/zh-itn/build/libzh_itn_bridge.so')
    inputs = rows(out / 'inference_inputs.jsonl')
    saved = json.loads((out / 'saved_asr_hypotheses.json').read_text())
    existing = rows(out / 'results.jsonl')
    done = {r['id'] for r in existing}
    assert len(done) == len(existing)
    definition_cache = {}
    rules_cache = {}

    def definitions_for(path):
        if path not in definition_cache:
            definition_cache[path] = EntityStore(path).list_entities(domain='general')
        return definition_cache[path]

    def apply_rules(text, path, confidence=None):
        if path not in rules_cache:
            definitions = definitions_for(path)
            try:
                matcher = EntityCandidateMatcher(definitions, selected_domain='general', mode='auto')
                rules_cache[path] = (EntityProtector(definitions, protect_numeric_spans=False), matcher, None)
            except ValueError as exc:
                rules_cache[path] = (None, None, str(exc))
        protector, matcher, error = rules_cache[path]
        if error:
            return None, [], error
        parts, changes = [], []
        for segment in split_for_refinement(text, one_punctuation_window=True):
            prepared = prepare_entity_segment(itn.normalize(segment).text, protector, matcher,
                                             allow_auto=True, confidence=confidence)
            parts.append(prepared.baseline_text)
            changes.extend(prepared.normalizations)
        return join_refined_segments(parts), changes, None

    began = time.perf_counter()
    run_count = 0
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        health = (await client.get(protocol['asr_url'] + '/health')).raise_for_status().json()
        if not (out / 'asr_health_before.json').exists():
            write(out / 'asr_health_before.json', health)
        for index, sample in enumerate(inputs):
            if sample['id'] in done:
                continue
            if limit is not None and run_count >= limit:
                break
            started = time.perf_counter()
            pcm = Path(sample['pcm_path']).read_bytes()
            assert hashlib.sha256(pcm).hexdigest() == sample['pcm_sha256'], sample['id']
            active = set()
            primary = None
            measured = dict(primary_finish_text=None, primary_pcm_seconds=0., replay_pcm_seconds=0.,
                            primary_http_calls=0, replay_http_calls=0, primary_wall_seconds=0., replay_wall_seconds=0.)
            initial_payload = None

            async def request(endpoint, sid=None, data=b'', params=None):
                nonlocal primary, initial_payload
                query = dict(params or {})
                is_primary = (endpoint == '/stream/start' and primary is None and not query.get('context')) or sid == primary and primary is not None
                prefix = 'primary' if is_primary else 'replay'
                measured[prefix + '_http_calls'] += 1
                if endpoint == '/stream/chunk':
                    measured[prefix + '_pcm_seconds'] += len(data) / 64000
                if sid:
                    query['session_id'] = sid
                tick = time.perf_counter()
                try:
                    payload = (await client.post(protocol['asr_url'] + endpoint, params=query, content=data)).raise_for_status().json()
                    append(out / 'asr_trace.jsonl', dict(id=sample['id'], endpoint=endpoint, session_id=sid,
                        primary=is_primary, context=query.get('context', ''), data_sha256=hashlib.sha256(data).hexdigest(),
                        data_bytes=len(data), result=payload, seconds=time.perf_counter()-tick))
                    if endpoint == '/stream/start':
                        active.add(payload['session_id'])
                        if is_primary:
                            primary = payload['session_id']
                            assert not payload.get('context_applied')
                    if endpoint in ('/stream/finish', '/stream/cancel'):
                        active.discard(sid)
                    if endpoint == '/stream/finish' and is_primary:
                        measured['primary_finish_text'] = payload.get('text', '')
                        initial_payload = dict(payload)
                    return payload
                finally:
                    measured[prefix + '_wall_seconds'] += time.perf_counter() - tick

            guard = EntityAudioRepair(request, definitions_for(sample['library_path']))
            try:
                ready = await guard.request('/stream/start', params=dict(language='Chinese', chunk_seconds=2.))
                for offset in range(0, len(pcm), 64000):
                    await guard.request('/stream/chunk', ready['session_id'], pcm[offset:offset+64000])
                    await asyncio.sleep(.08)
                final = await guard.request('/stream/finish', ready['session_id'])
                returned = time.perf_counter()
                assert initial_payload is not None
            except Exception as exc:
                append(out / 'errors.jsonl', dict(id=sample['id'], type=type(exc).__name__, error=str(exc)))
                raise
            finally:
                guard.close()
                tasks = set(getattr(guard, '_cleanup_tasks', ()))
                if tasks:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 180)
                for sid in list(active):
                    await request('/stream/cancel', sid)
            final_text = final.get('text', '')
            rules_text, changes, rule_error = apply_rules(final_text, sample['library_path'], final.get('confidence'))
            saved_rules, saved_changes, _ = apply_rules(saved[sample['id']], sample['library_path'])
            snapshot = guard.snapshot()
            snapshot['evaluation_backend'] = measured
            record = dict(id=sample['id'], domain=sample['domain'], library_path=sample['library_path'],
                pcm_sha256=sample['pcm_sha256'], audio_seconds=len(pcm)/64000,
                timing=dict(total_seconds=returned-started, settled_total_seconds=time.perf_counter()-started),
                result=dict(raw_text=final_text, entity_rules_auto_text=rules_text,
                            entity_rules_error=rule_error, entity_rules_changes=changes,
                            saved_baseline_rules_text=saved_rules, saved_baseline_rules_changes=saved_changes,
                            entity_audio_repair=snapshot))
            append(out / 'results.jsonl', record)
            done.add(sample['id'])
            run_count += 1
            progress = dict(completed=len(done), expected=2000, last_id=sample['id'], domain=sample['domain'],
                            last_seconds=round(time.perf_counter()-started, 3),
                            elapsed_this_process_seconds=round(time.perf_counter()-began, 3),
                            audio_changes=snapshot['change_count'], rules_error=rule_error)
            write(out / 'progress.json', progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    for name, digest in protocol['frozen_sha256'].items():
        assert sha(out / name) == digest, name
    write(out / 'collection_verification.json', dict(complete=len(done)==2000, samples=len(done),
          unique_ids=len(done), frozen_inputs_and_sources_verified=True, no_reference_labels_at_inference=True,
          actual_backend_http=True, primary_and_entity_output_share_one_stream=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'collect'))
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    out = args.run_dir.resolve()
    if args.stage == 'prepare':
        prepare(out)
    else:
        asyncio.run(collect(out, args.limit))


if __name__ == '__main__':
    main()
