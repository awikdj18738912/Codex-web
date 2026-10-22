"""Replay the previous 300 Oral candidates with automatic known-label admission.

No Oral regeneration and no manual label repair: compare the exact same source
plans, retain the original 20 Oral failures in the denominator. ASR calls and
semantic audits are real calls to the persistent single-GPU teacher.
"""
from __future__ import annotations
import asyncio
from collections import Counter
from contextvars import ContextVar
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = OUT.parent / 'refiner_retention_repair_300_20261017'
os.environ.update(VLLM_BASE_URL='http://127.0.0.1:8010/v1', VLLM_MODEL_NAME='refiner-teacher',
                  OPENROUTER_API_KEY='', OPENROUTER_MODEL='', CLEAN_TARGET_MODE='construction')
sys.path[:0] = [str(ROOT / 'pipeline'), str(ROOT)]
from configs import settings
from src.generators.clean_text_generator import CleanTextGenerator
from src.generators.asr_simulator import ASRSimulator, _natural_prefixes
from src.pipeline.dedup import deduplicate
from src.services.llm_client import AsyncLLMClient

for name in ['final', 'source_snapshot']:
    (OUT / name).mkdir(parents=True, exist_ok=True)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'pipeline/scripts'/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


assemble = module('replay_assemble', '04_assemble.py')
finalize = module('replay_finalize', '05_finalize.py')
exporter = module('replay_export', 'export_sft.py')
CONTEXT = ContextVar('context', default={})
CALLS = Counter()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n')


class Client(AsyncLLMClient):
    async def generate(self, **kwargs):
        t = time.monotonic()
        row = dict(CONTEXT.get(), request=kwargs)
        try:
            answer = await super().generate(**kwargs)
            row['response'] = answer
            return answer
        except Exception as exc:
            row['error'] = type(exc).__name__+':'+str(exc)
            raise
        finally:
            row['seconds'] = round(time.monotonic()-t, 3)
            CALLS[row.get('stage', 'unknown')] += 1
            append(OUT/'api_calls.jsonl', row)


async def call(stage, case_id, coroutine):
    token = CONTEXT.set(dict(stage=stage, id=case_id))
    try:
        return await coroutine
    finally:
        CONTEXT.reset(token)


def all_jobs():
    weights = settings.SCENE_DISTRIBUTION
    quotas = {s: int(1000*w) for s, w in weights.items()}
    order = sorted(weights, key=lambda s: 1000*weights[s]-quotas[s], reverse=True)
    for scene in order[:1000-sum(quotas.values())]:
        quotas[scene] += 1
    jobs = [(s, i) for s, n in quotas.items() for i in range(1, n+1)]
    random.Random(2026101709).shuffle(jobs)
    return jobs[:300]


async def main():
    if (OUT/'attempts.jsonl').exists():
        raise RuntimeError('Refusing to overwrite an existing replay')
    t0 = time.monotonic()
    paths = sorted((ROOT/'pipeline').rglob('*.py')) + sorted((ROOT/'pipeline/configs').glob('*.json'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:
        dest = OUT/'source_snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    dump(OUT/'manifest.json', dict(baseline=str(BASE), source_sha256=hashes,
        method='Same 300 observed Oral plans; Oral failures remain rejected. New construction labels, real semantic/ASR calls, no manual relabeling.',
        model='refiner-teacher', endpoint='http://127.0.0.1:8010/v1', target_mode=settings.CLEAN_TARGET_MODE))
    baseline = {r['id']: r for r in (json.loads(s) for s in (BASE/'attempts.jsonl').read_text().splitlines())}
    queue = asyncio.Queue()
    for serial, (scene, number) in enumerate(all_jobs()):
        cid = f'scale_{scene}_{number:04d}'
        assert cid in baseline
        queue.put_nowait((serial, scene, cid, baseline[cid]))
    assert queue.qsize() == len(baseline) == 300
    client = Client(max_concurrent=4, timeout=600, max_retries=1)
    cleaner = CleanTextGenerator(client, target_mode='construction')
    cleaner.qc.llm = client
    sem = asyncio.Semaphore(4)
    statuses, accepted = Counter(), []

    def progress():
        dump(OUT/'progress.json', dict(completed=sum(statuses.values()), target=300,
             statuses=dict(statuses), elapsed_seconds=round(time.monotonic()-t0, 1), api_calls=dict(CALLS)))

    async def worker():
        while not queue.empty():
            serial, scene, cid, old = queue.get_nowait()
            row = dict(id=cid, scene=scene)
            try:
                oral = old.get('oral')
                if not oral:
                    row['status'] = 'oral_rejected'
                    continue
                row['oral'] = oral
                clean = await call('clean', cid, cleaner.generate_one(scene, oral))
                row['clean'] = clean
                if not clean:
                    row['status'] = 'clean_rejected'
                    continue
                rng_seed = 2026101723+serial*401
                row['asr_selected'] = 'clean' not in clean.get('speech_phenomena', []) and random.Random(rng_seed).random()<settings.ASR_SIMULATION_RATIO
                row['natural_fragment_eligible'] = bool(_natural_prefixes(clean['clean_text'], 8))
                simulator = ASRSimulator(client, mode='llm_sim', rng=random.Random(rng_seed))
                asr = await call('asr', cid, simulator.simulate_one(scene, clean))
                row['asr'] = asr
                if not asr:
                    row['status'] = 'asr_rejected'
                    continue
                records, stats = assemble._assemble_from_asr(scene, [asr])
                row['assembly_stats'] = stats
                if not records:
                    row['status'] = 'assembly_rejected'
                    continue
                checked, passed, reason = await call('audit', cid, finalize.qc_one(records[0], client, sem))
                row.update(audit=checked.get('meta', {}).get('quality_audit'), audit_reason=reason)
                if not passed:
                    row['status'] = 'audit_rejected'
                    continue
                if exporter.convert_one(checked) is None:
                    row['status'] = 'export_rejected'
                    continue
                row['status'] = 'accepted'
                accepted.append(checked)
                append(OUT/'accepted_candidates.jsonl', checked)
            except Exception as exc:
                row.update(status='exception', error=type(exc).__name__+':'+str(exc))
                logging.exception('Replay failed %s', cid)
            finally:
                statuses[row['status']] += 1
                append(OUT/'attempts.jsonl', row)
                queue.task_done()
                if sum(statuses.values())%25==0:
                    progress()

    progress()
    await asyncio.gather(*(worker() for _ in range(4)))
    accepted.sort(key=lambda r:r['id'])
    unique = deduplicate(accepted, threshold=settings.DEDUP_THRESHOLD)
    with (OUT/'final/train.jsonl').open('w') as f:
        for row in unique:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    dump(OUT/'final/pilot_sft.json', [exporter.convert_one(r) for r in unique])
    changed = [name for name, digest in hashes.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    summary = dict(candidates=300, accepted_before_dedup=len(accepted), unique_retained=len(unique),
        retained_rate=len(unique)/300, statuses=dict(statuses), api_calls=dict(CALLS),
        real_fragments=sum(bool(r['meta'].get('is_fragment')) for r in unique),
        elapsed_seconds=round(time.monotonic()-t0, 2), source_files_changed=changed,
        comparison_caveat='Replay of fixed Oral candidates; not a fresh end-to-end generation throughput measurement. New semantic/ASR calls remain stochastic.')
    dump(OUT/'summary.json', summary)
    progress()
    await client.client.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
