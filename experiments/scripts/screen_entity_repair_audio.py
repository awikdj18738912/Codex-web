"""Collect an unprompted streaming ASR trace for material selection only.

Audio is submitted faster than playback. This screens errors and is not a
user-visible latency measurement. No glossary or reference is loaded.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.evaluate_entity_ctc_review_combination import ASR_URL, LiveAsr


async def collect(pcm_path, output):
    pcm = pcm_path.read_bytes()
    assert len(pcm) % 4 == 0
    trace = []
    began = time.perf_counter()
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        backend = LiveAsr(client)
        health = (await client.get(ASR_URL + '/health')).raise_for_status().json()
        try:
            start = await backend('/stream/start', params={'language': 'Chinese', 'chunk_seconds': 2.})
            assert not start.get('context_applied')
            sid = start['session_id']
            for offset in range(0, len(pcm), 64000):
                data = pcm[offset:offset + 64000]
                payload = await backend('/stream/chunk', sid, data)
                trace.append(dict(audio_seconds=(offset + len(data)) / 64000, payload=payload))
                if len(trace) % 60 == 0:
                    print('SCREEN', len(trace), '/', len(pcm) / 64000, flush=True)
            final = await backend('/stream/finish', sid)
            result = dict(pcm_path=str(pcm_path), pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                duration_seconds=len(pcm) / 64000, start=start, health=health,
                chunk_payloads=trace, final_payload=final,
                elapsed_seconds=time.perf_counter() - began,
                timing_scope='Accelerated actual streaming ASR; error screening only, not paced latency')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            output.with_suffix('.txt').write_text(str(final.get('text', '')) + '\n')
            print('COMPLETE', round(result['elapsed_seconds'], 2), flush=True)
        finally:
            await backend.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pcm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(collect(args.pcm.resolve(), args.output.resolve()))
