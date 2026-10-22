"""Actual entity-repair runs for a frozen diagnostic audio manifest.

Accelerated delivery is for quality diagnosis. Optional per-packet CTC drain
avoids a file sender artificially overrunning the acoustic worker. No timing
from this runner is a real-time latency result. Entity labels are not loaded.
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
from system.entity_store import EntityStore
from experiments.scripts.evaluate_entity_ctc_review_combination import LiveAsr
from system.entity_audio_repair import EntityAudioRepair
from system.entity_ctc_sidecar import EntityCtcRuntime


async def run(manifest_path, output_dir, mode):
    manifest = json.loads(manifest_path.read_text())
    pcm = Path(manifest['pcm_path']).read_bytes()
    assert hashlib.sha256(pcm).hexdigest() == manifest['pcm_sha256']
    library = Path(manifest['library_path'])
    assert hashlib.sha256(library.read_bytes()).hexdigest() == manifest['library_sha256']
    defs = EntityStore(library).list_entities()
    runtime = EntityCtcRuntime(ROOT / 'results/entity_backend_comparison_20261015/models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30') if mode == 'on' else None
    ctc = await asyncio.to_thread(runtime.create_session, defs) if runtime else None
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (mode + '.json')
    if target.exists():
        raise FileExistsError(target)
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        backend = LiveAsr(client)
        statuses, public = [], []
        began = time.perf_counter()

        async def on_status(snapshot):
            statuses.append(dict(wall_seconds=time.perf_counter()-began, snapshot=snapshot))

        guard = EntityAudioRepair(backend, defs, ctc_session=ctc, ctc_mode=mode, on_status=on_status)
        try:
            start = await guard.request('/stream/start', params={'language': 'Chinese', 'chunk_seconds': 2.})
            assert not start.get('context_applied')
            sid = start['session_id']
            for offset in range(0, len(pcm), 64000):
                part = pcm[offset:offset+64000]
                payload = await guard.request('/stream/chunk', sid, part)
                public.append(dict(audio_seconds=(offset+len(part))/64000,
                    text=payload.get('text', ''), revision=guard.revision))
                if ctc is not None and ctc._futures:
                    await asyncio.wrap_future(ctc._futures[-1])
                await asyncio.sleep(0)
                if len(public) % 60 == 0:
                    print(mode, 'AUDIO', len(public), '/', len(pcm)/64000, flush=True)
            final = await guard.request('/stream/finish', sid)
            snapshot = guard.snapshot()
            result = dict(id=manifest['id'], mode=mode, pcm_sha256=manifest['pcm_sha256'],
                library_sha256=manifest['library_sha256'], start=start,
                primary_text=backend.primary_final, text=final.get('text', ''),
                entity_audio_repair=snapshot, public_chunks=public, status_events=statuses,
                elapsed_seconds=time.perf_counter()-began,
                timing_scope='Accelerated quality diagnostic, CTC audio worker drained per packet; not user latency',
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), ROOT/'system/entity_audio_repair.py',
                        ROOT/'system/entity_streaming_live.py', ROOT/'system/entity_ctc_sidecar.py',
                        ROOT/'system/entity_direct_spelling.py', ROOT/'system/entity_mixed_spelling.py']})
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print('COMPLETE', mode, 'changes', len(snapshot.get('changes', [])), flush=True)
        finally:
            guard.close()
            tasks = tuple(getattr(guard, '_cleanup_tasks', ()))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await backend.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['off', 'on'], required=True)
    args = parser.parse_args()
    asyncio.run(run(args.manifest.resolve(), args.output_dir.resolve(), args.mode))
