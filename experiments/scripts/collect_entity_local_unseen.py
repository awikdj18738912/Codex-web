"""Collect a reference-free first pass for frozen long validation streams."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/entity_local_replay_20261012'
OUT=BASE/'v2/unseen_evaluation'


async def main():
    import httpx
    OUT.mkdir(parents=True,exist_ok=True)
    source_dirs=[BASE/'v2/unseen_long',BASE/'v2/unseen_long_supplement']
    manifests=[json.loads((p/'manifest.json').read_text()) for p in source_dirs]
    cases=[r for m in manifests for r in m['cases']]
    assert len(cases)==7 and len({r['id'] for r in cases})==7
    destination=OUT/'inputs.jsonl'
    done={json.loads(s)['id'] for s in destination.read_text().splitlines() if s} if destination.exists() else set()
    async with httpx.AsyncClient(timeout=180,trust_env=False) as client:
        for row in cases:
            if row['id'] in done:continue
            pcm=Path(row['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
            ready=(await client.post('http://127.0.0.1:8766/stream/start',params={'language':'Chinese','chunk_seconds':2})).raise_for_status().json()
            assert not ready.get('context_applied');sid=ready['session_id'];payloads=[]
            try:
                for offset in range(0,len(pcm),64000):
                    payloads.append((await client.post('http://127.0.0.1:8766/stream/chunk',params={'session_id':sid},content=pcm[offset:offset+64000])).raise_for_status().json())
                final=(await client.post('http://127.0.0.1:8766/stream/finish',params={'session_id':sid})).raise_for_status().json()
            finally:await client.post('http://127.0.0.1:8766/stream/cancel',params={'session_id':sid})
            result={k:row[k] for k in ('id','pcm_path','pcm_sha256','library','library_sha256','duration_seconds')}
            result.update(kind=row['kind'],initial={'ready':ready,'chunk_payloads':payloads,'final_payload':final})
            with destination.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
            print('FRESH_INITIAL',row['id'],row['duration_seconds'],flush=True)
    plan=json.loads((BASE/'v2/sparse_development/plan.json').read_text())
    plan.update(cases=len(cases),input_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                frozen_selection_sha256={str(p/'manifest.json'):hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest() for p in source_dirs},
                independent_audio_components=True,source_component_count=113,scope='Seven source-recording-disjoint real clip concatenations. Five positives/two negatives. References not used in first pass or replay.')
    for name,path in [('original_entity_audio_repair.py',BASE/'original_entity_audio_repair.py'),
                      ('candidate_entity_audio_repair.py',BASE/'candidate_v2_entity_audio_repair.py')]:
        link=OUT/name
        if not link.exists():link.symlink_to(path)
    (OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':asyncio.run(main())
