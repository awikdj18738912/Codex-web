"""Replay frozen Qwen PCM/events through the actual live guard and CTC model.

The prompted audio-review backend is deliberately inconclusive. This isolates
CTC publication, shadow isolation and end-of-record draining without labels.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.scripts.evaluate_entity_ctc_fusion import definitions
from system.entity_audio_repair import EntityAudioRepair
from system.entity_ctc_sidecar import EntityCtcRuntime


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


class SavedPrimary:
    def __init__(self, trace):
        self.trace = trace
        self.chunk_index = 0
        self.sessions = {}
        self.starts = 0

    async def __call__(self, endpoint, sid=None, data=b'', params=None):
        if endpoint == '/stream/start':
            self.starts += 1
            sid = str(self.starts)
            replay = bool((params or {}).get('context'))
            self.sessions[sid] = replay
            return dict(session_id=sid, context_applied=replay)
        if endpoint == '/stream/cancel':
            self.sessions.pop(sid, None)
            return dict(cancelled=True)
        if endpoint == '/stream/chunk':
            if self.sessions[sid]:
                text = self.trace['final_payload'].get('text', '')
                return dict(text=text, completed_segments=[dict(segment_id=1, text=text)])
            payload = self.trace['chunk_payloads'][self.chunk_index]['payload']
            self.chunk_index += 1
            return payload
        if endpoint == '/stream/finish':
            if self.sessions[sid]:
                text = self.trace['final_payload'].get('text', '')
                return dict(text=text, completed_segments=[])
            return self.trace['final_payload']
        raise AssertionError(endpoint)


async def run_sample(row, trace, expected, runtime, mode):
    pcm = Path(row['pcm_path']).read_bytes()
    assert hashlib.sha256(pcm).hexdigest() == row['pcm_sha256']
    assert sum(c['data_bytes'] for c in trace['chunk_payloads']) == len(pcm)
    defs = definitions(row['library_path'])
    ctc = await asyncio.to_thread(runtime.create_session, defs)
    backend = SavedPrimary(trace)
    guard = EntityAudioRepair(backend, defs, ctc_session=ctc, ctc_mode=mode)
    start = await guard.request('/stream/start', params=dict(language='Chinese', chunk_seconds=2.))
    offset = 0
    for chunk in trace['chunk_payloads']:
        size = chunk['data_bytes']
        await guard.request('/stream/chunk', start['session_id'], pcm[offset:offset+size])
        offset += size
        await asyncio.sleep(0)
    # Archived packets are replayed faster than real time. Let the ordered
    # CTC worker reach the final real-audio boundary before issuing finish,
    # as it would during a wall-clock paced recording.
    if ctc._futures:
        await asyncio.wrap_future(ctc._futures[-1])
    final = await guard.request('/stream/finish', start['session_id'])
    output = final.get('text', '')
    raw = trace['final_payload'].get('text', '')
    stats = guard.snapshot()
    return dict(id=row['id'], mode=mode, raw=raw, expected=expected, output=output,
                equals_archived_hybrid=output==expected,
                equals_original=output==raw,
                ctc=stats['incremental']['ctc'],
                replay_sessions=backend.starts-1,
                final_revision=final.get('entity_audio_repair_revision', 0))


async def main(run, dest, limit):
    inputs={r['id']:r for r in rows(run/'inputs.jsonl')}
    traces={r['id']:r for r in rows(run/'qwen_primary_streams.jsonl')}
    selected=[r for r in rows(run/'hybrid_results.jsonl') if r['split']=='test'
              and r['minimum_gap']==-2. and r['minimum_margin']==.1
              and r['changes'] and inputs[r['id']]['duration_seconds']<=16]
    selected=sorted(selected,key=lambda r:r['id'])[:limit]
    modeldir=run/'models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30'
    runtime=EntityCtcRuntime(modeldir)
    records=[]
    for result in selected:
        for mode in ('shadow','on'):
            item=await run_sample(inputs[result['id']],traces[result['id']],result['text'],runtime,mode)
            records.append(item)
            print(mode,item['id'],item['equals_archived_hybrid'],item['ctc']['errors'],flush=True)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(selected=len(selected),records=records),ensure_ascii=False,indent=2)+'\n')
    assert all(r['equals_original'] for r in records if r['mode']=='shadow')
    assert all(r['ctc']['errors']==0 for r in records)
    print('VERIFIED',len(selected),'recordings',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=8)
    a=p.parse_args()
    asyncio.run(main(a.run_dir.resolve(),a.output.resolve(),a.limit))
