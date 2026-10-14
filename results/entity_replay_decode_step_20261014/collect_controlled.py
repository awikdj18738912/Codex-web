"""Same reconstructed ASR draft; actual replay audio, no gold at inference.

These elapsed times measure entity review only, not full-Web latency. The
first-pass hypotheses were reconstructed from earlier validated edits.
"""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'source_snapshot'))
from system.entity_audio_repair import EntityAudioRepair
from system.entity_store import EntityStore
from system.web_app import _stream_request_async

spec=importlib.util.spec_from_file_location('system._decode_controlled',OUT/'candidate.py')
candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)


class FixedPrimary:
    def __init__(self,source,asr_url):
        self.source=source;self.asr_url=asr_url;self.initial=False
        self.sid='fixed-primary';self.pcm=bytearray();self.calls=0
        self.model_ms=0.;self.replay_outputs=[];self.replay_starts=[]

    async def __call__(self,endpoint,sid=None,data=b'',params=None):
        if endpoint=='/stream/start' and not self.initial:
            self.initial=True
            return dict(session_id=self.sid,context_applied=False,chunk_seconds=2.)
        if sid==self.sid:
            if endpoint=='/stream/chunk':
                self.pcm.extend(data)
                return dict(text='',completed_segments=[])
            if endpoint=='/stream/finish':
                return dict(text=self.source,completed_segments=[dict(segment_id=1,text=self.source)])
            if endpoint=='/stream/cancel':return dict(cancelled=True)
            raise ValueError(endpoint)
        self.calls+=1
        if endpoint=='/stream/start':self.replay_starts.append(dict(params or {}))
        result=await _stream_request_async(self.asr_url,endpoint,sid,data,params)
        self.model_ms+=float((result.get('asr_timing') or {}).get('model_ms',0.) or 0.)
        if endpoint=='/stream/finish':
            self.replay_outputs.append(dict(text=result.get('text',''),session_id=sid))
        return result


async def main():
    inputs=OUT/'controlled_regression_inputs.jsonl';samples=[json.loads(l) for l in inputs.open()]
    worker=json.loads((OUT/'worker.json').read_text());args=worker['argv']
    asr_url=args[args.index('--asr-url')+1]
    metadata=json.loads((OUT/'library_metadata.json').read_text())['libraries']
    libraries={k:EntityStore(row['snapshot_path']).list_entities(domain='general') for k,row in metadata.items()}
    protocol=dict(input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((OUT/'candidate.py').read_bytes()).hexdigest(),
        source_hashes=worker['source_hashes'],sample_order=[s['id'] for s in samples],
        scope='Same frozen predicted source text and PCM; live ASR review, no first-pass model/refiner/browser cost. Not full-Web timing.',
        settings=dict(primary_decode_seconds=2,old_review_seconds=2,new_review_seconds=10,serial_pairs=True,transport_bytes=64000),gold_at_inference=False)
    pp=OUT/'controlled_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    dest=OUT/'controlled_results.jsonl';rows=[json.loads(l) for l in dest.open()] if dest.exists() else []
    done={(r['id'],r['arm']) for r in rows};assert len(done)==len(rows)
    started=time.perf_counter()
    for sample in samples:
        pcm=Path(sample['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==sample['pcm_sha256']
        assert hashlib.sha256(sample['source_text'].encode()).hexdigest()==sample['source_text_sha256']
        order=('old','new') if int(hashlib.sha256(sample['id'].encode()).hexdigest()[:2],16)%2 else ('new','old')
        for arm in order:
            if (sample['id'],arm) in done:continue
            backend=FixedPrimary(sample['source_text'],asr_url)
            cls=EntityAudioRepair if arm=='old' else candidate.EfficientEntityRepair
            repair=cls(backend,libraries[sample['library_key']],streaming_replay=False)
            before=time.perf_counter()
            ready=await repair.request('/stream/start',params=dict(language='Chinese',chunk_seconds='2'))
            for offset in range(0,len(pcm),64000):
                await repair.request('/stream/chunk',ready['session_id'],pcm[offset:offset+64000])
            final=await repair.request('/stream/finish',ready['session_id'])
            elapsed=time.perf_counter()-before
            assert bytes(backend.pcm)==pcm
            row=dict(id=sample['id'],arm=arm,library_key=sample['library_key'],corpus=sample['corpus'],cohort=sample['cohort'],
                pcm_sha256=sample['pcm_sha256'],source_text_sha256=sample['source_text_sha256'],source_text=sample['source_text'],old_change_count=sample['old_change_count'],
                entity_review_seconds=elapsed,backend_http_calls=backend.calls,reported_model_ms=backend.model_ms,replay_starts=backend.replay_starts,replay_outputs=backend.replay_outputs,result=final)
            with dest.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            done.add((sample['id'],arm))
            progress=dict(sessions=len(done),expected_sessions=len(samples)*2,last_id=sample['id'],last_arm=arm,elapsed_seconds=round(time.perf_counter()-started,2))
            (OUT/'controlled_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            if len(done)%20==0 or len(done)<5:print('PROGRESS',json.dumps(progress),flush=True)
    print('COMPLETE',len(done),flush=True)


if __name__=='__main__':asyncio.run(main())
