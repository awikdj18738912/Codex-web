"""Isolated stateful CTC evidence collection; inference never opens labels.

Qwen's real no-context primary streams are held fixed for paired quality
comparison. The CTC model consumes unchanged audio in new 160 ms increments.
This quality replay is not an end-to-end concurrent latency measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from entity_ctc_frontend import StreamingCtcModel
from entity_ctc_word_spotter import StreamingWordSpotter


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []


def words(path):
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
        return [r[0] for r in db.execute('SELECT canonical_text FROM protected_entities WHERE enabled=1 ORDER BY id')]


def collect(out,split,limit,shard_index=0,num_shards=1):
    modeldir=out/'models/sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30'
    began=time.perf_counter()
    model=StreamingCtcModel(modeldir/'model.int8.onnx',modeldir/'tokens.txt',num_threads=2)
    load_seconds=time.perf_counter()-began
    selected=[r for r in rows(out/'inputs.jsonl') if split=='all' or r['split']==split]
    selected=selected[shard_index::num_shards]
    if limit:selected=selected[:limit]
    dest=out/(f'ctc_results_shard{shard_index}.jsonl' if num_shards>1 else 'ctc_results.jsonl')
    old=rows(out/'ctc_results.jsonl')
    if dest.name!='ctc_results.jsonl':old+=rows(dest)
    done={r['id'] for r in old};assert len(done)==len(old)
    templates={};started=time.perf_counter()
    for i,s in enumerate(selected):
        if s['id'] in done:continue
        pcm=Path(s['pcm_path']).read_bytes();assert hashlib.sha256(pcm).hexdigest()==s['pcm_sha256']
        audio=np.frombuffer(pcm,dtype='<f4');library=s['library_path'];build=0.
        if library not in templates:
            before=time.perf_counter()
            templates[library]=StreamingWordSpotter(model.tokens,words(library),context_weight=2.,
                beam=12.,minimum_gap=-3.,blank_id=model.blank_id,frame_seconds=model.frame_shift_seconds)
            build=time.perf_counter()-before
        template=templates[library]
        spotter=template.create_stream() if hasattr(template,'create_stream') else StreamingWordSpotter(
            model.tokens,words(library),context_weight=2.,beam=12.,minimum_gap=-3.,blank_id=model.blank_id)
        stream=model.create_stream();events=[];offsets=[];before=time.perf_counter()
        def consume(blocks):
            for block in blocks:
                offsets.append((block['frame_offset'],len(block['log_probs'])))
                hyps=spotter.accept(block['log_probs'],block['frame_offset'],block['available_audio_seconds'])
                # Retain actual new/improved acoustic hypotheses at each block.
                events.append(dict(audio_seconds=block['available_audio_seconds'],
                    after_input_finished=block['after_input_finished'],ctc_text=block['text'],
                    ctc_tokens=block['token_ids'],ctc_frame_times=stream.token_frames.copy(),
                    hypotheses=hyps,compute_elapsed=time.perf_counter()-before))
        for offset in range(0,len(audio),2560):consume(stream.accept_audio(audio[offset:offset+2560]))
        before_finish=time.perf_counter();consume(stream.finish());finish_seconds=time.perf_counter()-before_finish
        hyps=spotter.finish();elapsed=time.perf_counter()-before
        coverage=spotter.stats['coverage']
        if not (out/'ctc_libraries').exists():(out/'ctc_libraries').mkdir()
        coverage_file=out/'ctc_libraries'/(Path(library).stem+'.json')
        if not coverage_file.exists():coverage_file.write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
        compact_stats={k:v for k,v in spotter.stats.items() if k!='coverage'}
        record=dict(id=s['id'],split=s['split'],domain=s['domain'],library_path=library,
            pcm_sha256=s['pcm_sha256'],audio_seconds=len(audio)/16000,ctc_text=stream.text,
            ctc_tokens=stream.token_ids,ctc_token_frames=stream.token_frames,
            ctc_frontend=stream.stats,spotter=compact_stats,coverage_file=str(coverage_file),
            requested_words=coverage['requested_words'],covered_words=coverage['covered_words'],
            hypotheses=hyps,events=events,compute_seconds=elapsed,finish_seconds=finish_seconds,
            graph_build_seconds_first_use=build,model_load_seconds=load_seconds,
            frame_offsets=offsets,quality_replay_only=True,
            timing_scope='CPU2threads actual incremental CTC+wordspotter compute, accelerated audio; Qwen replay latency excluded')
        with dest.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        done.add(s['id'])
        if i%25==0 or i==len(selected)-1:
            progress=dict(split=split,completed=sum(x['id'] in done for x in selected),expected=len(selected),
                last_id=s['id'],last_seconds=round(elapsed,3),elapsed_seconds=round(time.perf_counter()-started,3))
            progress_name=f'ctc_progress_shard{shard_index}.json' if num_shards>1 else 'ctc_progress.json'
            (out/progress_name).write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(progress,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--split',choices=['dev','test','control','all'],default='dev');p.add_argument('--limit',type=int)
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args()
    if not 0<=a.shard_index<a.num_shards:p.error('invalid shard')
    collect(a.run_dir.resolve(),a.split,a.limit,a.shard_index,a.num_shards)
