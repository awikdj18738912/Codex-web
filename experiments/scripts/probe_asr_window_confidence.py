"""Actual isolated Qwen streaming probe; no service restart or gate change."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from system.asr_window_confidence import QwenWindowConfidenceDecoder, project_window_scores
from system.qwen_confidence import QwenVLLMConfidenceDecoder


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main(args):
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Authorized GPU required')
    config=dict(model=str(args.model),physical_gpu=args.physical_gpu,
        cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        versions={name:importlib.metadata.version(name) for name in ['qwen-asr','vllm','transformers','torch']},
        streaming=dict(chunk_size_seconds=2.,unfixed_chunk_num=2,unfixed_token_num=5,max_new_tokens=32),
        vllm=dict(gpu_memory_utilization=args.gpu_memory_utilization,max_model_len=8192,enforce_eager=True,max_num_seqs=1),
        no_production_service_restart=True,no_refiner_calls=True,no_gate_changes=True,
        scope='isolated_real_ASR_token_capture_and_project_partition_not_live_VAD_reproduction',
        model_metadata_sha256={name:sha(args.model/name) for name in ['config.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors.index.json']})
    write(out/'configuration.json',config)
    (out/'executed_code').mkdir()
    for path in [Path(__file__),ROOT/'system/asr_window_confidence.py',ROOT/'system/qwen_confidence.py',ROOT/'system/window_refinement.py',ROOT/'system/chunking.py']:
        (out/'executed_code'/path.name).write_bytes(path.read_bytes())
    print('LOADING_QWEN',flush=True);started=time.perf_counter()
    asr=Qwen3ASRModel.LLM(model=str(args.model),max_new_tokens=32,dtype='bfloat16',
        gpu_memory_utilization=args.gpu_memory_utilization,max_model_len=8192,enforce_eager=True,max_num_seqs=1,max_num_batched_tokens=8192)
    decoder=QwenWindowConfidenceDecoder(asr,top_logprobs=5)
    baseline=QwenVLLMConfidenceDecoder(asr,top_logprobs=5)
    model_load_seconds=time.perf_counter()-started
    print('MODEL_READY',model_load_seconds,flush=True)
    cases=[]
    for number,path in enumerate(args.audio):
        case=out/f'case_{number:02d}';case.mkdir()
        pcm,rate=sf.read(path,dtype='float32')
        if rate!=16000 or pcm.ndim!=1:
            raise ValueError('Probe requires existing 16k mono WAV, no implicit resampling')
        if number>0:
            pcm=pcm[:30*rate]
        provenance=dict(audio_path=str(path.resolve()),audio_sha256=sha(path),used_samples=len(pcm),
            used_audio_seconds=len(pcm)/rate,used_float32_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
            initial_state_new=True,VAD_segmentation=False,utterance_boundary='one isolated audio case')
        write(case/'audio_provenance.json',provenance)
        state=decoder.init_streaming_state(language='Chinese',chunk_size_sec=2.,unfixed_chunk_num=2,unfixed_token_num=5)
        step=2*rate
        traces=[];timings=[];stream_texts=[]
        for index,start in enumerate(range(0,len(pcm),step)):
            before=getattr(state,'_window_score_ledger',None)
            previous=before.revision if before is not None else 0
            call=time.perf_counter();decoder.streaming_transcribe(pcm[start:start+step],state)
            timings.append((time.perf_counter()-call)*1000)
            ledger=getattr(state,'_window_score_ledger',None)
            if ledger is not None and ledger.revision!=previous:
                projection=state._window_confidence_projection
                groups=project_window_scores(projection,final=False)
                traces.append(dict(audio_end_seconds=min(start+step,len(pcm))/rate,
                    generation=state._window_generation_trace,projection=projection,project_partition=groups))
            stream_texts.append(state.text)
            if (index+1)%5==0:
                print('PROGRESS',number,index+1,'text_chars',len(state.text),flush=True)
        before=state._window_score_ledger.revision
        decoder.finish_streaming_transcribe(state)
        if state._window_score_ledger.revision!=before:
            traces.append(dict(audio_end_seconds=len(pcm)/rate,generation=state._window_generation_trace,
                projection=state._window_confidence_projection,
                project_partition=project_window_scores(state._window_confidence_projection,final=False)))
        final=project_window_scores(state._window_confidence_projection,final=True,source_chunks=[state.text])
        write(case/'final.json',final)
        with (case/'stream_trace.jsonl').open('w') as stream:
            for trace in traces:
                stream.write(json.dumps(trace,ensure_ascii=False)+'\n')
        reference_equal=None;all_updates_equal=None
        if number==0:
            reference=baseline.init_streaming_state(language='Chinese',chunk_size_sec=2.,unfixed_chunk_num=2,unfixed_token_num=5)
            reference_updates=[]
            for start in range(0,len(pcm),step):
                baseline.streaming_transcribe(pcm[start:start+step],reference)
                reference_updates.append(reference.text)
            baseline.finish_streaming_transcribe(reference)
            reference_equal=reference.text==state.text
            all_updates_equal=reference_updates==stream_texts
            write(case/'baseline_transcript_comparison.json',dict(baseline_text=reference.text,scored_text=state.text,
                exact_final_equal=reference_equal,all_stream_updates_equal=all_updates_equal,
                same_model_sampling_parameters=True,VAD_not_in_this_probe=True))
        item=dict(case=number,**provenance,text_characters=len(state.text),decode_updates=len(traces),
            punctuation_fragments=len(final['fragments']),windows=len(final['windows']),
            fully_scored_fragments=sum(f['confidence']['score'] is not None for f in final['fragments']),
            fully_scored_windows=sum(w['confidence']['score'] is not None for w in final['windows']),
            final_tail_fragments=final['windows'][-1]['fragment_count'] if final['windows'] else 0,
            min_fragment_coverage=min((f['confidence']['coverage'] for f in final['fragments'] if f['confidence']['coverage'] is not None),default=None),
            baseline_final_equal=reference_equal,baseline_all_stream_updates_equal=all_updates_equal,
            stream_calls_mean_ms=sum(timings)/len(timings),
            evidence_keeps_generation_history=True,score_calibrated=False,production_skip_allowed=False)
        cases.append(item);write(out/'progress.json',dict(cases=cases))
        print('CASE_DONE',json.dumps(item,ensure_ascii=False),flush=True)
    summary=dict(cases=cases,model_load_seconds=model_load_seconds,
        scope=config['scope'],no_production_service_restart=True,no_refiner_calls=True,
        calibrated=False,production_skip_allowed=False,
        transcription_accuracy_or_gate_benefit_not_evaluated=True)
    write(out/'summary.json',summary)
    print('DONE',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('/home/aim0/data/models/ASR/Qwen3-ASR-1.7B'))
    parser.add_argument('--physical-gpu',type=int,default=1)
    parser.add_argument('--gpu-memory-utilization',type=float,default=.42)
    parser.add_argument('--audio',type=Path,nargs='+',default=[
        ROOT/'results/qwen3_asr_speed_fresh_20260929/audio/test_1_16k_mono.wav',
        ROOT/'results/qwen3_asr_speed_fresh_20260929/chunks/test_3_16k_mono_chunk_000.wav',
        ROOT/'results/oncology_podcast_20261008/audio/oncology_podcast_0000.wav'])
    main(parser.parse_args())
