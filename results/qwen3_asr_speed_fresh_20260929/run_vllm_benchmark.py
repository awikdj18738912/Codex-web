from __future__ import annotations
import json, pathlib, time
import torch
from qwen_asr import Qwen3ASRModel


def main():
    ROOT = pathlib.Path(__file__).resolve().parent
    MODEL = '/home/aim0/data/models/ASR/Qwen3-ASR-1.7B'
    manifest = json.loads((ROOT / 'chunks_manifest.json').read_text())
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in the selected environment')
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f'Expected one visible GPU, got {torch.cuda.device_count()}')
    print(f'Loading vLLM Qwen3-ASR on {torch.cuda.get_device_name(0)}', flush=True)
    load_start = time.perf_counter()
    model = Qwen3ASRModel.LLM(
        model=MODEL,
        dtype='bfloat16',
        gpu_memory_utilization=0.80,
        max_num_seqs=1,
        max_inference_batch_size=1,
        max_new_tokens=512,
    )
    load_elapsed = time.perf_counter() - load_start
    print(f'Model initialized in {load_elapsed:.2f}s', flush=True)

    def transcribe(path: pathlib.Path):
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        result = model.transcribe(audio=str(path), language='Chinese', context='', return_time_stamps=False)
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - started
        item = result[0]
        return elapsed, str(item.text), str(getattr(item, 'language', 'Chinese'))

    first = ROOT / 'chunks' / manifest[0]['file']
    warm_start = time.perf_counter()
    warm_elapsed, warm_text, warm_language = transcribe(first)
    warm_record = {'file': first.name, 'call_elapsed_s': warm_elapsed, 'outer_elapsed_s': time.perf_counter()-warm_start, 'language': warm_language, 'text': warm_text}
    (ROOT / 'vllm_warmup.json').write_text(json.dumps(warm_record, ensure_ascii=False, indent=2)+'\n')
    print(f'Warmup complete in {warm_elapsed:.3f}s', flush=True)

    records=[]
    partial=ROOT/'vllm_results.jsonl'
    partial.write_text('')
    for i,item in enumerate(manifest,1):
        path=ROOT/'chunks'/item['file']
        elapsed,text,language=transcribe(path)
        record={**item,'call_elapsed_s':elapsed,'language':language,'text':text}
        records.append(record)
        with partial.open('a') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
        if i%10==0 or i==len(manifest):
            print(f'vLLM: {i}/{len(manifest)} chunks, cumulative_call_s={sum(x["call_elapsed_s"] for x in records):.2f}',flush=True)

    audio_s=sum(r['duration_s'] for r in records)
    wall_s=sum(r['call_elapsed_s'] for r in records)
    summary={
     'runtime':'qwen-asr vLLM Python API',
     'model':MODEL,
     'weight_dtype':'BF16 safetensors',
     'vllm_version':__import__('vllm').__version__,
     'qwen_asr_version':__import__('importlib.metadata').metadata.version('qwen-asr'),
     'gpu':'physical GPU 0, NVIDIA GeForce RTX 3090, CUDA_VISIBLE_DEVICES=0',
     'vllm_config':{'dtype':'bfloat16','gpu_memory_utilization':0.8,'max_num_seqs':1,'max_inference_batch_size':1},
     'audio_format':'16 kHz mono PCM16 WAV; external non-overlapping 30 s chunks',
     'language':'Chinese','max_tokens':512,
     'model_load_s':load_elapsed,'warmup':warm_record,
     'measured_chunks':len(records),'total_audio_s':audio_s,'total_call_elapsed_s':wall_s,
     'call_rtf':wall_s/audio_s,'call_speed_x_realtime':audio_s/wall_s,'per_source':{}
    }
    for source in sorted({r['source'] for r in records}):
     group=[r for r in records if r['source']==source]
     duration=sum(r['duration_s'] for r in group); elapsed=sum(r['call_elapsed_s'] for r in group)
     summary['per_source'][source]={'chunks':len(group),'audio_s':duration,'call_elapsed_s':elapsed,'rtf':elapsed/duration,'speed_x_realtime':duration/elapsed,'transcript':''.join(r['text'] for r in group)}
    (ROOT/'vllm_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('vllm_version','qwen_asr_version','model_load_s','measured_chunks','total_audio_s','total_call_elapsed_s','call_rtf','call_speed_x_realtime')},indent=2))


if __name__ == "__main__":
    main()
