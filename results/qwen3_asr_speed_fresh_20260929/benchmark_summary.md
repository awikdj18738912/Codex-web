# Qwen3-ASR 1.7B fresh speed comparison

Fresh run on 2026-09-29. The earlier benchmark outputs were not used.

## Result

| Runtime | Total request/inference time | RTF | Throughput |
|---|---:|---:|---:|
| AudioCpp BF16 | 58.92s (server wall 58.53s) | 0.0280 | 35.75x realtime |
| vLLM BF16 | 38.27s | 0.0182 | 55.04x realtime |

vLLM was **1.54x faster** on this run, with **35.1% less measured time** than AudioCpp.

Both processed 73 chunks covering 2106.35s of audio (about 35.1 minutes).

## Per recording

| Input | Audio duration | AudioCpp | vLLM | vLLM speed ratio |
|---|---:|---:|---:|---:|
| 测试音频1.mp3 | 40.8s | 1.16s | 0.78s | 1.50x |
| 测试音频2.mp3 | 175.1s | 3.82s | 2.40s | 1.59x |
| 测试音频3.mp3 | 322.7s | 9.46s | 6.12s | 1.54x |
| 测试音频4.mp3 | 724.4s | 19.54s | 12.65s | 1.55x |
| 测试音频5.mp3 | 843.3s | 24.95s | 16.32s | 1.53x |

## Method

- Same local checkpoint: `/home/aim0/data/models/ASR/Qwen3-ASR-1.7B`; BF16 safetensors for both runtimes.
- Dedicated physical GPU 0 (NVIDIA GeForce RTX 3090); GPU 1 was not used by this run.
- Five source recordings were converted once to 16 kHz mono PCM16 WAV and split into the same 73 non-overlapping chunks of at most 30 seconds.
- Each chunk was processed sequentially, with Chinese forced and empty context. AudioCpp used its 512-token default; vLLM was explicitly set to 512 max new tokens. One first-chunk warm-up per runtime was excluded from the totals.
- AudioCpp time is the summed localhost multipart request time; its server reported 58.53 seconds of summed task wall time. vLLM time is the summed `Qwen3ASRModel.transcribe` call time. Model initialization is excluded.
- Runtime versions: AudioCpp `AudioCpp server multipart transcription` (binary commit `30ef302`); vLLM `0.14.0`, `qwen-asr` `0.0.6`; vLLM used `max_num_seqs=1`, `max_inference_batch_size=1`, BF16, and `gpu_memory_utilization=0.80`.

## Limits

- This is a speed comparison, not a transcription-accuracy evaluation. Both returned non-empty output for all chunks, but outputs were not identical; on recording 3 the concatenated lengths were 1,133 characters for AudioCpp and 618 for vLLM. That can affect decode time, so use WER against a reference before choosing solely on quality-sensitive workloads.
- One measured pass per runtime was collected after warm-up. vLLM model initialization took 81.64 seconds, excluded from inference totals; cold-start time is not compared because AudioCpp used a lazily loaded server and the boundaries differ.

## Artifacts

- `comparison.json`: combined summary.
- `audiocpp_results.json` and `vllm_results.json`: per-recording results and totals.
- `audiocpp_results.jsonl` and `vllm_results.jsonl`: per-chunk timings and transcripts.
- `chunks_manifest.json`, `audio/`, `chunks/`: exact inputs used.
- `run_audiocpp_benchmark.py` and `run_vllm_benchmark.py`: harnesses for reproducing the run. Start AudioCpp with `audiocpp_server --config audiocpp_server.json --no-ui --backend cuda --device 0 --threads 8 --model-spec-override <repo>/audio.cpp/model_specs/qwen3_asr.json` on port 38761 before the AudioCpp harness.
