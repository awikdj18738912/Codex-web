# Qwen3-ASR 1.7B controlled BF16 API comparison

Fresh audit on 2026-09-30 after identifying differences in the prior benchmark method.

## Result

| Runtime | Weight | Three pass times | Median |
|---|---|---|---:|
| AudioCpp | BF16 safetensors | 59.51s, 59.82s, 59.84s | 59.82s |
| vLLM | BF16 safetensors | 40.86s, 36.24s, 36.22s | 36.24s |

By median multipart client time, vLLM was **1.65x faster** (39.4% less elapsed time). AudioCpp also reports its internal server wall time: 59.24s, 59.56s, and 59.59s; its median is 1.64x the vLLM API median.

This audit uses the exact same local BF16 safetensors checkpoint on both runtimes. It corrects the GGUF comparison’s Q8_0-versus-BF16 difference and the earlier benchmark’s different timing interfaces.

## Method

- Same model path: `/home/aim0/data/models/ASR/Qwen3-ASR-1.7B`. Both safetensors shards were checked and contain BF16 tensors.
- Same NVIDIA GeForce RTX 3090 physical GPU 0; GPU 1 was not used by this audit.
- Both runtimes received the same 73 pre-split 16 kHz mono PCM16 WAV chunks, sequentially, in three full measured passes. Each pass covered 2106.35 seconds of audio.
- Both used `POST /v1/audio/transcriptions` with multipart WAV uploads. The same client harness timed both requests; audio file bytes were read before the timer started.
- Chinese was forced and the output limit was 512 tokens. One warm-up request per runtime was excluded.
- AudioCpp: CUDA backend, 8 threads, source commit `30ef302`.
- vLLM: `0.14.0`, `qwen-asr` `0.0.6`, BF16, `max_num_seqs=1`, `gpu_memory_utilization=0.80`.

## Observations

- AudioCpp's three rounds were stable, with total transcript length of 8700 characters each.
- vLLM's first round took longer (40.86s) and returned 9427 characters. Its second and third rounds took 36.24s and 36.22s, returned 8723 characters each, and produced identical per-chunk transcripts. The reported vLLM median includes the first round.
- This measures speed, not recognition accuracy. The two runtimes' transcript strings are not identical.

## Artifacts

- `comparison.json`: machine-readable configuration and comparison.
- `audiocpp_api_results.json` / `vllm_api_results.json`: per-round totals.
- `audiocpp_api_results.jsonl` / `vllm_api_results.jsonl`: per-chunk times and transcripts.
- `chunks_manifest.json`, `audio/`, `chunks/`: exact inputs used.
- `run_server_api_benchmark.py`, `audiocpp_server.json`: benchmark client and AudioCpp configuration.
