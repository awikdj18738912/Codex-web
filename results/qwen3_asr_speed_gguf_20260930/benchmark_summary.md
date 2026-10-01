# Qwen3-ASR 1.7B: AudioCpp Q8_0 GGUF vs vLLM BF16

Fresh run on 2026-09-30. This report uses only the run recorded in this directory; previous benchmark result files were not used.

## Result

| Runtime | Weights | Measured time | RTF | Speed vs real time |
|---|---|---:|---:|---:|
| AudioCpp | Q8_0 GGUF | 50.15s client (49.80s server) | 0.02381 | 42.00x |
| vLLM | BF16 safetensors | 37.76s API calls | 0.01793 | 55.78x |

In this run, vLLM's measured time was 1.33x faster (24.7% less) than AudioCpp's client time. This is not a same-precision comparison: AudioCpp used Q8_0 GGUF, while vLLM used the local BF16 safetensors checkpoint.

Both processed 73 identical chunks covering 2106.35 seconds of audio (about 35.1 minutes).

## Method

- AudioCpp model: `/home/aim0/data/models/ASR/audiocpp-qwen3asr-q8-benchmark-20260929/qwen3-asr-1.7b-q8_0.gguf`.
- vLLM model: `/home/aim0/data/models/ASR/Qwen3-ASR-1.7B`, BF16 safetensors.
- Dedicated physical GPU 0 (NVIDIA GeForce RTX 3090); GPU 1 was not used by this run.
- Same 16 kHz mono PCM16 WAV files and same 73 external, non-overlapping chunks of at most 30 seconds as the fresh BF16 run.
- Requests were sequential. Chinese was forced, context was empty, and both paths used a 512-token maximum. One first-chunk warm-up per runtime was excluded.
- AudioCpp client time sums localhost multipart request duration; the server separately reported 49.80 seconds of task wall time. vLLM time sums `Qwen3ASRModel.transcribe` calls. Model initialization is excluded.
- AudioCpp used the CUDA backend with 8 threads, from source commit `30ef302`. vLLM was `0.14.0` with `qwen-asr` `0.0.6`; its initialization took 52.80 seconds.

## Limits

- This is a speed run, not an accuracy evaluation. Output text differed. On recording 3, AudioCpp returned 1,134 characters and vLLM returned 618; this can affect decode time.
- The vLLM path in this run loads the HF safetensors checkpoint. It did not load the AudioCpp GGUF file, so weight precision and file format differ.
- One measured pass per runtime was collected after warm-up. Cold-start time is not compared because the load boundaries differ.

## Artifacts

- `comparison.json`: combined summary and relative speed calculation.
- `audiocpp_results.json` / `vllm_results.json`: per-recording and total timings.
- `audiocpp_results.jsonl` / `vllm_results.jsonl`: per-chunk timings and transcripts.
- `chunks_manifest.json`, `audio/`, `chunks/`: exact inputs used.
- `run_audiocpp_benchmark.py`, `run_vllm_benchmark.py`, `audiocpp_server.json`: benchmark harnesses and AudioCpp server config.
