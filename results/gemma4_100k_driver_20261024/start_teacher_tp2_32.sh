#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/home/aim0/data/conda/envs/agentic-asr/lib/python3.12/site-packages/nvidia/cu13
exec /home/aim0/data/conda/envs/agentic-asr/bin/python -u \
  -m vllm.entrypoints.openai.api_server \
  --model /home/aim0/data/ASR/models/gemma-4-31B-it-qat-w4a16-ct \
  --served-model-name gemma4-31b-teacher --host 127.0.0.1 --port 8014 \
  --language-model-only --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 3072 --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.94 --kv-cache-memory-bytes 10737418240 \
  --reasoning-parser gemma4 --enforce-eager --disable-custom-all-reduce
