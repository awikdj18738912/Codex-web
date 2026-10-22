#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
exec /home/aim0/data/conda/envs/agentic-asr/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/aim0/data/ASR/models/Qwen3.8-27B-AWQ-INT4 \
  --served-model-name qwen38-teacher --host 127.0.0.1 --port 8012 \
  --language-model-only --max-model-len 4096 --max-num-seqs 4 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.80 \
  --cpu-offload-gb 3 --dtype bfloat16 --enforce-eager --reasoning-parser qwen3
