#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
exec /home/aim0/data/conda/envs/agentic-asr/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/aim0/data/ASR/models/Qwen3.8-27B-AWQ-INT4 \
  --served-model-name qwen38-teacher --host 127.0.0.1 --port 8012 \
  --language-model-only --max-model-len 3072 --max-num-seqs 8 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization "${QWEN38_GPU_UTIL:-0.90}" \
  --cpu-offload-gb "${QWEN38_OFFLOAD_GB:-0}" --dtype bfloat16 --enforce-eager --reasoning-parser qwen3
