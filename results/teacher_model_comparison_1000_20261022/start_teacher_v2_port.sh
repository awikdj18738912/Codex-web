#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec /home/aim0/data/conda/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/aim0/data/models/ASR/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit \
  --served-model-name refiner-teacher --host 127.0.0.1 --port 8011 \
  --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 4 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.92 \
  --dtype bfloat16 --enforce-eager
