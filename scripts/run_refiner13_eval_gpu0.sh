#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/aim0/桌面/tan/ASR/Codex-web"
RUN_DIR="${PROJECT_ROOT}/results/refiner13_qwen25_lora_20261026"
ADAPTER="/home/aim0/data/ASR/training/qwen25_05b_refiner13_lora_20261026_retry3/checkpoint-1072"
cd "${PROJECT_ROOT}" || exit 2
mkdir -p "${RUN_DIR}/evaluation"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

date --iso-8601=seconds > "${RUN_DIR}/evaluation.started"
"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python" \
  "${PROJECT_ROOT}/experiments/scripts/evaluate_refiner13_qwen.py" \
  --adapter "${ADAPTER}" --output "${RUN_DIR}/evaluation" \
  > "${RUN_DIR}/evaluation.log" 2>&1
evaluation_exit_code=$?
printf '%s %s\n' "${evaluation_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/evaluation.exit"
exit "${evaluation_exit_code}"
