#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/aim0/桌面/tan/ASR/Codex-web"
RUN_DIR="${PROJECT_ROOT}/results/refiner13_qwen25_lora_epoch3_20261027"
CONFIG="${PROJECT_ROOT}/experiments/llamafactory/refiner13_qwen25_lora_epoch3_20261027.yaml"
OUTPUT="/home/aim0/data/ASR/training/qwen25_05b_refiner13_lora_epoch3_20261027"
cd "${PROJECT_ROOT}" || exit 2
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

date --iso-8601=seconds > "${RUN_DIR}/train.started"
"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/llamafactory-cli" train "${CONFIG}" > "${RUN_DIR}/train.log" 2>&1
train_exit_code=$?
printf '%s %s\n' "${train_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/train.exit"
if (( train_exit_code != 0 )); then
  exit "${train_exit_code}"
fi

date --iso-8601=seconds > "${RUN_DIR}/eval.started"
"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python" \
  "${PROJECT_ROOT}/experiments/scripts/evaluate_refiner13_qwen.py" \
  --adapter "${OUTPUT}" --skip-base --output "${RUN_DIR}/evaluation" \
  > "${RUN_DIR}/eval.log" 2>&1
eval_exit_code=$?
printf '%s %s\n' "${eval_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/eval.exit"
exit "${eval_exit_code}"
