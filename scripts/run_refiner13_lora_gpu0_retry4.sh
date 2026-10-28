#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/aim0/桌面/tan/ASR/Codex-web"
RUN_DIR="${PROJECT_ROOT}/results/refiner13_qwen25_lora_20261026"
CONFIG="${PROJECT_ROOT}/experiments/llamafactory/refiner13_qwen25_lora_20261026_retry3.yaml"
TRAIN_LOG="${RUN_DIR}/train_retry4.log"
EXIT_FILE="${RUN_DIR}/train_retry4.exit"

cd "${PROJECT_ROOT}" || exit 2
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

date --iso-8601=seconds > "${RUN_DIR}/train_retry4.started"
"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/llamafactory-cli" train "${CONFIG}" > "${TRAIN_LOG}" 2>&1
train_exit_code=$?
printf '%s %s\n' "${train_exit_code}" "$(date --iso-8601=seconds)" > "${EXIT_FILE}"
exit "${train_exit_code}"
