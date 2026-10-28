#!/usr/bin/env bash
set -u
PROJECT_ROOT="/home/aim0/桌面/tan/ASR/Codex-web"
RUN_DIR="${PROJECT_ROOT}/results/refiner13_pregate_20261027"
cd "${PROJECT_ROOT}" || exit 2
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
date --iso-8601=seconds > "${RUN_DIR}/train.started"
"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python" -u \
  "${PROJECT_ROOT}/experiments/scripts/train_refiner13_pregate.py" > "${RUN_DIR}/train.log" 2>&1
training_exit_code=$?
printf '%s %s\n' "${training_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/train.exit"
exit "${training_exit_code}"
