#!/usr/bin/env bash
set -u
PROJECT_ROOT="/home/aim0/桌面/tan/ASR/Codex-web"
RUN_DIR="${PROJECT_ROOT}/results/refiner_noC02_gate_rbt3_20261027"
cd "${PROJECT_ROOT}" || exit 2
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES=0
export PREGATE_DEVICE=cuda:0
export PREGATE_ENCODER=/home/aim0/data/models/ASR/TextGate/hfl-rbt3
export PREGATE_DATA=/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02
export PREGATE_OUTPUT=/home/aim0/data/ASR/training/refiner_noC02_gate_rbt3_20261027
export PREGATE_REFINER_PREDICTIONS="${PROJECT_ROOT}/results/refiner13_no_C02_20261027/evaluation/no_C02_test.jsonl"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
date --iso-8601=seconds > "${RUN_DIR}/train.started"
"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python" -u \
  "${PROJECT_ROOT}/experiments/scripts/train_refiner13_pregate.py" > "${RUN_DIR}/train.log" 2>&1
training_exit_code=$?
printf '%s %s\n' "${training_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/train.exit"
if (( training_exit_code != 0 )); then exit "${training_exit_code}"; fi
"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python" -u \
  "${PROJECT_ROOT}/experiments/scripts/compare_rbt3_gate.py" > "${RUN_DIR}/comparison.log" 2>&1
comparison_exit_code=$?
printf '%s %s\n' "${comparison_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/comparison.exit"
exit "${comparison_exit_code}"
