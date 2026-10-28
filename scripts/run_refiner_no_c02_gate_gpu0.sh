#!/usr/bin/env bash
set -u
PROJECT_ROOT="/home/aim0/桌面/tan/ASR/Codex-web"
RUN_DIR="${PROJECT_ROOT}/results/refiner13_no_C02_gate_20261027"
cd "${PROJECT_ROOT}" || exit 2
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES=0
export PREGATE_DEVICE=cuda:0
export PREGATE_DATA=/home/aim0/data/datasets/ASR/refiner13_qwen25_lora_20261027_no_C02
export PREGATE_OUTPUT=/home/aim0/data/ASR/training/refiner_noC02_pregate_gpu0_20261027
export PREGATE_REFINER_PREDICTIONS="${PROJECT_ROOT}/results/refiner13_no_C02_20261027/evaluation/no_C02_test.jsonl"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python" -u - <<'PY'
import os,json
from pathlib import Path
import torch
assert torch.cuda.device_count()==1
free,_=torch.cuda.mem_get_info(0)
assert free>5*1024**3,'Need at least 5 GiB free for concurrent tiny gate training'
data=Path(os.environ['PREGATE_DATA'])
rows=[json.loads(s) for s in (data/'split_audit.jsonl').open()]
assert all(r['category']!='C02' for r in rows)
assert Path(os.environ['PREGATE_REFINER_PREDICTIONS']).exists()
assert not Path(os.environ['PREGATE_OUTPUT']).exists(),'Output already exists'
print('Gate GPU0 preflight free GiB',free/1024**3,flush=True)
PY
preflight_code=$?
if (( preflight_code != 0 )); then exit "${preflight_code}"; fi
date --iso-8601=seconds > "${RUN_DIR}/train.started"
"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python" -u \
  "${PROJECT_ROOT}/experiments/scripts/train_refiner13_pregate.py" > "${RUN_DIR}/train.log" 2>&1
training_exit_code=$?
printf '%s %s\n' "${training_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/train.exit"
exit "${training_exit_code}"
