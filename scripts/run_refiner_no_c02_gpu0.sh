#!/usr/bin/env bash
set -u
PROJECT_ROOT="/home/aim0/桌面/tan/ASR/Codex-web"
RUN_DIR="${PROJECT_ROOT}/results/refiner13_no_C02_20261027"
CONFIG="${PROJECT_ROOT}/experiments/llamafactory/refiner13_qwen25_lora_no_C02_20261027.yaml"
TRAIN_PYTHON="${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python"
cd "${PROJECT_ROOT}" || exit 2
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
"${TRAIN_PYTHON}" -u - "${CONFIG}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import yaml,torch
config_path=Path(sys.argv[1]);config=yaml.safe_load(config_path.read_text())
preflight=json.loads(Path('results/refiner13_no_C02_20261027/preflight.json').read_text())
assert preflight['status']=='passed'
assert preflight['config_sha256']==hashlib.sha256(config_path.read_bytes()).hexdigest()
data=Path(config['dataset_dir']);manifest=json.loads((data/'manifest.json').read_text())
assert preflight['dataset_manifest_sha256']==hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
for name,sha in manifest['files_sha256'].items():
    assert hashlib.sha256((data/name).read_bytes()).hexdigest()==sha,name
output=Path(config['output_dir'])
assert not output.exists() or not any(output.iterdir()),'Output already populated'
assert torch.cuda.device_count()==1
free,total=torch.cuda.mem_get_info(0)
assert free>6*1024**3,'Insufficient GPU memory'
print('GPU0 preflight free GiB',free/1024**3,flush=True)
PY
preflight_code=$?
if (( preflight_code != 0 )); then exit "${preflight_code}"; fi
date --iso-8601=seconds > "${RUN_DIR}/train.started"
"${TRAIN_PYTHON}" -u -c 'from llamafactory.cli import main; main()' train "${CONFIG}" > "${RUN_DIR}/train.log" 2>&1
training_exit_code=$?
printf '%s %s\n' "${training_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/train.exit"
if (( training_exit_code != 0 )); then exit "${training_exit_code}"; fi
date --iso-8601=seconds > "${RUN_DIR}/evaluation.started"
"${TRAIN_PYTHON}" -u "${PROJECT_ROOT}/experiments/scripts/evaluate_refiner_no_c02.py" > "${RUN_DIR}/evaluation.log" 2>&1
evaluation_exit_code=$?
printf '%s %s\n' "${evaluation_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/evaluation.exit"
exit "${evaluation_exit_code}"
