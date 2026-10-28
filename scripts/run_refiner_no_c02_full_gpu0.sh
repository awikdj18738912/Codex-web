#!/usr/bin/env bash
set -u
PROJECT_ROOT="/home/aim0/桌面/tan/ASR/Codex-web"
RUN_DIR="${PROJECT_ROOT}/results/refiner13_no_C02_20261027"
CONFIG="${PROJECT_ROOT}/experiments/llamafactory/refiner13_qwen25_full_no_C02_20261027.yaml"
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
data=Path(config['dataset_dir']);manifest=json.loads((data/'manifest.json').read_text())
for name,sha in manifest['files_sha256'].items():
    assert hashlib.sha256((data/name).read_bytes()).hexdigest()==sha,name
assert config['finetuning_type']=='full' and config['dataset_dir']==str(data)
assert not any(x['category']=='C02' for x in map(json.loads,(data/'split_audit.jsonl').open()))
out=Path(config['output_dir']);assert not out.exists() or not any(out.iterdir()),'full output already exists'
assert torch.cuda.device_count()==1
free,total=torch.cuda.mem_get_info(0)
if free<18*1024**3:
    raise RuntimeError(f'Full SFT requires 18 GiB free by current host policy; found {free/1024**3:.2f} GiB')
print('GPU0 full-SFT preflight',torch.cuda.get_device_name(0),f'{free/1024**3:.2f}/{total/1024**3:.2f} GiB free',flush=True)
PY
preflight_code=$?
if (( preflight_code != 0 )); then exit "${preflight_code}"; fi
date --iso-8601=seconds > "${RUN_DIR}/full.started"
"${TRAIN_PYTHON}" -u -c 'from llamafactory.cli import main; main()' train "${CONFIG}" > "${RUN_DIR}/full_train.log" 2>&1
training_exit_code=$?
printf '%s %s\n' "${training_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/full.exit"
if (( training_exit_code != 0 )); then exit "${training_exit_code}"; fi
date --iso-8601=seconds > "${RUN_DIR}/full_evaluation.started"
"${TRAIN_PYTHON}" -u "${PROJECT_ROOT}/experiments/scripts/evaluate_refiner_no_c02_full.py" > "${RUN_DIR}/full_evaluation.log" 2>&1
evaluation_exit_code=$?
printf '%s %s\n' "${evaluation_exit_code}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/full_evaluation.exit"
exit "${evaluation_exit_code}"
