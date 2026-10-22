#!/usr/bin/env bash
set -euo pipefail
if (( $# )); then
  echo "Use the prepared YAML configuration; command-line overrides require a new preparation run." >&2
  exit 1
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PYTHON="${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python"
CONFIG="${PROJECT_ROOT}/experiments/llamafactory/qwen25_05b_refiner_reviewed_v5.yaml"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "${PROJECT_ROOT}"
"${TRAIN_PYTHON}" - "${CONFIG}" <<'PY'
import json, sys
from pathlib import Path
import torch, yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
report_path = Path('results/qwen25_05b_refiner_preparation_20261021/preparation_report.json')
report = json.loads(report_path.read_text())
if report.get('status') != 'prepared':
    raise SystemExit('Training preparation is incomplete.')
import hashlib
if report['config_sha256'] != hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest():
    raise SystemExit('Config changed since preparation; rerun prepare_qwen25_refiner_training.py.')
data = Path(config['dataset_dir'])
export = json.loads((data / 'export_report.json').read_text())
if report['source_export_report_sha256'] != hashlib.sha256((data / 'export_report.json').read_bytes()).hexdigest():
    raise SystemExit('Export manifest changed since preparation; rerun preparation.')
for name, expected in export['files'].items():
    h = hashlib.sha256()
    with (data / name).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != expected['sha256']:
        raise SystemExit('Prepared dataset changed: ' + name)
if not Path(config['tokenized_path']).is_dir():
    raise SystemExit('Prepared token cache is missing; rerun preparation.')
for name, expected in report['tokenized_files'].items():
    h = hashlib.sha256()
    with (Path(config['tokenized_path']) / name).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise SystemExit('Prepared token cache changed: ' + name)
output = Path(config['output_dir'])
if output.exists() and any(output.iterdir()):
    raise SystemExit(f'Training output already exists: {output}. Choose a fresh output or explicitly prepare a resume run.')
if torch.cuda.device_count() != 1:
    raise SystemExit('Select exactly one GPU with CUDA_VISIBLE_DEVICES; default is GPU0.')
free, total = torch.cuda.mem_get_info(0)
if free < 20 * 1024**3:
    raise SystemExit(f'Only {free/1024**3:.1f} GiB free. Release the selected GPU before full SFT (requires at least 20 GiB free).')
print(f'GPU: {torch.cuda.get_device_name(0)}; {free/1024**3:.1f}/{total/1024**3:.1f} GiB free', flush=True)
PY
mkdir -p logs/qwen25_05b_refiner_reviewed_v5
LOG="${PROJECT_ROOT}/logs/qwen25_05b_refiner_reviewed_v5/train_$(date +%Y%m%d_%H%M%S).log"
"${TRAIN_PYTHON}" -u -c 'from llamafactory.cli import main; main()' train "${CONFIG}" 2>&1 | tee "${LOG}"
