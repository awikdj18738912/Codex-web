#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FACTORY_REVISION="ce9dc9e072f80fa3abe0989d4ab90da25f083438"
BASE_PYTHON="${REFINER_TRAIN_BASE_PYTHON:-/home/aim0/data/conda/envs/agentic-asr/bin/python}"
TRAIN_ENV="${PROJECT_ROOT}/.runtime/llamafactory-env"
cd "${PROJECT_ROOT}"
if [[ ! -d LLaMA-Factory/.git ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/hiyouga/LlamaFactory.git LLaMA-Factory
  git -C LLaMA-Factory fetch --depth 1 origin "${FACTORY_REVISION}"
  git -C LLaMA-Factory checkout --detach "${FACTORY_REVISION}"
fi
if [[ "$(git -C LLaMA-Factory rev-parse HEAD)" != "${FACTORY_REVISION}" ]]; then
  echo "LLaMA-Factory revision differs from the tested training setup." >&2
  exit 1
fi
"${BASE_PYTHON}" - <<'PY'
from importlib.metadata import version
expected = {'torch':'2.11.0', 'torchvision':'0.26.0', 'torchaudio':'2.11.0'}
for name, wanted in expected.items():
    if version(name).split('+')[0] != wanted:
        raise SystemExit(f'{name} differs from the tested shared runtime; refusing an implicit GPU-stack replacement')
PY
if [[ ! -x "${TRAIN_ENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${TRAIN_ENV}"
fi
"${TRAIN_ENV}/bin/python" -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  -e ./LLaMA-Factory -c experiments/llamafactory/constraints.txt
"${TRAIN_ENV}/bin/python" experiments/scripts/isolate_llamafactory_env.py
"${TRAIN_ENV}/bin/python" -m pip check
"${TRAIN_ENV}/bin/python" experiments/scripts/export_refiner_llamafactory_data.py
echo "Training environment ready: ${TRAIN_ENV}"
