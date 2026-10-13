#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PYTHON="${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python"
if [[ ! -x "${TRAIN_PYTHON}" ]]; then
  echo "Run bash scripts/setup_llamafactory_refiner.sh first." >&2
  exit 1
fi
cd "${PROJECT_ROOT}"
exec "${TRAIN_PYTHON}" experiments/scripts/run_refiner_llamafactory.py "$@"
