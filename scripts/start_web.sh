#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_EXE="${CONDA_EXE:-/home/aim0/anaconda3/bin/conda}"
AGENTIC_ENV="${AGENTIC_ENV:-agentic-asr}"
REFINER_MODEL="${REFINER_MODEL:-/home/aim0/data/models/ASR/AgenticASR-Refiner}"
ASR_URL="${ASR_URL:-http://127.0.0.1:8766}"
ASR_API_STYLE="${ASR_API_STYLE:-current}"
ENTITY_DB="${ENTITY_DB:-${PROJECT_ROOT}/data/entities.db}"
WEB_GPU="${WEB_GPU:-1}"
WEB_PORT="${WEB_PORT:-8081}"
ENTITY_FUZZY_MODE="${ENTITY_FUZZY_MODE:-auto}"
REFINEMENT_GATE_MODE="${REFINEMENT_GATE_MODE:-tri_state}"
FINAL_REFINEMENT_MODE="${FINAL_REFINEMENT_MODE:-off}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/results/web/session.jsonl}"
ZH_ITN_ENABLED="${ZH_ITN_ENABLED:-0}"
ZH_ITN_LIBRARY="${ZH_ITN_LIBRARY:-${PROJECT_ROOT}/zh-itn/zh-itn/build/libzh_itn_bridge.so}"

if curl --silent --show-error --fail --max-time 3 \
  "http://127.0.0.1:${WEB_PORT}/" >/dev/null 2>&1; then
  echo "Refiner/Web is already healthy on port ${WEB_PORT}; reusing it."
  exit 0
fi
if ss -ltn "sport = :${WEB_PORT}" 2>/dev/null | tail -n +2 | grep -q .; then
  echo "Web port ${WEB_PORT} is occupied but health check failed; refusing to restart it." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ ! -d "${REFINER_MODEL}" ]]; then
  echo "Refiner model not found: ${REFINER_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${ENTITY_DB}" ]]; then
  echo "Entity database not found: ${ENTITY_DB}" >&2
  exit 1
fi
ZH_ITN_ARGS=()
case "${ZH_ITN_ENABLED,,}" in
  1|true|yes|on)
    if [[ ! -f "${ZH_ITN_LIBRARY}" ]]; then
      if [[ "${ZH_ITN_LIBRARY}" == "${PROJECT_ROOT}/zh-itn/zh-itn/build/libzh_itn_bridge.so" ]]; then
        bash "${SCRIPT_DIR}/build_zh_itn_bridge.sh"
      else
        echo "zh-itn library not found: ${ZH_ITN_LIBRARY}" >&2
        exit 1
      fi
    fi
    ZH_ITN_ARGS=(--enable-zh-itn --zh-itn-library "${ZH_ITN_LIBRARY}")
    ;;
  0|false|no|off)
    ;;
  *)
    echo "ZH_ITN_ENABLED must be 0/1, false/true, no/yes, or off/on" >&2
    exit 1
    ;;
esac

exec env CUDA_VISIBLE_DEVICES="${WEB_GPU}" "${CONDA_EXE}" run --no-capture-output -n "${AGENTIC_ENV}" \
  python -m system.web_app \
  --refiner-model "${REFINER_MODEL}" \
  --refiner-device cuda:0 \
  --asr-url "${ASR_URL}" \
  --asr-api-style "${ASR_API_STYLE}" \
  --language "${LANGUAGE:-Chinese}" \
  --entity-db "${ENTITY_DB}" \
  --entity-fuzzy-mode "${ENTITY_FUZZY_MODE}" \
  --refinement-gate-mode "${REFINEMENT_GATE_MODE}" \
  --final-refinement-mode "${FINAL_REFINEMENT_MODE}" \
  "${ZH_ITN_ARGS[@]}" \
  --output "${OUTPUT_PATH}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --port "${WEB_PORT}"
