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
LEARNED_WINDOW_GATE_MODE="${LEARNED_WINDOW_GATE_MODE:-off}"
LEARNED_WINDOW_GATE_MODEL_DIR="${LEARNED_WINDOW_GATE_MODEL_DIR:-${PROJECT_ROOT}/results/gate_terminal_gain_round4_20261012/linear_training}"
FINAL_REFINEMENT_MODE="${FINAL_REFINEMENT_MODE:-off}"
CONFIDENCE_SUPPLEMENT_MODE="${CONFIDENCE_SUPPLEMENT_MODE:-off}"
CONFIDENCE_SUPPLEMENT_THRESHOLD="${CONFIDENCE_SUPPLEMENT_THRESHOLD:-0.8}"
WINDOW_EDIT_POLICY="${WINDOW_EDIT_POLICY:-compat}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/results/web/session.jsonl}"
ZH_ITN_ENABLED="${ZH_ITN_ENABLED:-0}"
ENTITY_ASR_CONTEXT_ENABLED="${ENTITY_ASR_CONTEXT_ENABLED:-0}"
ENTITY_AUDIO_REPAIR_ENABLED="${ENTITY_AUDIO_REPAIR_ENABLED:-1}"
ENTITY_CTC_MODE="${ENTITY_CTC_MODE:-off}"
ENTITY_CTC_MODEL_DIR="${ENTITY_CTC_MODEL_DIR:-}"
ZH_ITN_LIBRARY="${ZH_ITN_LIBRARY:-${PROJECT_ROOT}/zh-itn/zh-itn/build/libzh_itn_bridge.so}"

case "${WINDOW_EDIT_POLICY}" in
  compat|strict_local) ;;
  *) echo "WINDOW_EDIT_POLICY must be compat or strict_local" >&2; exit 1 ;;
esac
case "${LEARNED_WINDOW_GATE_MODE}" in
  off|shadow|on) ;;
  *) echo "LEARNED_WINDOW_GATE_MODE must be off, shadow, or on" >&2; exit 1 ;;
esac
case "${CONFIDENCE_SUPPLEMENT_MODE}" in
  off|shadow|on) ;;
  *) echo "CONFIDENCE_SUPPLEMENT_MODE must be off, shadow, or on" >&2; exit 1 ;;
esac
case "${ENTITY_CTC_MODE}" in
  off|shadow|on) ;;
  *) echo "ENTITY_CTC_MODE must be off, shadow, or on" >&2; exit 1 ;;
esac
if [[ "${CONFIDENCE_SUPPLEMENT_MODE}" == "on" && "${LEARNED_WINDOW_GATE_MODE}" == "on" ]]; then
  echo "Confidence supplement on requires LEARNED_WINDOW_GATE_MODE=off or shadow to preserve mandatory review." >&2
  exit 1
fi
case "${ZH_ITN_ENABLED,,}" in
  1|true|yes|on) REQUESTED_ZH_ITN_ENABLED=1 ;;
  0|false|no|off) REQUESTED_ZH_ITN_ENABLED=0 ;;
  *) echo "ZH_ITN_ENABLED must be a boolean value" >&2; exit 1 ;;
esac

if curl --silent --show-error --fail --max-time 3 \
  "http://127.0.0.1:${WEB_PORT}/" >/dev/null 2>&1; then
  CURRENT_WEB_HEALTH="$(curl --silent --show-error --fail --max-time 3 \
    "http://127.0.0.1:${WEB_PORT}/health")"
  python -c '
import json, sys
health = json.load(sys.stdin)
expected = {"refinement_gate_mode": sys.argv[1], "final_refinement_mode": sys.argv[2]}
for key, requested in expected.items():
    if health.get(key) != requested:
        sys.exit(f"Web configuration mismatch: {key}={health.get(key)!r}; requested {requested!r}. Use a separate port or explicitly restart the service.")
if health.get("entity_ctc_mode", "off") != sys.argv[4]:
    sys.exit("Web entity CTC mode mismatch; use a separate port or explicitly restart.")
if sys.argv[4] != "off" and health.get("entity_ctc_model_dir") != str(__import__('pathlib').Path(sys.argv[5]).resolve()):
    sys.exit("Web entity CTC model directory mismatch; use a separate port or explicitly restart.")
if health.get("window_pipeline", {}).get("version") != "fixed_k3_context_v2":
    sys.exit("Web uses an older window pipeline. Use a separate port or explicitly restart the service.")
zh_itn = sys.argv[3] == "1"
pipeline = {"version": "fixed_k3_context_v2", "window_size": 3,
            "one_punctuation_window": True, "fixed_groups": True,
            "stream_reprocess_rejected": False,
            "final_reprocess_rejected": sys.argv[2] == "on",
            "group_numeric_self_corrections": zh_itn,
            "adjacent_context": True, "context_lookahead_chunks": 1,
            "context_protocol": "v1-read-only-adjacent-context"}
if health.get("window_pipeline") != pipeline or health.get("zh_itn_enabled") != zh_itn:
    sys.exit("Web window pipeline or zh-itn configuration mismatch. Use a separate port or explicitly restart the service.")
' "${REFINEMENT_GATE_MODE}" "${FINAL_REFINEMENT_MODE}" "${REQUESTED_ZH_ITN_ENABLED}" "${ENTITY_CTC_MODE}" "${ENTITY_CTC_MODEL_DIR}" <<< "${CURRENT_WEB_HEALTH}"
  CURRENT_WINDOW_EDIT_POLICY="$(python -c \
    'import json,sys; print(json.load(sys.stdin).get("window_edit_policy", "compat"))' <<< "${CURRENT_WEB_HEALTH}")"
  python -c '
import json,sys
config=json.load(sys.stdin).get("confidence_supplement", {})
if config.get("mode", "off") != sys.argv[1]:
    sys.exit("Web confidence supplement mode mismatch; use a separate port or explicitly restart.")
if sys.argv[1] != "off" and config.get("low_fragment_threshold") != float(sys.argv[2]):
    sys.exit("Web confidence supplement threshold mismatch; use a separate port or explicitly restart.")
' "${CONFIDENCE_SUPPLEMENT_MODE}" "${CONFIDENCE_SUPPLEMENT_THRESHOLD}" <<< "${CURRENT_WEB_HEALTH}"
  if [[ "${CURRENT_WINDOW_EDIT_POLICY}" != "${WINDOW_EDIT_POLICY}" ]]; then
    echo "Web on port ${WEB_PORT} uses ${CURRENT_WINDOW_EDIT_POLICY}; requested ${WINDOW_EDIT_POLICY}. Use a separate port or explicitly restart the service." >&2
    exit 1
  fi
  CURRENT_LEARNED_GATE_MODE="$(python -c \
    'import json,sys; print(json.load(sys.stdin).get("learned_window_gate", {}).get("mode", "off"))' <<< "${CURRENT_WEB_HEALTH}")"
  if [[ "${CURRENT_LEARNED_GATE_MODE}" != "${LEARNED_WINDOW_GATE_MODE}" ]]; then
    echo "Web on port ${WEB_PORT} uses learned gate ${CURRENT_LEARNED_GATE_MODE}; requested ${LEARNED_WINDOW_GATE_MODE}. Use a separate port or explicitly restart the service." >&2
    exit 1
  fi
  if [[ "${LEARNED_WINDOW_GATE_MODE}" != "off" ]]; then
    CURRENT_LEARNED_GATE_MODEL_DIR="$(python -c \
      'import json,sys; print(json.load(sys.stdin).get("learned_window_gate", {}).get("model_dir", ""))' <<< "${CURRENT_WEB_HEALTH}")"
    REQUESTED_LEARNED_GATE_MODEL_DIR="$(python -c \
      'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${LEARNED_WINDOW_GATE_MODEL_DIR}")"
    if [[ "${CURRENT_LEARNED_GATE_MODEL_DIR}" != "${REQUESTED_LEARNED_GATE_MODEL_DIR}" ]]; then
      echo "Web on port ${WEB_PORT} uses a different learned gate model directory. Use a separate port or explicitly restart the service." >&2
      exit 1
    fi
  fi
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
ENTITY_CONTEXT_ARGS=()
ENTITY_AUDIO_ARGS=()
ENTITY_CTC_ARGS=()
WEB_PYTHON=("${CONDA_EXE}" run --no-capture-output -n "${AGENTIC_ENV}" python)
if [[ "${ENTITY_CTC_MODE}" != "off" ]]; then
  if [[ ! -f "${ENTITY_CTC_MODEL_DIR}/model.int8.onnx" || ! -f "${ENTITY_CTC_MODEL_DIR}/tokens.txt" ]]; then
    echo "ENTITY_CTC_MODEL_DIR must contain model.int8.onnx and tokens.txt" >&2
    exit 1
  fi
  ENTITY_CTC_PYTHON="${ENTITY_CTC_PYTHON:-${PROJECT_ROOT}/.runtime/entity_ctc/bin/python}"
  if [[ ! -x "${ENTITY_CTC_PYTHON}" ]]; then
    echo "ENTITY_CTC_PYTHON must point to a Python with onnxruntime and kaldi-native-fbank" >&2
    exit 1
  fi
  WEB_PYTHON=("${ENTITY_CTC_PYTHON}")
  if ! "${ENTITY_CTC_PYTHON}" -c 'import onnxruntime, kaldi_native_fbank' >/dev/null 2>&1; then
    echo "ENTITY_CTC_PYTHON lacks onnxruntime or kaldi-native-fbank" >&2
    exit 1
  fi
  ENTITY_CTC_ARGS=(--entity-ctc-mode "${ENTITY_CTC_MODE}" --entity-ctc-model-dir "${ENTITY_CTC_MODEL_DIR}")
fi
case "${ENTITY_ASR_CONTEXT_ENABLED,,}" in
  1|true|yes|on) ENTITY_CONTEXT_ARGS=(--enable-entity-asr-context) ;;
  0|false|no|off) ;;
  *) echo "ENTITY_ASR_CONTEXT_ENABLED must be a boolean value" >&2; exit 1 ;;
esac
case "${ENTITY_AUDIO_REPAIR_ENABLED,,}" in
  1|true|yes|on) ENTITY_AUDIO_ARGS=(--enable-entity-audio-repair) ;;
  0|false|no|off) ;;
  *) echo "ENTITY_AUDIO_REPAIR_ENABLED must be a boolean value" >&2; exit 1 ;;
esac
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

exec env CUDA_VISIBLE_DEVICES="${WEB_GPU}" "${WEB_PYTHON[@]}" -m system.web_app \
  --refiner-model "${REFINER_MODEL}" \
  --refiner-device cuda:0 \
  --asr-url "${ASR_URL}" \
  --asr-api-style "${ASR_API_STYLE}" \
  --language "${LANGUAGE:-Chinese}" \
  --entity-db "${ENTITY_DB}" \
  --entity-fuzzy-mode "${ENTITY_FUZZY_MODE}" \
  --refinement-gate-mode "${REFINEMENT_GATE_MODE}" \
  --learned-window-gate-mode "${LEARNED_WINDOW_GATE_MODE}" \
  --learned-window-gate-model-dir "${LEARNED_WINDOW_GATE_MODEL_DIR}" \
  --final-refinement-mode "${FINAL_REFINEMENT_MODE}" \
  --confidence-supplement-mode "${CONFIDENCE_SUPPLEMENT_MODE}" \
  --confidence-supplement-threshold "${CONFIDENCE_SUPPLEMENT_THRESHOLD}" \
  --window-edit-policy "${WINDOW_EDIT_POLICY}" \
  "${ZH_ITN_ARGS[@]}" \
  "${ENTITY_CONTEXT_ARGS[@]}" \
  "${ENTITY_AUDIO_ARGS[@]}" \
  "${ENTITY_CTC_ARGS[@]}" \
  --output "${OUTPUT_PATH}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --port "${WEB_PORT}"
