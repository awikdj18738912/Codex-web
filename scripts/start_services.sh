#!/usr/bin/env bash
set -euo pipefail

# Start missing services and reuse healthy ones already listening on the
# configured ports. This is intentionally a supervisor-lite entry point: it
# never restarts a healthy model process just because the command is invoked
# again.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/.runtime/services"
QWEN_PORT="${QWEN_PORT:-8766}"
WEB_PORT="${WEB_PORT:-8081}"
ASR_URL="${ASR_URL:-http://127.0.0.1:${QWEN_PORT}}"
WEB_URL="${WEB_URL:-http://127.0.0.1:${WEB_PORT}}"
ENTITY_DB="${ENTITY_DB:-${PROJECT_ROOT}/../AgenticASR_entity_e5bedb3/data/entities.db}"

mkdir -p "${RUNTIME_DIR}"

is_healthy() {
  curl --silent --show-error --fail --max-time 3 "$1" >/dev/null 2>&1
}

port_in_use() {
  ss -ltn "sport = :$1" 2>/dev/null | tail -n +2 | grep -q .
}

wait_for_health() {
  local url="$1"
  local name="$2"
  local attempts=0
  while (( attempts < 180 )); do
    if is_healthy "$url"; then
      echo "${name} is ready: ${url}"
      return 0
    fi
    sleep 1
    attempts=$((attempts + 1))
  done
  echo "${name} did not become healthy; see ${RUNTIME_DIR}/${name}.log" >&2
  return 1
}

start_qwen_if_missing() {
  if is_healthy "${ASR_URL}/health"; then
    echo "Qwen ASR already running; reuse ${ASR_URL}"
    return 0
  fi
  if port_in_use "${QWEN_PORT}"; then
    echo "Qwen ASR port ${QWEN_PORT} is occupied but health check failed; refusing to replace it" >&2
    return 1
  fi
  echo "Starting Qwen ASR in the background..."
  QWEN_PORT="${QWEN_PORT}" nohup bash "${SCRIPT_DIR}/start_qwen_asr.sh" \
    >"${RUNTIME_DIR}/qwen-asr.log" 2>&1 < /dev/null &
  echo $! >"${RUNTIME_DIR}/qwen-asr.pid"
  wait_for_health "${ASR_URL}/health" "qwen-asr"
}

start_web_if_missing() {
  if is_healthy "${WEB_URL}/"; then
    echo "Refiner/Web already running; reuse ${WEB_URL}"
    return 0
  fi
  if port_in_use "${WEB_PORT}"; then
    echo "Web port ${WEB_PORT} is occupied but health check failed; refusing to replace it" >&2
    return 1
  fi
  echo "Starting Refiner/Web in the background..."
  WEB_PORT="${WEB_PORT}" ENTITY_DB="${ENTITY_DB}" nohup bash "${SCRIPT_DIR}/start_web.sh" \
    >"${RUNTIME_DIR}/refiner-web.log" 2>&1 < /dev/null &
  echo $! >"${RUNTIME_DIR}/refiner-web.pid"
  wait_for_health "${WEB_URL}/" "refiner-web"
}

cd "${PROJECT_ROOT}"
start_qwen_if_missing
start_web_if_missing
echo "Both services are running and will be reused on subsequent invocations."
