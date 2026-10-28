#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PY=/home/aim0/data/conda/envs/agentic-asr/bin/python
OUT=/home/aim0/data/datasets/ASR/refiner_balanced_50k_20261026
INPUTS=/home/aim0/data/datasets/ASR/refiner_balanced_50k_inputs_20261026
CHILD_PID=''
trap 'if [[ -n "${CHILD_PID}" ]]; then kill -TERM "${CHILD_PID}" 2>/dev/null || true; fi; exit 130' INT TERM
for RECOVERY in 0 1 2 3 4 5; do
  until "${PY}" -c 'import requests; requests.get("http://127.0.0.1:8014/health",timeout=5).raise_for_status()' >/dev/null 2>&1; do
    echo "$(date -Iseconds) waiting for GPU0-only teacher on 8014"
    sleep 20
  done
  "${PY}" -u experiments/scripts/generate_refiner_50k.py --gpu-ids 0 --workers 16 --output "${OUT}" --inputs "${INPUTS}" &
  CHILD_PID=$!
  echo "$(date -Iseconds) first 50k generation pid=${CHILD_PID} GPUs=0 workers=16 recovery=${RECOVERY}"
  wait "${CHILD_PID}"
  RESULT=$?
  CHILD_PID=''
  STATUS="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","unknown"))' "${OUT}/progress.json" 2>/dev/null || echo unknown)"
  echo "$(date -Iseconds) child exit=${RESULT} status=${STATUS}"
  if [[ "${STATUS}" == balanced_gaps_complete_pending_human_review || "${STATUS}" == stopped_after_draining ]]; then exit 0; fi
  if [[ "${STATUS}" != waiting_teacher || "${RESULT}" != 0 ]]; then
    echo 'Generation requires attention; original records and errors retained.'
    exit 1
  fi
  sleep 20
done
exit 1
