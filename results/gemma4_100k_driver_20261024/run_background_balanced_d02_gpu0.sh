#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
OUT=/home/aim0/data/datasets/ASR/gemma4_selfgen_all17_100k_v2_20261024
PY=/home/aim0/data/conda/envs/agentic-asr/bin/python
CHILD_PID=''
trap 'if [[ -n "${CHILD_PID}" ]]; then kill -TERM "${CHILD_PID}" 2>/dev/null || true; fi; exit 130' INT TERM
for RECOVERY in 0 1 2 3 4 5; do
  until "${PY}" -c 'import requests; requests.get("http://127.0.0.1:8014/health",timeout=3).raise_for_status()' >/dev/null 2>&1; do
    echo "$(date -Iseconds) waiting for GPU0-only teacher on 8014"
    sleep 30
  done
  "${PY}" -u experiments/scripts/generate_refiner_multigpu.py \
    --output "${OUT}" --workers 2 --batch-size 4 --per-server-limit 2 \
    --request-timeout 900 --api http://127.0.0.1:8014/v1 \
    --categories D02 --stop-after 1681 &
  CHILD_PID=$!
  echo "$(date -Iseconds) generation pid=${CHILD_PID}, recovery=${RECOVERY}, TP=1, GPU=0, workers=2"
  wait "${CHILD_PID}"
  RESULT=$?
  CHILD_PID=''
  STATUS="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","unknown"))' "${OUT}/progress.json" 2>/dev/null || echo unknown)"
  echo "$(date -Iseconds) child exit=${RESULT} status=${STATUS}"
  if [[ "${STATUS}" == complete_pending_human_review || "${STATUS}" == coverage_gaps_stage_finished || "${STATUS}" == stopped_after_draining ]]; then exit 0; fi
  if [[ "${STATUS}" != waiting_teacher || "${RESULT}" != 0 ]]; then
    echo 'Generation requires attention; original data and errors retained.'
    exit 1
  fi
  sleep 30
done
echo 'Repeated audit transport failures; supervisor requires attention.'
exit 1
