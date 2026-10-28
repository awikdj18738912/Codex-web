#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PY=/home/aim0/data/conda/envs/agentic-asr/bin/python
OUT=/home/aim0/data/datasets/ASR/refiner_balanced_eight_gaps_20261025
CHILD_PID=''
trap 'if [[ -n "${CHILD_PID}" ]]; then kill -TERM "${CHILD_PID}" 2>/dev/null || true; fi; exit 130' INT TERM
for RECOVERY in 0 1 2 3 4 5; do
  until "${PY}" -c 'import requests; requests.get("http://127.0.0.1:8014/health",timeout=5).raise_for_status()' >/dev/null 2>&1; do
    echo "$(date -Iseconds) waiting for existing GPU0 teacher on 8014"
    sleep 20
  done
  "${PY}" -u experiments/scripts/generate_balanced_refiner_gaps.py \
    --output "${OUT}" \
    --selected /home/aim0/data/datasets/ASR/refiner_balanced_17x5882_strict_labels_20261025/selected.pending_review.jsonl \
    --d02-snapshot /home/aim0/data/datasets/ASR/refiner_balanced_gap_inputs_20261025/completed_D02_machine_passed.jsonl \
    --public-root /home/aim0/data/datasets/ASR/public_refiner_sources_20261025 \
    --fcgec-test /home/aim0/data/datasets/ASR/refiner_balanced_17x5882_20261025/heldout_sources/FCGEC_test.json \
    --workers 16 --per-server-limit 16 --batch-size 4 --target-per-class 5882 \
    --categories A02,A03,A04,C01,D01,E03,E05,E06 --request-timeout 900 &
  CHILD_PID=$!
  echo "$(date -Iseconds) balanced eight-class generation pid=${CHILD_PID} GPU=0 workers=16 recovery=${RECOVERY}"
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
echo 'Repeated teacher transport failures; inspect before more generation.'
exit 1
