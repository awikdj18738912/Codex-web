#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/aim0/桌面/tan/ASR/Codex-web"
TARGET_UNIT="refiner13-full-gpu1.service"
TARGET_SCRIPT="${PROJECT_ROOT}/scripts/run_refiner13_full_gpu1.sh"
RUN_DIR="${PROJECT_ROOT}/results/refiner13_qwen25_full_20261027"
STOP_LOG="${RUN_DIR}/stop_0900.log"

mkdir -p "${RUN_DIR}"
exec >> "${STOP_LOG}" 2>&1
printf '%s deadline check started\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"

if ! systemctl --user is-active --quiet "${TARGET_UNIT}"; then
  printf 'The GPU1 training/evaluation unit has already completed; nothing to stop.\n'
  exit 0
fi

unit_command="$(systemctl --user show "${TARGET_UNIT}" -p ExecStart --value)"
if [[ "${unit_command}" != *"${TARGET_SCRIPT}"* ]]; then
  printf 'Refusing to stop unexpected unit command: %s\n' "${unit_command}"
  exit 2
fi

printf 'Stopping %s at the user-requested 09:00 deadline.\n' "${TARGET_UNIT}"
systemctl --user stop "${TARGET_UNIT}"

if systemctl --user is-active --quiet "${TARGET_UNIT}"; then
  printf 'Unit remains active after stop; escalating only this unit.\n'
  systemctl --user kill --kill-who=all --signal=SIGKILL "${TARGET_UNIT}"
fi

unit_state="$(systemctl --user show "${TARGET_UNIT}" -p ActiveState --value)"
printf 'Final unit state: %s\n' "${unit_state}"
printf 'GPU1 compute processes after stop (may include unrelated jobs):\n'
nvidia-smi -i 1 --query-compute-apps=pid,used_gpu_memory --format=csv,noheader || true

if [[ "${unit_state}" != "inactive" && "${unit_state}" != "failed" ]]; then
  exit 3
fi

"${PROJECT_ROOT}/.runtime/llamafactory-env/bin/python" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path('/home/aim0/桌面/tan/ASR/Codex-web')
run_path = root / 'results/refiner13_qwen25_full_20261027/run.json'
if run_path.exists():
    run = json.loads(run_path.read_text(encoding='utf-8'))
    run['deadline_check_0900'] = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'user_requested_stop_if_still_running': True,
        'unit_stopped': True,
    }
    if run.get('status') not in {'complete', 'evaluation_complete'}:
        run['status'] = 'stopped_at_user_deadline'
    run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
PY

printf '%s deadline check finished\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)"
