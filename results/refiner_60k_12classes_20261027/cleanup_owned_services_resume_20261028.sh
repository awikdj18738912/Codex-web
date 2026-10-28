#!/usr/bin/env bash
# systemd invokes this after all workflow children are gone, even on failure.
set -u
RUN='/home/aim0/桌面/tan/ASR/Codex-web/results/refiner_60k_12classes_20261027/workflow_resume_20261028'
if [[ -f "${RUN}/controlled_handover.flag" ]]; then
    echo 'Controlled supervisor handover; GPU services retained' >> "${RUN}/cleanup_stop_post.log"
    exit 0
fi
for unit in refiner-generation-60k-12classes-v4-20261028.service refiner-generation-60k-12classes-v3-20261027.service refiner-generation-60k-12classes-v2-20261027.service refiner-generation-60k-12classes-20261027.service refiner-teacher-gpu0-60k-20261027.service; do
    if /usr/bin/systemctl --user is-active --quiet "${unit}"; then
        /usr/bin/systemctl --user stop "${unit}" >> "${RUN}/cleanup_stop_post.log" 2>&1
    fi
done
/usr/bin/nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "${RUN}/cleanup_stop_post.log" 2>&1
