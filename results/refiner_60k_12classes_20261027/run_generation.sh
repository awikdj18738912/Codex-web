#!/usr/bin/env bash
set -euo pipefail
cd /home/aim0/桌面/tan/ASR/Codex-web
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
for ((attempt=0; attempt<90; attempt++)); do
    if /usr/bin/python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8014/health", timeout=5)' 2>/dev/null; then
        exec /home/aim0/data/conda/envs/agentic-asr/bin/python -u experiments/scripts/generate_refiner_60k_12classes.py \
          --output /home/aim0/data/datasets/ASR/refiner_balanced_12classes_60k_20261027 \
          --inputs /home/aim0/data/datasets/ASR/refiner_60k_12classes_inputs_20261027
    fi
    if ! systemctl --user is-active --quiet refiner-teacher-gpu0-60k-20261027.service; then
        echo 'Teacher service stopped before readiness' >&2
        exit 1
    fi
    sleep 10
done
echo 'Teacher readiness timeout' >&2
exit 1
