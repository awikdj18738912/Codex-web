#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
QWEN_TTS_PYTHON="$PROJECT_ROOT/.runtime/qwen3-tts-env/bin/python"
if [[ ! -x "$QWEN_TTS_PYTHON" ]]; then
  echo "Qwen TTS environment is missing: $QWEN_TTS_PYTHON" >&2
  exit 1
fi
cd "$PROJECT_ROOT"
exec "$QWEN_TTS_PYTHON" -m pipeline.src.speech.qwen_tts_service "$@"
