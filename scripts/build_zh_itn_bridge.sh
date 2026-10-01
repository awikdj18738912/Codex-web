#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ITN_ROOT="${PROJECT_ROOT}/zh-itn/zh-itn"
BUILD_DIR="${ITN_ROOT}/build"

mkdir -p "${BUILD_DIR}"
"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared \
  -I"${ITN_ROOT}/include" \
  "${ITN_ROOT}/src/itn_safe.cc" \
  "${ITN_ROOT}/src/itn_zh_conservative_v2.cc" \
  "${ITN_ROOT}/src/zh_itn_bridge.cc" \
  -o "${BUILD_DIR}/libzh_itn_bridge.so"
printf '%s\n' "${BUILD_DIR}/libzh_itn_bridge.so"
