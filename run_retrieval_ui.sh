#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
APP_PATH="${SCRIPT_DIR}/CLIP-pretrained-model/codes/ui_app.py"

echo "Starting retrieval UI at http://127.0.0.1:7860"
"${PYTHON_BIN}" "${APP_PATH}"
