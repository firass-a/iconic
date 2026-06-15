#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
LOG_FILE="${LOG_FILE:-}"

_log() {
  echo "$1"
  if [ -n "$LOG_FILE" ]; then
    echo "$1" >> "$LOG_FILE"
  fi
}

_log "[setup] checking torch + gymnasium..."
if python3 -c "import torch; import gymnasium" 2>/dev/null; then
  _log "[setup] OK, skipping pip"
else
  _log "[setup] installing CPU torch + gymnasium (CPU wheel ~200MB)..."
  pip install -q --default-timeout=300 --no-warn-script-location \
    --index-url https://download.pytorch.org/whl/cpu torch
  pip install -q --default-timeout=300 --no-warn-script-location gymnasium
fi

_log "[setup] starting 05_pc_ppo_custom_train.py"
exec python3 -u 05_pc_ppo_custom_train.py
