#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

if python3 -c "import torch; import gymnasium" 2>/dev/null; then
  echo "[setup] torch OK, skipping pip"
else
  echo "[setup] installing CPU torch + gymnasium..."
  pip install -q --default-timeout=300 --no-warn-script-location \
    --index-url https://download.pytorch.org/whl/cpu torch
  pip install -q --default-timeout=300 --no-warn-script-location gymnasium
fi

exec python3 -u 06_pc_ppo_eval.py
