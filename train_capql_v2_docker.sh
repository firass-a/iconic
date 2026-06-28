#!/bin/bash
# Run inside Docker (called by train_capql_v2_docker.ps1).
set -euo pipefail

export PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:${PATH}"
export CAPQL_TRAIN_MODEL_DIR="${CAPQL_TRAIN_MODEL_DIR:-/workspace/models/capql_v2_l7afla}"

PY=/opt/gym_dssat_pdi/bin/python3
if [ ! -x "$PY" ]; then
  PY=/opt/gym_dssat_pdi/bin/python3.11
fi

DEPS=/workspace/pydeps
if ! "$PY" -c "import torch" 2>/dev/null; then
  if [ -d "$DEPS/torch" ]; then
    export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"
  else
    echo "torch not in image — installing to $DEPS (one-time)..."
    mkdir -p "$DEPS"
    "$PY" -m pip install --target "$DEPS" torch gymnasium matplotlib numpy
    export PYTHONPATH="$DEPS"
  fi
fi

mkdir -p "$CAPQL_TRAIN_MODEL_DIR"
cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples

"$PY" -c "import torch, gymnasium, matplotlib; print('torch', torch.__version__)"
"$PY" -u 07_capql_train_v2.py 2>&1 | tee /workspace/train_capql_v2.log
