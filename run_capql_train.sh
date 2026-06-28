#!/bin/bash
set -e
export PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:$PATH"
export CAPQL_TRAIN_MODEL_DIR="${CAPQL_TRAIN_MODEL_DIR:-/workspace/models/capql_v2_l7afla}"
PY=/opt/gym_dssat_pdi/bin/python3
if [ ! -x "$PY" ]; then PY=/opt/gym_dssat_pdi/bin/python3.11; fi
cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
mkdir -p "$CAPQL_TRAIN_MODEL_DIR"
"$PY" -c "import torch, gymnasium, matplotlib"
"$PY" -u 07_capql_train_v2.py 2>&1 | tee /workspace/train_capql_v2.log
