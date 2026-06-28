#!/bin/bash
# Smart Farm DSS API — run inside gym-dssat Docker image
export PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:$PATH"
export GYM_SAMPLES_DIR="/workspace/gym-dssat-pdi/gym_dssat_pdi_samples"
export CAPQL_MODEL_DIR="/workspace/models/capql_v2"
export CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"

PY=/opt/gym_dssat_pdi/bin/python3
cd /workspace/interface/api
$PY -m pip install -q fastapi uvicorn pydantic
exec $PY -m uvicorn main:app --host 0.0.0.0 --port 8000
