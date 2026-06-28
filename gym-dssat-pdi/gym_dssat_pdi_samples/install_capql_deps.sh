#!/bin/bash
export PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:$PATH"
DEPS=/workspace/pydeps
mkdir -p "$DEPS"
python3 -m pip install --target "$DEPS" torch gymnasium matplotlib numpy
echo "PYTHONPATH=$DEPS"
