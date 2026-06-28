#!/bin/bash
set -e
export PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:$PATH"
export UFGA_CLI_PATH="${UFGA_CLI_PATH:-/workspace/weather/UFGA.CLI}"
cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
/opt/gym_dssat_pdi/bin/python3 _smoke_rain.py
