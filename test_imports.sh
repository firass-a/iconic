#!/bin/bash
export PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:$PATH"
export PYTHONPATH="/workspace/pydeps:$PYTHONPATH"
/opt/gym_dssat_pdi/bin/python3 -c "import torch; import gymnasium; print('ok', torch.__version__, gymnasium.__version__)"
