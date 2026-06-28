#!/bin/bash
export PATH="/opt/gym_dssat_pdi/bin:/opt/dssat_pdi:$PATH"
/opt/gym_dssat_pdi/bin/python3 -c "import torch; print('torch', torch.__version__)"
/opt/gym_dssat_pdi/bin/python3 -c "import gymnasium" 2>/dev/null && echo gymnasium ok || echo gymnasium missing
/opt/gym_dssat_pdi/bin/python3 -c "import gym; print('gym', gym.__version__)"
