"""Quick smoke test: UFGA.CLI wired and rain > 0 after a few DSSAT days."""
import sys

import numpy as np
from importlib import import_module

from weather_config import resolve_ufga_cli

CAPQLEnv = import_module('capql_env_v2').CAPQLEnv

cli = resolve_ufga_cli()
print('UFGA.CLI:', cli, flush=True)
if not cli:
    print('FAIL: UFGA.CLI not found', flush=True)
    sys.exit(1)

env = CAPQLEnv(mode='all', dssat_seed=3001, run_dssat_location='run_dssat')
print('Env weather CLI:', getattr(env.sos_env, '_weather_cli', None), flush=True)
obs, _ = env.reset()
print('Reset OK', flush=True)

total_rain = 0.0
max_rain = 0.0
info = {}
days = 10
for i in range(days):
    obs, _, done, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    rain = float(env.sos_env._last_crop_obs.get('rain', 0.0) or 0.0)
    total_rain += rain
    max_rain = max(max_rain, rain)
    if i == 0:
        print(f'  day 1 rain={rain:.2f} mm', flush=True)
    if done:
        break

sos = info.get('sos_state', {})
print(f'After {days} days: total_rain={total_rain:.2f} max_daily={max_rain:.2f}', flush=True)
print(f'sos total_rain={float(sos.get("total_rain", 0.0)):.2f}', flush=True)
env.close()
if total_rain <= 0:
    print('WARN: no rain recorded — check UFGA.CLI mount', flush=True)
    sys.exit(2)
print('OK', flush=True)
