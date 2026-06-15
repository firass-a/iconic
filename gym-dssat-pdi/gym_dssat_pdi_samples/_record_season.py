"""
Record per-day actions for plotting.

Always records ONE 'uniform' season (-> /figures/daily_actions.csv).
If MODE=all (default), also records the three corners under the SAME weather
seed so they can be overlaid (-> /figures/daily_actions_{yield,water,fert}.csv).

Run inside Docker (model + DSSAT needed).
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import torch  # noqa: F401

from importlib import import_module
pc_env_mod = import_module('pc_env')
PCSmartFarmEnv = pc_env_mod.PCSmartFarmEnv
OBS_DIM = pc_env_mod.OBS_DIM
ACTION_LOW = pc_env_mod.ACTION_LOW
ACTION_HIGH = pc_env_mod.ACTION_HIGH

from ppo_agent import PPOAgent


MODEL_PATH = os.environ.get('MODEL_PATH', '/models/pc_ppo_custom_best.pt')
OUT_DIR    = Path(os.environ.get('OUT_DIR', '/figures'))
MODE       = os.environ.get('MODE', 'all').lower()       # 'all' or 'uniform'
DSSAT_SEED = int(os.environ.get('PROFILE_SEED', '42'))

PROFILES = [
    ('uniform', np.array([1/3, 1/3, 1/3], dtype=np.float32), OUT_DIR / 'daily_actions.csv'),
]
if MODE == 'all':
    PROFILES += [
        ('yield', np.array([1.0, 0.0, 0.0], dtype=np.float32), OUT_DIR / 'daily_actions_yield.csv'),
        ('water', np.array([0.0, 1.0, 0.0], dtype=np.float32), OUT_DIR / 'daily_actions_water.csv'),
        ('fert',  np.array([0.0, 0.0, 1.0], dtype=np.float32), OUT_DIR / 'daily_actions_fert.csv'),
    ]

print(f'  model : {MODEL_PATH}')
print(f'  seed  : {DSSAT_SEED}  (same weather year for all profiles)')
print(f'  mode  : {MODE}   ({len(PROFILES)} profile(s))')

if not os.path.isfile(MODEL_PATH):
    sys.exit(f'ERROR: model not found at {MODEL_PATH}')

agent = PPOAgent(
    obs_dim=OBS_DIM, act_dim=2,
    action_low=ACTION_LOW, action_high=ACTION_HIGH,
    device='cpu',
)
agent.load(MODEL_PATH)

OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, pref, out_path in PROFILES:
    print(f'  [{name}] pref={pref.tolist()} ...', end=' ', flush=True)
    env = PCSmartFarmEnv(
        mode='all', dssat_seed=DSSAT_SEED,
        preference=pref.copy(), rng_seed=DSSAT_SEED,
        random_weather=True,
    )
    obs, _ = env.reset(seed=DSSAT_SEED)
    rows = []
    day = 0
    done = False
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        fs = info.get('full_state', {}) or {}
        rows.append({
            'day': day,
            'action_N': float(action[0]),
            'action_W': float(action[1]),
            'grnwt': float(fs.get('grnwt', 0.0) or 0.0),
            'swfac': float(fs.get('swfac', 0.0) or 0.0),
            'nstres': float(fs.get('nstres', 0.0) or 0.0),
        })
        day += 1
        done = term or trunc
    env.close()
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} days -> {out_path.name}')

print('  done.')
