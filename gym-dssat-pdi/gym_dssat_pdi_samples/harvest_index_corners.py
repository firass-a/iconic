"""
Harvest index (HI = grnwt / topwt) at preference corners — eval only, no training.

PC-PPO : loads pc_ppo_custom_best.pt
CAPQL  : loads /workspace/models/capql_v2/actor.pt

Usage (Docker):
  docker run --rm -v .../iconic:/workspace -v .../samples:/work -v .../models:/models \\
    -w /work gym-dssat:torch python harvest_index_corners.py
"""
import csv
import os
import sys
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from smart_farm_rewards import harvest_index

EPISODES = int(os.environ.get('HI_EPISODES', '5'))
DSSAT_SEED = int(os.environ.get('DSSAT_SEED', '3000'))

PC_CORNERS = {
    'Yield':      ([1.0, 0.0, 0.0], 'corner_yield'),
    'Water':      ([0.0, 1.0, 0.0], 'corner_water'),
    'Fertilizer': ([0.0, 0.0, 1.0], 'corner_fert'),
    'Balanced':   ([1 / 3, 1 / 3, 1 / 3], 'uniform'),
}

CAPQL_CORNERS = {
    'Yield':      np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'N Efficiency': np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'Water':      np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'Balanced':   np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32),
}

TOPWT_NORM = 20000.0
GRNWT_NORM = 12000.0


def _hi(grnwt, topwt):
    return harvest_index(grnwt, topwt)


# ------------------------------------------------------------------ PC-PPO
def eval_pc_ppo(model_path):
    pc = import_module('pc_env')
    from ppo_agent import PPOAgent

    agent = PPOAgent(
        obs_dim=pc.OBS_DIM,
        act_dim=2,
        action_low=pc.ACTION_LOW,
        action_high=pc.ACTION_HIGH,
        device='cpu',
    )
    agent.load(model_path)

    rows = []
    for label, (w, _) in PC_CORNERS.items():
        w = np.array(w, dtype=np.float32)
        for ep in range(EPISODES):
            env = pc.PCSmartFarmEnv(
                mode='all', dssat_seed=DSSAT_SEED + ep,
                preference=w, rng_seed=1000 + ep, random_weather=True,
            )
            obs, _ = env.reset(seed=1000 + ep)
            last_grnwt = last_topwt = 0.0
            done = False
            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                fs = info.get('full_state', {}) or {}
                g = float(fs.get('grnwt', 0.0) or 0.0)
                t = float(fs.get('topwt', 0.0) or 0.0)
                if g > 0:
                    last_grnwt = g
                if t > 0:
                    last_topwt = t
                done = term or trunc
            fs = info.get('full_state', {}) or {}
            grnwt = max(last_grnwt, float(fs.get('grnwt', 0.0) or 0.0))
            topwt = max(last_topwt, float(fs.get('topwt', 0.0) or 0.0))
            env.close()
            rows.append({
                'model': 'PC-PPO',
                'preference': label,
                'episode': ep + 1,
                'grnwt': grnwt,
                'topwt': topwt,
                'harvest_index': _hi(grnwt, topwt),
            })
    return rows


# ------------------------------------------------------------------ CAPQL v2
class _Actor(nn.Module):
    def __init__(self, obs_dim=14, hidden=256):
        super().__init__()
        act_low = torch.tensor([0.0, 0.0])
        act_high = torch.tensor([200.0, 50.0])
        self._half = (act_high - act_low) / 2.0
        self._mid = (act_high + act_low) / 2.0
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs) * self._half + self._mid


def eval_capql(model_path):
    CAPQLEnv = import_module('capql_env_v2').CAPQLEnv

    actor = _Actor()
    actor.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    actor.eval()

    rows = []
    for label, w_corner in CAPQL_CORNERS.items():
        for ep in range(EPISODES):
            env = CAPQLEnv(mode='all', dssat_seed=DSSAT_SEED + ep,
                           run_dssat_location='run_dssat')
            obs, _ = env.reset()
            env.current_w = w_corner.copy()
            obs[-3:] = w_corner

            last_grnwt = last_topwt = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    action = actor(
                        torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    ).squeeze(0).numpy()
                obs, _, term, trunc, info = env.step(action)
                obs[-3:] = w_corner
                # decode from observation (crop feats are normalized)
                g = float(obs[1]) * GRNWT_NORM
                t = float(obs[0]) * TOPWT_NORM
                if g > 0:
                    last_grnwt = g
                if t > 0:
                    last_topwt = t
                done = term or trunc

            sos = info.get('sos_state', {}) or {}
            grnwt = max(last_grnwt, float(sos.get('grnwt', 0.0) or 0.0))
            topwt = max(last_topwt, 0.0)
            env.close()
            rows.append({
                'model': 'CAPQL v2',
                'preference': label,
                'episode': ep + 1,
                'grnwt': grnwt,
                'topwt': topwt,
                'harvest_index': _hi(grnwt, topwt),
            })
    return rows


def summarize(rows):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r['model'], r['preference'])].append(r)
    out = []
    for (model, pref), items in sorted(groups.items()):
        hi = [x['harvest_index'] for x in items]
        y = [x['grnwt'] for x in items]
        out.append({
            'model': model,
            'preference': pref,
            'n': len(items),
            'hi_mean': float(np.mean(hi)),
            'hi_std': float(np.std(hi)),
            'grnwt_mean': float(np.mean(y)),
        })
    return out


def main():
    pc_model = os.environ.get('PC_PPO_MODEL', '/models/pc_ppo_custom_best.pt')
    capql_model = os.environ.get(
        'CAPQL_MODEL',
        '/workspace/models/capql_v2/actor.pt'
        if os.path.isfile('/workspace/models/capql_v2/actor.pt')
        else os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'models', 'capql_v2', 'actor.pt',
        )),
    )

    all_rows = []
    if os.path.isfile(pc_model):
        print(f'PC-PPO model: {pc_model}')
        all_rows.extend(eval_pc_ppo(pc_model))
    else:
        print(f'SKIP PC-PPO — missing {pc_model}', file=sys.stderr)

    if os.path.isfile(capql_model):
        print(f'CAPQL model:  {capql_model}')
        all_rows.extend(eval_capql(capql_model))
    else:
        print(f'SKIP CAPQL — missing {capql_model}', file=sys.stderr)

    if not all_rows:
        sys.exit(1)

    out_csv = os.environ.get(
        'HI_OUTPUT',
        os.path.join(os.path.dirname(__file__), 'figures', 'harvest_index_corners.csv'),
    )
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    fields = ['model', 'preference', 'episode', 'grnwt', 'topwt', 'harvest_index']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f'Wrote {out_csv}')

    print()
    print(f'Harvest index (HI = grnwt/topwt), {EPISODES} eps/corner, random weather')
    print(f'{"Model":<12} {"Preference":<16} {"n":>3}  {"HI":>8}  {"Yield":>8}')
    print('-' * 55)
    for s in summarize(all_rows):
        print(
            f'{s["model"]:<12} {s["preference"]:<16} {s["n"]:>3}  '
            f'{s["hi_mean"]:.3f}±{s["hi_std"]:.3f}  {s["grnwt_mean"]:,.0f}'
        )


if __name__ == '__main__':
    main()
