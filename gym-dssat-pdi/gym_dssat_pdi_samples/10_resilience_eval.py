"""
10_resilience_eval.py — Phase 1 resilience stress-test for CAPQL v2.

Loads the trained CAPQL v2 actor and runs it under 3 fault scenarios
with increasing severity. No retraining — pure evaluation.

Fault scenarios
---------------
1. sensor_rate  : sensor dropout probability per day (existing SoS mechanism)
                  levels: 0.0  0.02  0.05  0.10  0.20  0.50
2. obs_noise    : Gaussian noise σ added to all 11 crop+SoS obs features
                  levels: 0.0  0.05  0.10  0.20  0.30  0.50
3. blackout_day : all crop+SoS obs zeroed from day X to harvest
                  levels: None  120  90  60  30  10

For each scenario × 4 preference corners × N_EVAL episodes.

Outputs (in /workspace/)
-----------
  resilience_log.csv          — raw results per episode
  resilience_curves.png       — 3×3 grid: scenario × metric
  resilience_heatmap.png      — summary heatmap: corner × fault level

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 10_resilience_eval.py
"""
import os
import csv
import sys
import time
import numpy as np
from importlib import import_module

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn

CAPQLEnv = import_module('capql_env_v2').CAPQLEnv

# ====================================================================== #
# Config                                                                   #
# ====================================================================== #
MODEL_DIR  = '/tmp/capql_v2'
OUT_DIR    = '/workspace/'
LOG_PATH   = '/workspace/resilience_log.csv'
PLOT_CURVE = '/workspace/resilience_curves.png'
PLOT_HEAT  = '/workspace/resilience_heatmap.png'

N_EVAL     = 3          # episodes per (corner × fault level)
BASE_SEEDS = [3000, 3001, 3002]

OBS_DIM    = 14
ACTION_DIM = 2
HIDDEN     = 256
DEVICE     = torch.device('cpu')

ACT_LOW  = np.array([0.0,   0.0],  dtype=np.float32)
ACT_HIGH = np.array([200.0, 50.0], dtype=np.float32)
_ACT_MID  = torch.FloatTensor([100.0, 25.0])
_ACT_HALF = torch.FloatTensor([100.0, 25.0])

EVAL_CORNERS = {
    'yield':    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'n_eff':    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'water':    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'balanced': np.array([1/3, 1/3, 1/3], dtype=np.float32),
}
CORNER_ORDER = ['yield', 'n_eff', 'water', 'balanced']
COLOR = {'yield': '#27ae60', 'n_eff': '#e67e22',
         'water': '#2980b9', 'balanced': '#8e44ad'}
LABEL = {'yield': 'Yield', 'n_eff': 'N-Efficiency',
         'water': 'Water', 'balanced': 'Balanced'}
MARKER = {'yield': 'o', 'n_eff': 's', 'water': '^', 'balanced': 'D'}

FAULT_SCENARIOS = {
    'sensor_rate': {
        'label':   'Sensor Dropout Rate',
        'levels':  [0.0, 0.02, 0.05, 0.10, 0.20, 0.50],
        'xlabels': ['0 %', '2 %', '5 %', '10 %', '20 %', '50 %'],
    },
    'obs_noise': {
        'label':   'Observation Noise  (σ)',
        'levels':  [0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
        'xlabels': ['0', '0.05', '0.10', '0.20', '0.30', '0.50'],
    },
    'blackout_day': {
        'label':   'Sensor Blackout Start (day)',
        'levels':  [None, 120, 90, 60, 30, 10],
        'xlabels': ['No fault', 'Day 120', 'Day 90', 'Day 60', 'Day 30', 'Day 10'],
    },
}
METRICS = ['R_yield', 'R_ane', 'R_water_eff']
METRIC_LABEL = {'R_yield': 'R_yield', 'R_ane': 'R_ane', 'R_water_eff': 'R_water'}


# ====================================================================== #
# Actor  (identical architecture to training)                             #
# ====================================================================== #
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  ACTION_DIM), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs) * _ACT_HALF + _ACT_MID


def load_actor(model_dir):
    actor = Actor().to(DEVICE)
    path  = os.path.join(model_dir, 'actor.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Actor not found: {path}')
    actor.load_state_dict(torch.load(path, map_location=DEVICE))
    actor.eval()
    print(f'✓  Actor loaded from {path}')
    return actor


@torch.no_grad()
def select_action(actor, obs):
    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    action = actor(obs_t).squeeze(0).cpu().numpy()
    return np.clip(action, ACT_LOW, ACT_HIGH).astype(np.float32)


# ====================================================================== #
# Faulty environment wrapper                                               #
# ====================================================================== #
class FaultyEnv:
    """
    Wraps CAPQLEnv with two additional fault types:

    obs_noise_std  : add Gaussian noise N(0, σ) to obs[:11] every step
    blackout_day   : zero obs[:11] from this day onwards (total sensor loss)

    The base sensor_rate fault is passed directly to CAPQLEnv at construction.
    """

    def __init__(self, base_env, obs_noise_std=0.0, blackout_day=None):
        self.env          = base_env
        self.noise_std    = obs_noise_std
        self.blackout_day = blackout_day
        self._day         = 0

        self.action_space      = base_env.action_space
        self.observation_space = base_env.observation_space

    def reset(self, **kwargs):
        obs, info  = self.env.reset(**kwargs)
        self._day  = 0
        return self._corrupt(obs), info

    def step(self, action):
        obs, r, done, trunc, info = self.env.step(action)
        self._day += 1
        return self._corrupt(obs), r, done, trunc, info

    def close(self):
        self.env.close()

    def _corrupt(self, obs):
        obs = obs.copy()
        # Complete blackout: zero all crop + SoS features (keep w unchanged)
        if self.blackout_day is not None and self._day >= self.blackout_day:
            obs[:11] = 0.0
        # Gaussian observation noise on crop + SoS features
        if self.noise_std > 0.0:
            obs[:11] += np.random.normal(0.0, self.noise_std, size=11).astype(np.float32)
            obs[:11]  = np.clip(obs[:11], -2.0, 2.0)
        return obs


def make_env(scenario, level, seed):
    """Build a FaultyEnv for the given fault scenario and severity level."""
    if scenario == 'sensor_rate':
        base = CAPQLEnv(dssat_seed=seed, run_dssat_location='run_dssat',
                        enable_faults=(level > 0.0), fault_rate=float(level or 0.0))
        return FaultyEnv(base)

    if scenario == 'obs_noise':
        base = CAPQLEnv(dssat_seed=seed, run_dssat_location='run_dssat',
                        enable_faults=False)
        return FaultyEnv(base, obs_noise_std=float(level))

    if scenario == 'blackout_day':
        base = CAPQLEnv(dssat_seed=seed, run_dssat_location='run_dssat',
                        enable_faults=True, fault_rate=0.02)
        return FaultyEnv(base, blackout_day=level)   # level may be None

    raise ValueError(f'Unknown scenario: {scenario}')


# ====================================================================== #
# Single episode eval                                                      #
# ====================================================================== #
def run_episode(actor, env, w_corner):
    obs, _ = env.reset()
    # Fix preference to the corner
    env.env.current_w = w_corner.copy()
    obs[-3:] = w_corner

    cum_daily = 0.0
    done      = False
    last_info = {}

    while not done:
        action = select_action(actor, obs)
        obs, r_daily, done, _, last_info = env.step(action)
        obs[-3:] = w_corner     # keep preference fixed during episode
        cum_daily += r_daily

    sos = last_info.get('sos_state', {})
    rv  = last_info.get('reward_vec', np.zeros(3, dtype=np.float32))

    return {
        'R_yield':       float(rv[0]),
        'R_ane':         float(rv[1]),
        'R_water_eff':   float(rv[2]),
        'yield_kg_ha':   float(sos.get('grnwt',         0.0)),
        'total_N_kg_ha': float(sos.get('total_nitrogen', 0.0)),
        'total_W_mm':    float(sos.get('total_water',    0.0)),
        'cum_reward':    cum_daily + float(np.dot(w_corner, rv)),
    }


# ====================================================================== #
# Main evaluation loop                                                     #
# ====================================================================== #
def run_all(actor):
    LOG_FIELDS = ['scenario', 'level_idx', 'level_label',
                  'corner', 'episode',
                  'R_yield', 'R_ane', 'R_water_eff',
                  'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm', 'cum_reward']

    log_rows = []
    # summary[scenario][level_idx][corner] = {metric: mean}
    summary = {s: {li: {} for li in range(len(cfg['levels']))}
               for s, cfg in FAULT_SCENARIOS.items()}

    total_runs = (sum(len(c['levels']) for c in FAULT_SCENARIOS.values())
                  * len(EVAL_CORNERS) * N_EVAL)
    done_runs  = 0
    t0 = time.time()

    print(f'\n  Total episodes to run: {total_runs}')
    print('  scenario          level              corner        '
          'R_yld    R_ane    R_wef    yield')
    print('  ' + '-' * 84)

    for sc_name, sc_cfg in FAULT_SCENARIOS.items():
        for li, level in enumerate(sc_cfg['levels']):
            lbl = sc_cfg['xlabels'][li]

            for c_name, w_corner in EVAL_CORNERS.items():
                ep_rows = []

                for ep_i in range(N_EVAL):
                    seed = BASE_SEEDS[ep_i]
                    env  = make_env(sc_name, level, seed)

                    try:
                        row = run_episode(actor, env, w_corner)
                    finally:
                        env.close()

                    row.update({'scenario':    sc_name,
                                'level_idx':   li,
                                'level_label': lbl,
                                'corner':      c_name,
                                'episode':     ep_i + 1})
                    log_rows.append(row)
                    ep_rows.append(row)
                    done_runs += 1

                # Corner mean
                means = {k: float(np.mean([r[k] for r in ep_rows]))
                         for k in METRICS + ['yield_kg_ha', 'total_N_kg_ha',
                                              'total_W_mm', 'cum_reward']}
                summary[sc_name][li][c_name] = means

                elapsed = time.time() - t0
                eta     = elapsed / done_runs * (total_runs - done_runs)
                print(f'  {sc_name:<17} {lbl:<18} {LABEL[c_name]:<13} '
                      f'{means["R_yield"]:+.3f}  {means["R_ane"]:+.3f}  '
                      f'{means["R_water_eff"]:+.3f}  '
                      f'{means["yield_kg_ha"]:6.0f}   '
                      f'ETA {eta/60:.1f}m')

    # Write CSV
    with open(LOG_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f'\n✓  Log → {LOG_PATH}')

    return summary


# ====================================================================== #
# Plot 1 — Resilience curves  (3 scenarios × 3 metrics)                  #
# ====================================================================== #
def plot_curves(summary, out_path):
    scenarios = list(FAULT_SCENARIOS.keys())
    metrics   = METRICS

    fig, axes = plt.subplots(len(scenarios), len(metrics),
                             figsize=(15, 10), sharey='col')
    fig.suptitle(
        'CAPQL v2 — Resilience Under Fault Injection\n'
        '(solid line = mean over 3 episodes per corner per fault level)',
        fontsize=13, fontweight='bold',
    )

    for row, sc_name in enumerate(scenarios):
        sc_cfg = FAULT_SCENARIOS[sc_name]
        n_lvl  = len(sc_cfg['levels'])
        x      = np.arange(n_lvl)

        for col, metric in enumerate(metrics):
            ax = axes[row][col]

            for c_name in CORNER_ORDER:
                ys = [summary[sc_name][li].get(c_name, {}).get(metric, np.nan)
                      for li in range(n_lvl)]
                ax.plot(x, ys, color=COLOR[c_name], marker=MARKER[c_name],
                        lw=2, ms=7, label=LABEL[c_name])

            # Zero reference
            ax.axhline(0, color='#aaaaaa', lw=0.8, ls='--', alpha=0.6)
            # Baseline (level 0) vertical band
            ax.axvspan(-0.3, 0.3, color='#f0f0f0', alpha=0.6, zorder=0)

            ax.set_xticks(x)
            ax.set_xticklabels(sc_cfg['xlabels'], fontsize=7.5, rotation=20)
            ax.set_ylim(-1.1, 1.15)
            ax.grid(True, alpha=0.25)
            ax.set_axisbelow(True)

            if col == 0:
                ax.set_ylabel(sc_cfg['label'], fontsize=9, fontweight='bold')
            if row == 0:
                ax.set_title(METRIC_LABEL[metric], fontsize=11, fontweight='bold')
            if row == 0 and col == len(metrics) - 1:
                ax.legend(fontsize=8, loc='upper right', framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  {out_path}')


# ====================================================================== #
# Plot 2 — Summary heatmap                                                #
# ====================================================================== #
def plot_heatmap(summary, out_path):
    """
    One heatmap per fault scenario.
    Rows = corners,  Cols = fault levels,  Colour = mean(R_yield+R_ane+R_wef)/3
    """
    scenarios = list(FAULT_SCENARIOS.keys())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(16, 4))
    fig.suptitle(
        'CAPQL v2 — Resilience Heatmap\n'
        'Colour = mean of (R_yield + R_ane + R_water) / 3',
        fontsize=12, fontweight='bold',
    )

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    for ax, sc_name in zip(axes, scenarios):
        sc_cfg = FAULT_SCENARIOS[sc_name]
        n_lvl  = len(sc_cfg['levels'])

        data = np.zeros((len(CORNER_ORDER), n_lvl))
        for ri, c_name in enumerate(CORNER_ORDER):
            for li in range(n_lvl):
                m = summary[sc_name][li].get(c_name, {})
                vals = [m.get(k, 0.0) for k in METRICS]
                data[ri, li] = float(np.mean(vals))

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

        # Annotate cells
        for ri in range(len(CORNER_ORDER)):
            for li in range(n_lvl):
                ax.text(li, ri, f'{data[ri, li]:+.2f}',
                        ha='center', va='center', fontsize=8,
                        color='black' if abs(data[ri, li]) < 0.6 else 'white',
                        fontweight='bold')

        ax.set_xticks(range(n_lvl))
        ax.set_xticklabels(sc_cfg['xlabels'], fontsize=7.5, rotation=25)
        ax.set_yticks(range(len(CORNER_ORDER)))
        ax.set_yticklabels([LABEL[c] for c in CORNER_ORDER], fontsize=9)
        ax.set_title(sc_cfg['label'], fontsize=10, fontweight='bold', pad=8)

        # Highlight baseline column
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, len(CORNER_ORDER),
                                   fill=False, edgecolor='black', lw=2.5))

    plt.colorbar(im, ax=axes[-1], label='Mean reward score',
                 shrink=0.8, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  {out_path}')


# ====================================================================== #
# Main                                                                     #
# ====================================================================== #
def main():
    W = 70
    print('=' * W)
    print('10_resilience_eval.py — CAPQL v2 Resilience Stress Test')
    print('=' * W)
    print(f'  Model   : {MODEL_DIR}')
    print(f'  Corners : {list(EVAL_CORNERS.keys())}')
    print(f'  Episodes: {N_EVAL} per (corner × fault level)')
    print(f'  Faults  : sensor_rate / obs_noise / blackout_day')
    print('=' * W)

    actor = load_actor(MODEL_DIR)

    t0 = time.time()
    summary = run_all(actor)

    plot_curves(summary, PLOT_CURVE)
    plot_heatmap(summary, PLOT_HEAT)

    elapsed = time.time() - t0
    print(f'\n✓  Total time: {elapsed / 60:.1f} min')
    print(f'✓  Outputs in {OUT_DIR}')
    print('=' * W)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n❌  ERROR: {type(e).__name__}: {e}', file=sys.stderr)
        raise
