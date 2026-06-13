"""
11_resilience_eval_v2.py  —  Realistic resilience stress-test for CAPQL v2.

Uses FaultyEnvV2 (realistic sensor→feature mapping, actuator faults).
Tests the UNMODIFIED CAPQL v2 model (14-dim obs, MLP actor).

Fault scenarios tested
──────────────────────
1. sensor_dropout   : sensor nodes lose power (Type 1, observable)
   - Sensor 0 (soil moisture) dropout rate  0→0.5
   - Sensor 3 (N-meter)       dropout rate  0→0.5
   - All sensors              dropout rate  0→0.5

2. packet_loss      : per-step packet loss probability (Type 1, observable)
   - levels 0→0.5

3. sensor_stuck     : soil moisture sensor (sensor_0) freezes at different days
   - stuck at day  [10, 30, 60, 90, 120]

4. sensor_bias      : soil moisture sensor reads systematically high or low
   - bias ∈ [-0.3, -0.2, -0.1, +0.1, +0.2, +0.3]

5. actuator_N       : N spreader efficiency degradation
   - η_N ∈ [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

6. actuator_W       : irrigation pump efficiency degradation
   - η_W ∈ [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

Outputs (in /workspace/)
────────────────────────
  resilience_v2_log.csv      raw results per episode
  resilience_v2_curves.png   line plots per scenario
  resilience_v2_heatmap.png  summary heatmap

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 11_resilience_eval_v2.py
"""

import os, csv, sys, time
import numpy as np
from importlib import import_module

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn

CAPQLEnv    = import_module('capql_env_v2').CAPQLEnv
FaultyEnvV2 = import_module('faulty_env_v2').FaultyEnvV2

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════
MODEL_DIR   = '/tmp/capql_v2'
OUT_DIR     = '/workspace/'
LOG_PATH    = '/workspace/resilience_v2_log.csv'
PLOT_CURVE  = '/workspace/resilience_v2_curves.png'
PLOT_HEAT   = '/workspace/resilience_v2_heatmap.png'

N_EVAL      = 3          # episodes per (corner × fault level)
BASE_SEEDS  = [4000, 4001, 4002]

OBS_DIM     = 14         # CAPQL v2 expects 14-dim (no mask flags)
ACTION_DIM  = 2
HIDDEN      = 256
DEVICE      = torch.device('cpu')

ACT_LOW     = np.array([0.0,   0.0],  dtype=np.float32)
ACT_HIGH    = np.array([200.0, 50.0], dtype=np.float32)
_ACT_MID    = torch.FloatTensor([100.0, 25.0])
_ACT_HALF   = torch.FloatTensor([100.0, 25.0])

EVAL_CORNERS = {
    'yield':    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'n_eff':    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'water':    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'balanced': np.array([1/3, 1/3, 1/3], dtype=np.float32),
}
CORNER_ORDER = ['yield', 'n_eff', 'water', 'balanced']
COLOR  = {'yield': '#27ae60', 'n_eff': '#e67e22',
          'water': '#2980b9', 'balanced': '#8e44ad'}
LABEL  = {'yield': 'Yield', 'n_eff': 'N-Efficiency',
          'water': 'Water', 'balanced': 'Balanced'}
MARKER = {'yield': 'o', 'n_eff': 's', 'water': '^', 'balanced': 'D'}

# ──────────────────────────────────────────────────────────────────────
# Scenario definitions
# Each entry: (scenario_id, label, fault_kwargs_list, x_labels)
# fault_kwargs are passed directly to FaultyEnvV2 (return_mask=False)
# ──────────────────────────────────────────────────────────────────────
SCENARIOS = [
    (
        'dropout_soil',
        'Soil Moisture Sensor Dropout (sensor_0)',
        [
            {'dropout_rate': 0.00},
            {'dropout_rate': 0.05},
            {'dropout_rate': 0.10},
            {'dropout_rate': 0.20},
            {'dropout_rate': 0.50},
        ],
        ['0%', '5%', '10%', '20%', '50%'],
    ),
    (
        'dropout_all',
        'All Sensors Dropout Rate',
        [
            {'dropout_rate': 0.00},
            {'dropout_rate': 0.05},
            {'dropout_rate': 0.10},
            {'dropout_rate': 0.20},
            {'dropout_rate': 0.50},
        ],
        ['0%', '5%', '10%', '20%', '50%'],
    ),
    (
        'packet_loss',
        'Packet Loss Probability (Type 1)',
        [
            {'packet_loss_prob': 0.00},
            {'packet_loss_prob': 0.05},
            {'packet_loss_prob': 0.10},
            {'packet_loss_prob': 0.20},
            {'packet_loss_prob': 0.50},
        ],
        ['0%', '5%', '10%', '20%', '50%'],
    ),
    (
        'stuck_soil',
        'Soil Moisture Sensor Stuck (Type 2) — freeze day',
        [
            {},                                              # no fault
            {'stuck_sensors': [0], 'stuck_day': 120},
            {'stuck_sensors': [0], 'stuck_day':  90},
            {'stuck_sensors': [0], 'stuck_day':  60},
            {'stuck_sensors': [0], 'stuck_day':  30},
            {'stuck_sensors': [0], 'stuck_day':  10},
        ],
        ['No fault', 'Day 120', 'Day 90', 'Day 60', 'Day 30', 'Day 10'],
    ),
    (
        'bias_soil',
        'Soil Moisture Sensor Bias (Type 2)',
        [
            {'sensor_bias': {0:  0.00}},
            {'sensor_bias': {0: -0.20}},
            {'sensor_bias': {0: -0.10}},
            {'sensor_bias': {0: +0.10}},
            {'sensor_bias': {0: +0.20}},
            {'sensor_bias': {0: +0.30}},
        ],
        ['0', '-0.20', '-0.10', '+0.10', '+0.20', '+0.30'],
    ),
    (
        'actuator_N',
        'N Spreader Efficiency (Actuator)',
        [
            {'N_efficiency': 1.0},
            {'N_efficiency': 0.8},
            {'N_efficiency': 0.6},
            {'N_efficiency': 0.4},
            {'N_efficiency': 0.2},
            {'N_efficiency': 0.0},
        ],
        ['100%', '80%', '60%', '40%', '20%', '0%'],
    ),
    (
        'actuator_W',
        'Irrigation Pump Efficiency (Actuator)',
        [
            {'W_efficiency': 1.0},
            {'W_efficiency': 0.8},
            {'W_efficiency': 0.6},
            {'W_efficiency': 0.4},
            {'W_efficiency': 0.2},
            {'W_efficiency': 0.0},
        ],
        ['100%', '80%', '60%', '40%', '20%', '0%'],
    ),
]

METRICS = ['R_yield', 'R_ane', 'R_water_eff']
METRIC_LABEL = {'R_yield': 'R_yield', 'R_ane': 'R_ane', 'R_water_eff': 'R_water'}


# ══════════════════════════════════════════════════════════════════════
# Actor  (identical to training architecture)
# ══════════════════════════════════════════════════════════════════════
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
    actor.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True)
    )
    actor.eval()
    print(f'✓  Actor loaded from {path}')
    return actor


@torch.no_grad()
def select_action(actor, obs):
    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    action = actor(obs_t).squeeze(0).cpu().numpy()
    return np.clip(action, ACT_LOW, ACT_HIGH).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════
def run_episode(actor, env, w_corner):
    obs, _ = env.reset()
    # Pin preference to the corner
    env.env.current_w = w_corner.copy()
    obs[-3:] = w_corner

    done = False
    last_info = {}
    while not done:
        action = select_action(actor, obs)
        obs, _, done, _, last_info = env.step(action)

    rv  = last_info.get('reward_vec', np.zeros(3))
    sos = last_info.get('sos_state',  {})
    return {
        'R_yield':     float(rv[0]),
        'R_ane':       float(rv[1]),
        'R_water_eff': float(rv[2]),
        'yield_kg_ha': float(sos.get('grnwt', 0.0)),
        'total_N_kg_ha': float(sos.get('total_nitrogen', 0.0)),
        'total_W_mm':  float(sos.get('total_water', 0.0)),
    }


# ══════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════
def make_env(seed, fault_kwargs):
    base = CAPQLEnv(
        dssat_seed=seed,
        run_dssat_location='run_dssat',
        enable_faults=False,    # SoS internal faults disabled — we handle all
    )
    return FaultyEnvV2(base, return_mask=False, **fault_kwargs)


def run_eval(actor):
    records = []
    total_eps = sum(
        len(scenario[2]) * len(EVAL_CORNERS) * N_EVAL
        for scenario in SCENARIOS
    )
    ep_done   = 0
    t0        = time.time()

    header = (
        f"  {'scenario':<22} {'level':<16} {'corner':<14}"
        f"{'R_yld':>8}{'R_ane':>8}{'R_wef':>8}{'yield':>8}"
    )
    sep = '  ' + '─' * 78
    print(header)
    print(sep)

    for scenario_id, scenario_label, fault_levels, xlabels in SCENARIOS:
        for lvl_idx, (fault_kwargs, xlabel) in enumerate(
                zip(fault_levels, xlabels)):
            for corner_name, w_corner in EVAL_CORNERS.items():
                ep_results = []
                for seed in BASE_SEEDS:
                    env = make_env(seed, fault_kwargs)
                    try:
                        res = run_episode(actor, env, w_corner)
                    finally:
                        env.close()
                    ep_results.append(res)
                    ep_done += 1

                mean_r = {k: float(np.mean([r[k] for r in ep_results]))
                          for k in ep_results[0]}

                elapsed = time.time() - t0
                eta     = elapsed / ep_done * (total_eps - ep_done) / 60.0

                print(
                    f"  {scenario_id:<22} {xlabel:<16} {corner_name:<14}"
                    f"{mean_r['R_yield']:+8.3f}"
                    f"{mean_r['R_ane']:+8.3f}"
                    f"{mean_r['R_water_eff']:+8.3f}"
                    f"{mean_r['yield_kg_ha']:8.0f}"
                    f"   ETA {eta:.1f}m"
                )

                records.append({
                    'scenario':    scenario_id,
                    'level_label': xlabel,
                    'level_idx':   lvl_idx,
                    'corner':      corner_name,
                    **mean_r,
                })

    return records


# ══════════════════════════════════════════════════════════════════════
# Save CSV
# ══════════════════════════════════════════════════════════════════════
def save_csv(records, path):
    if not records:
        return
    keys = list(records[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)
    print(f'✓  Log → {path}')


# ══════════════════════════════════════════════════════════════════════
# Plotting — degradation curves
# ══════════════════════════════════════════════════════════════════════
def plot_curves(records, out_path):
    import pandas as pd
    df = pd.DataFrame(records)

    n_scenarios = len(SCENARIOS)
    n_metrics   = len(METRICS)
    fig, axes = plt.subplots(
        n_scenarios, n_metrics,
        figsize=(5 * n_metrics, 3.5 * n_scenarios),
        squeeze=False,
    )

    fig.suptitle(
        'CAPQL v2 — Resilience under Realistic Faults\n'
        '(baseline trained without faults)',
        fontsize=13, fontweight='bold',
    )

    for row, (scenario_id, scenario_label, fault_levels, xlabels) in \
            enumerate(SCENARIOS):
        sub = df[df['scenario'] == scenario_id]
        n_levels = len(fault_levels)

        for col, metric in enumerate(METRICS):
            ax = axes[row][col]

            for corner in CORNER_ORDER:
                csub = sub[sub['corner'] == corner]
                vals = []
                for li in range(n_levels):
                    lsub = csub[csub['level_idx'] == li]
                    vals.append(
                        float(lsub[metric].values[0]) if len(lsub) else np.nan
                    )
                ax.plot(
                    range(n_levels), vals,
                    color=COLOR[corner], marker=MARKER[corner],
                    lw=1.8, ms=6, label=LABEL[corner],
                )

            ax.axhline(0, color='#aaaaaa', lw=0.8, ls='--', alpha=0.5)
            ax.set_xticks(range(n_levels))
            ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha='right')
            ax.set_ylim(-1.1, 1.15)
            ax.set_ylabel(METRIC_LABEL[metric], fontsize=9)
            ax.set_title(f'{scenario_label}\n{METRIC_LABEL[metric]}',
                         fontsize=9, fontweight='bold', pad=4)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, loc='lower left', framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  {out_path}')


# ══════════════════════════════════════════════════════════════════════
# Plotting — summary heatmap  (one per scenario)
# ══════════════════════════════════════════════════════════════════════
def plot_heatmap(records, out_path):
    import pandas as pd
    df = pd.DataFrame(records)

    n_scenarios = len(SCENARIOS)
    fig, axes = plt.subplots(
        1, n_scenarios,
        figsize=(4 * n_scenarios, 5),
        squeeze=False,
    )
    fig.suptitle(
        'CAPQL v2 — Resilience Heatmap\n'
        'Mean R_yield across corners and fault levels',
        fontsize=12, fontweight='bold',
    )

    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    for col, (scenario_id, scenario_label, fault_levels, xlabels) in \
            enumerate(SCENARIOS):
        sub  = df[df['scenario'] == scenario_id]
        grid = np.full((len(CORNER_ORDER), len(fault_levels)), np.nan)

        for ci, corner in enumerate(CORNER_ORDER):
            for li in range(len(fault_levels)):
                lsub = sub[(sub['corner'] == corner) & (sub['level_idx'] == li)]
                if len(lsub):
                    grid[ci, li] = float(lsub['R_yield'].values[0])

        ax = axes[0][col]
        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=7.5, rotation=40, ha='right')
        ax.set_yticks(range(len(CORNER_ORDER)))
        ax.set_yticklabels([LABEL[c] for c in CORNER_ORDER], fontsize=8)
        ax.set_title(scenario_label, fontsize=8, fontweight='bold', pad=6)

        for ci in range(len(CORNER_ORDER)):
            for li in range(len(fault_levels)):
                v = grid[ci, li]
                if not np.isnan(v):
                    ax.text(li, ci, f'{v:+.2f}', ha='center', va='center',
                            fontsize=6.5, color='black')

        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  {out_path}')


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('11_resilience_eval_v2.py  —  CAPQL v2 Realistic Resilience Test')
    print('=' * 70)
    print(f'  Model  : {MODEL_DIR}')
    print(f'  Output : {OUT_DIR}')
    print(f'  Fault scenarios : {len(SCENARIOS)}')
    total = sum(
        len(s[2]) * len(EVAL_CORNERS) * N_EVAL for s in SCENARIOS
    )
    print(f'  Total episodes  : {total}')
    print('=' * 70 + '\n')

    actor   = load_actor(MODEL_DIR)
    t0      = time.time()
    records = run_eval(actor)

    print(f'\n✓  Total time: {(time.time()-t0)/60:.1f} min')
    save_csv(records, LOG_PATH)

    try:
        import pandas
        plot_curves(records, PLOT_CURVE)
        plot_heatmap(records, PLOT_HEAT)
    except ImportError:
        print('  pandas not available — skipping plots')

    print('\n✓  Done.  Results in', OUT_DIR)
    print('=' * 70)


if __name__ == '__main__':
    main()
