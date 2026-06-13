"""
13_resilience_eval_robust.py  —  Resilience stress-test for CAPQL-Robust.

Same 7 fault scenarios as 11_resilience_eval_v2.py but:
  • Loads the CAPQL-Robust MLP actor (19-dim obs: crop+mask+w)
  • Uses FaultyEnvV2(return_mask=True) so mask flags reach the model
  • Applies per-corner action caps (same squared-law as training)
  • Generates individual curves + side-by-side comparison with v2

Outputs (in /workspace/)
─────────────────────────
  resilience_robust_log.csv       raw results
  resilience_robust_curves.png    per-model degradation curves
  resilience_comparison.png       v2 vs robust side-by-side (R_yield)
  resilience_comparison_full.png  all 3 metrics × all 7 scenarios

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 -u 13_resilience_eval_robust.py 2>&1 | tee /workspace/gym-dssat-pdi/resilience_robust.log
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
# Paths
# ══════════════════════════════════════════════════════════════════════
MODEL_DIR       = '/workspace/capql_robust'
V2_LOG          = '/workspace/resilience_v2_log.csv'       # from script 11
LOG_PATH        = '/workspace/resilience_robust_log.csv'
PLOT_CURVES     = '/workspace/resilience_robust_curves.png'
PLOT_CMP_YIELD  = '/workspace/resilience_comparison.png'
PLOT_CMP_FULL   = '/workspace/resilience_comparison_full.png'

N_EVAL     = 3
BASE_SEEDS = [4000, 4001, 4002]

OBS_DIM    = 19          # crop(11) + mask(5) + w(3) — CAPQL-Robust
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
COLOR  = {'yield': '#27ae60', 'n_eff': '#e67e22',
          'water': '#2980b9', 'balanced': '#8e44ad'}
LABEL  = {'yield': 'Yield', 'n_eff': 'N-Eff',
          'water': 'Water', 'balanced': 'Balanced'}
MARKER = {'yield': 'o', 'n_eff': 's', 'water': '^', 'balanced': 'D'}

METRICS       = ['R_yield', 'R_ane', 'R_water_eff']
METRIC_LABEL  = {'R_yield': 'R_yield', 'R_ane': 'R_ane', 'R_water_eff': 'R_water'}

# ══════════════════════════════════════════════════════════════════════
# Scenario definitions  (identical to 11_resilience_eval_v2.py)
# ══════════════════════════════════════════════════════════════════════
SCENARIOS = [
    (
        'dropout_soil',
        'Soil Moisture Dropout (sensor_0)',
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
        'Soil Moisture Stuck (Type 2)',
        [
            {},
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
        'Soil Moisture Bias (Type 2)',
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
        'N Spreader Efficiency',
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
        'Irrigation Pump Efficiency',
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


# ══════════════════════════════════════════════════════════════════════
# Actor  (19-dim input, same MLP structure as training)
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
    print(f'  ✓  Actor loaded from {path}')
    return actor


@torch.no_grad()
def select_action(actor, obs, max_anfer, max_amir):
    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    action = actor(obs_t).squeeze(0).cpu().numpy()
    action[0] = np.clip(action[0], 0.0, max_anfer)
    action[1] = np.clip(action[1], 0.0, max_amir)
    return action.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════
def run_episode(actor, env, w_corner):
    obs, _ = env.reset()
    env.env.current_w = w_corner.copy()   # pin w on inner CAPQLEnv
    obs[-3:] = w_corner

    # Compute per-corner action caps (same squared-law as training)
    # Also update inner CAPQLEnv caps (reset() set them for a random w)
    w_neff    = float(w_corner[1])
    w_water   = float(w_corner[2])
    max_anfer = float(np.clip(200.0 * (1.0 - w_neff) ** 2,  2.0, 200.0))
    max_amir  = float(np.clip( 50.0 * (1.0 - w_water) ** 2, 3.0,  50.0))
    env.env._max_anfer = max_anfer
    env.env._max_amir  = max_amir

    done = False
    last_info = {}
    while not done:
        action = select_action(actor, obs, max_anfer, max_amir)
        obs, _, done, _, last_info = env.step(action)
        obs[-3:] = w_corner   # keep w pinned every step

    rv  = last_info.get('reward_vec', np.zeros(3))
    sos = last_info.get('sos_state',  {})
    return {
        'R_yield':       float(rv[0]),
        'R_ane':         float(rv[1]),
        'R_water_eff':   float(rv[2]),
        'yield_kg_ha':   float(sos.get('grnwt', 0.0)),
        'total_N_kg_ha': float(sos.get('total_nitrogen', 0.0)),
        'total_W_mm':    float(sos.get('total_water', 0.0)),
    }


def make_env(seed, fault_kwargs):
    base = CAPQLEnv(
        dssat_seed=seed,
        run_dssat_location='run_dssat',
        enable_faults=False,
    )
    return FaultyEnvV2(base, return_mask=True, **fault_kwargs)


# ══════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════
def run_eval(actor):
    records   = []
    total_eps = sum(
        len(s[2]) * len(EVAL_CORNERS) * N_EVAL for s in SCENARIOS
    )
    ep_done = 0
    t0      = time.time()

    print(
        f"  {'scenario':<22} {'level':<16} {'corner':<14}"
        f"{'R_yld':>8}{'R_ane':>8}{'R_wef':>8}{'yield':>8}"
    )
    print('  ' + '─' * 78)

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

                mean_r  = {k: float(np.mean([r[k] for r in ep_results]))
                           for k in ep_results[0]}
                elapsed = time.time() - t0
                eta     = elapsed / ep_done * (total_eps - ep_done) / 60.0

                print(
                    f"  {scenario_id:<22} {xlabel:<16} {corner_name:<14}"
                    f"{mean_r['R_yield']:+8.3f}"
                    f"{mean_r['R_ane']:+8.3f}"
                    f"{mean_r['R_water_eff']:+8.3f}"
                    f"{mean_r['yield_kg_ha']:8.0f}"
                    f"   ETA {eta:.1f}m",
                    flush=True,
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
# CSV
# ══════════════════════════════════════════════════════════════════════
def save_csv(records, path):
    if not records:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f'  ✓  Log → {path}')


# ══════════════════════════════════════════════════════════════════════
# Plot: CAPQL-Robust degradation curves only
# ══════════════════════════════════════════════════════════════════════
def plot_curves(records, out_path, title_suffix='CAPQL-Robust'):
    try:
        import pandas as pd
    except ImportError:
        print('  pandas not available — skipping curves plot')
        return
    df = pd.DataFrame(records)

    n_s = len(SCENARIOS)
    n_m = len(METRICS)
    fig, axes = plt.subplots(n_s, n_m, figsize=(5 * n_m, 3.5 * n_s), squeeze=False)
    fig.suptitle(
        f'{title_suffix} — Resilience under Realistic Faults',
        fontsize=13, fontweight='bold',
    )

    for row, (sid, slabel, flevels, xlabels) in enumerate(SCENARIOS):
        sub = df[df['scenario'] == sid]
        for col, metric in enumerate(METRICS):
            ax = axes[row][col]
            for corner in CORNER_ORDER:
                csub = sub[sub['corner'] == corner]
                vals = [
                    float(csub[csub['level_idx'] == li][metric].values[0])
                    if len(csub[csub['level_idx'] == li]) else np.nan
                    for li in range(len(flevels))
                ]
                ax.plot(range(len(flevels)), vals,
                        color=COLOR[corner], marker=MARKER[corner],
                        lw=1.8, ms=6, label=LABEL[corner])
            ax.axhline(0, color='#aaaaaa', lw=0.8, ls='--', alpha=0.5)
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha='right')
            ax.set_ylim(-1.1, 1.15)
            ax.set_ylabel(METRIC_LABEL[metric], fontsize=9)
            ax.set_title(f'{slabel}\n{METRIC_LABEL[metric]}',
                         fontsize=9, fontweight='bold', pad=4)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, loc='lower left', framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {out_path}')


# ══════════════════════════════════════════════════════════════════════
# Plot: side-by-side comparison v2 vs robust
# ══════════════════════════════════════════════════════════════════════
def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
        return pd.read_csv(path).to_dict('records')   # proper numeric dtypes
    except ImportError:
        with open(path) as f:
            return list(csv.DictReader(f))


def _to_float(rows, col):
    return [float(r[col]) for r in rows]


def plot_comparison(v2_records, rb_records, out_path, metric='R_yield'):
    """
    For each scenario: one subplot with two lines per corner.
    Solid = CAPQL-Robust,  dashed = CAPQL v2.
    Shows how much each model degrades as fault severity increases.
    """
    try:
        import pandas as pd
    except ImportError:
        print('  pandas not available — skipping comparison plot')
        return

    df_v2 = pd.DataFrame(v2_records)
    df_rb = pd.DataFrame(rb_records)

    n_s = len(SCENARIOS)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False)
    # 7 scenarios → 8 subplots, leave last one for legend
    axes_flat = [axes[r][c] for r in range(2) for c in range(4)]

    fig.suptitle(
        f'CAPQL v2  vs  CAPQL-Robust  —  {metric}  under Realistic Faults\n'
        'Solid = CAPQL-Robust   |   Dashed = CAPQL v2',
        fontsize=13, fontweight='bold',
    )

    for idx, (sid, slabel, flevels, xlabels) in enumerate(SCENARIOS):
        ax = axes_flat[idx]
        sub_v2 = df_v2[df_v2['scenario'] == sid]
        sub_rb = df_rb[df_rb['scenario'] == sid]

        for corner in CORNER_ORDER:
            cv2 = sub_v2[sub_v2['corner'] == corner]
            crb = sub_rb[sub_rb['corner'] == corner]

            v2_vals = [
                float(cv2[cv2['level_idx'] == li][metric].values[0])
                if len(cv2[cv2['level_idx'] == li]) else np.nan
                for li in range(len(flevels))
            ]
            rb_vals = [
                float(crb[crb['level_idx'] == li][metric].values[0])
                if len(crb[crb['level_idx'] == li]) else np.nan
                for li in range(len(flevels))
            ]

            ax.plot(range(len(flevels)), rb_vals,
                    color=COLOR[corner], marker=MARKER[corner],
                    lw=2.0, ms=6, ls='-')
            ax.plot(range(len(flevels)), v2_vals,
                    color=COLOR[corner], marker=MARKER[corner],
                    lw=1.5, ms=5, ls='--', alpha=0.7)

        ax.axhline(0, color='#aaaaaa', lw=0.8, ls=':', alpha=0.5)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha='right')
        ax.set_ylim(-1.1, 1.15)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(slabel, fontsize=9, fontweight='bold', pad=4)
        ax.grid(True, alpha=0.2)

    # Last subplot: legend
    ax_leg = axes_flat[7]
    ax_leg.axis('off')
    handles = []
    for corner in CORNER_ORDER:
        h_rb, = ax_leg.plot([], [], color=COLOR[corner], ls='-',  lw=2.0,
                            marker=MARKER[corner], ms=6, label=f'{LABEL[corner]} (Robust)')
        h_v2, = ax_leg.plot([], [], color=COLOR[corner], ls='--', lw=1.5,
                            marker=MARKER[corner], ms=5, alpha=0.7,
                            label=f'{LABEL[corner]} (v2)')
        handles += [h_rb, h_v2]
    ax_leg.legend(handles=handles, fontsize=8.5, loc='center',
                  framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {out_path}')


def plot_comparison_full(v2_records, rb_records, out_path):
    """
    All 3 metrics × 7 scenarios: 21-panel comparison grid.
    Each panel: Robust (solid) vs v2 (dashed), balanced corner only.
    """
    try:
        import pandas as pd
    except ImportError:
        print('  pandas not available — skipping full comparison plot')
        return

    df_v2 = pd.DataFrame(v2_records)
    df_rb = pd.DataFrame(rb_records)

    n_m = len(METRICS)
    n_s = len(SCENARIOS)
    fig, axes = plt.subplots(n_m, n_s, figsize=(4 * n_s, 4 * n_m), squeeze=False)
    fig.suptitle(
        'CAPQL v2  vs  CAPQL-Robust  —  All Metrics  (balanced corner)\n'
        'Solid = CAPQL-Robust   |   Dashed = CAPQL v2',
        fontsize=13, fontweight='bold',
    )

    for col, (sid, slabel, flevels, xlabels) in enumerate(SCENARIOS):
        sub_v2 = df_v2[(df_v2['scenario'] == sid) & (df_v2['corner'] == 'balanced')]
        sub_rb = df_rb[(df_rb['scenario'] == sid) & (df_rb['corner'] == 'balanced')]

        for row, metric in enumerate(METRICS):
            ax = axes[row][col]

            rb_vals = [
                float(sub_rb[sub_rb['level_idx'] == li][metric].values[0])
                if len(sub_rb[sub_rb['level_idx'] == li]) else np.nan
                for li in range(len(flevels))
            ]
            v2_vals = [
                float(sub_v2[sub_v2['level_idx'] == li][metric].values[0])
                if len(sub_v2[sub_v2['level_idx'] == li]) else np.nan
                for li in range(len(flevels))
            ]

            ax.plot(range(len(flevels)), rb_vals,
                    color='#c0392b', marker='o', lw=2.0, ms=6,
                    ls='-',  label='Robust')
            ax.plot(range(len(flevels)), v2_vals,
                    color='#2980b9', marker='s', lw=1.5, ms=5,
                    ls='--', alpha=0.8, label='v2')

            # Shade area between curves (resilience gap)
            x = list(range(len(flevels)))
            ax.fill_between(x, v2_vals, rb_vals,
                            where=[not (np.isnan(a) or np.isnan(b))
                                   for a, b in zip(v2_vals, rb_vals)],
                            alpha=0.15, color='#27ae60',
                            label='Δ resilience')

            ax.axhline(0, color='#aaaaaa', lw=0.8, ls=':', alpha=0.5)
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, fontsize=7, rotation=30, ha='right')
            ax.set_ylim(-1.1, 1.15)
            ax.set_ylabel(METRIC_LABEL[metric], fontsize=9)
            if row == 0:
                ax.set_title(slabel, fontsize=8, fontweight='bold', pad=4)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, loc='lower left', framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {out_path}')


# ══════════════════════════════════════════════════════════════════════
# Degradation summary table
# ══════════════════════════════════════════════════════════════════════
def print_summary(v2_records, rb_records):
    """
    For each scenario: mean degradation from clean baseline
    across all corners (Δ = clean - worst_fault_level).
    """
    try:
        import pandas as pd
    except ImportError:
        return

    df_v2 = pd.DataFrame(v2_records)
    df_rb = pd.DataFrame(rb_records)

    print('\n' + '=' * 70)
    print('  Resilience summary — mean R_yield degradation'
          ' (clean → worst fault level)')
    print(f"  {'Scenario':<22}  {'v2 clean':>10}  {'v2 worst':>10}"
          f"  {'rb clean':>10}  {'rb worst':>10}  {'Δ improvement':>14}")
    print('  ' + '─' * 68)

    for sid, slabel, flevels, xlabels in SCENARIOS:
        for df_m, label in [(df_v2, 'v2'), (df_rb, 'rb')]:
            pass  # placeholder — computed below

        sub_v2 = df_v2[df_v2['scenario'] == sid]
        sub_rb = df_rb[df_rb['scenario'] == sid]

        def corner_mean(sub, li):
            return float(sub[sub['level_idx'] == li]['R_yield'].mean())

        v2_clean = corner_mean(sub_v2, 0)
        v2_worst = min(corner_mean(sub_v2, li) for li in range(len(flevels)))
        rb_clean = corner_mean(sub_rb, 0)
        rb_worst = min(corner_mean(sub_rb, li) for li in range(len(flevels)))

        v2_deg = v2_clean - v2_worst
        rb_deg = rb_clean - rb_worst
        improvement = v2_deg - rb_deg   # positive = robust degrades less

        print(
            f"  {sid:<22}  {v2_clean:>+10.3f}  {v2_worst:>+10.3f}"
            f"  {rb_clean:>+10.3f}  {rb_worst:>+10.3f}"
            f"  {improvement:>+14.3f}"
        )
    print('=' * 70 + '\n')


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('13_resilience_eval_robust.py  —  CAPQL-Robust Resilience Test')
    print('=' * 70)
    print(f'  Model  : {MODEL_DIR}')
    print(f'  Obs dim: {OBS_DIM}  (crop + mask + w, return_mask=True)')
    total = sum(len(s[2]) * len(EVAL_CORNERS) * N_EVAL for s in SCENARIOS)
    print(f'  Episodes: {total}  ({len(SCENARIOS)} scenarios × '
          f'{len(EVAL_CORNERS)} corners × {N_EVAL} seeds)')
    print('=' * 70 + '\n')

    actor   = load_actor(MODEL_DIR)
    t0      = time.time()
    records = run_eval(actor)

    print(f'\n  ✓  Total time: {(time.time()-t0)/60:.1f} min')
    save_csv(records, LOG_PATH)

    try:
        import pandas

        plot_curves(records, PLOT_CURVES)

        v2_records = load_csv(V2_LOG)
        if v2_records:
            print_summary(v2_records, records)
            plot_comparison(v2_records, records, PLOT_CMP_YIELD, metric='R_yield')
            plot_comparison_full(v2_records, records, PLOT_CMP_FULL)
        else:
            print(f'  ⚠  v2 log not found at {V2_LOG} — skipping comparison plots')

    except ImportError:
        print('  pandas not available — skipping plots')

    print('\n  ✓  Done.  Results in /workspace/')
    print('=' * 70)


if __name__ == '__main__':
    main()
