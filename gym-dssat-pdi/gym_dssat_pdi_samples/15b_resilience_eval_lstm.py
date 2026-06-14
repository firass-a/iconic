"""
15b_resilience_eval_lstm.py  —  Resilience evaluation of CAPQL-Robust LSTM.

Runs the trained LSTM actor through 7 fault scenarios × severity levels
× 4 preference corners and records:
  R_yield, R_ane, R_water_eff, yield_kg_ha, total_N_kg_ha, total_W_mm

Prints a full table to console and saves:
  /workspace/resilience_lstm_log.csv
  /workspace/fig_lstm_fault_table.png   (heatmap of yield per fault × corner)

Run inside Docker:
    /opt/gym_dssat_pdi/bin/python3 15b_resilience_eval_lstm.py
"""

import os, sys, csv, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib import import_module

# ── Import model architecture from training script ─────────────────────
_train = import_module('14_capql_train_robust_v2')
LSTMActor        = _train.LSTMActor
HIDDEN           = _train.HIDDEN
CROP_MASK_DIM    = _train.CROP_MASK_DIM
N_OBJ            = _train.N_OBJ
DEVICE           = _train.DEVICE

CAPQLRobustEnvV2 = import_module('capql_env_robust_v2').CAPQLRobustEnvV2
_FaultStateV3    = import_module('capql_env_robust_v2')._FaultStateV3

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_DIR  = '/workspace/capql_robust_lstm'
OUT_CSV    = '/workspace/resilience_lstm_log.csv'
OUT_FIG    = '/workspace/fig_lstm_fault_table.png'

# ── Eval config ────────────────────────────────────────────────────────
N_EVAL_EPS = 5

CORNERS = {
    'yield':    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'n_eff':    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'water':    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'balanced': np.array([1/3, 1/3, 1/3], dtype=np.float32),
}

# ── Fault scenarios (same as resilience_eval_robust.py) ───────────────
SCENARIOS = [
    {
        'id':     'clean',
        'label':  'No Fault (Clean)',
        'levels': [{}],
        'xlabels':['clean'],
    },
    {
        'id':     'dropout_soil',
        'label':  'Soil Moisture Dropout',
        'levels': [
            {},
            {'dropout_rate': 0.05},
            {'dropout_rate': 0.10},
            {'dropout_rate': 0.20},
            {'dropout_rate': 0.50},
        ],
        'xlabels': ['0%', '5%', '10%', '20%', '50%'],
    },
    {
        'id':     'packet_loss',
        'label':  'Packet Loss (Type 1)',
        'levels': [
            {},
            {'packet_loss_prob': 0.05},
            {'packet_loss_prob': 0.10},
            {'packet_loss_prob': 0.20},
            {'packet_loss_prob': 0.50},
        ],
        'xlabels': ['0%', '5%', '10%', '20%', '50%'],
    },
    {
        'id':     'stuck_soil',
        'label':  'Stuck Sensor (sw_mean)',
        'levels': [
            {},
            {'stuck_sensors': [0], 'stuck_day': 120},
            {'stuck_sensors': [0], 'stuck_day': 90},
            {'stuck_sensors': [0], 'stuck_day': 60},
            {'stuck_sensors': [0], 'stuck_day': 30},
            {'stuck_sensors': [0], 'stuck_day': 10},
        ],
        'xlabels': ['No fault','Day 120','Day 90','Day 60','Day 30','Day 10'],
    },
    {
        'id':     'bias_soil',
        'label':  'Soil Bias (offset)',
        'levels': [
            {},
            {'sensor_bias': {0: -0.20}},
            {'sensor_bias': {0: -0.10}},
            {'sensor_bias': {0: +0.10}},
            {'sensor_bias': {0: +0.20}},
            {'sensor_bias': {0: +0.30}},
        ],
        'xlabels': ['0', '-0.20', '-0.10', '+0.10', '+0.20', '+0.30'],
    },
    {
        'id':     'actuator_N',
        'label':  'N Spreader Efficiency',
        'levels': [
            {'N_efficiency': 1.00},
            {'N_efficiency': 0.80},
            {'N_efficiency': 0.60},
            {'N_efficiency': 0.40},
            {'N_efficiency': 0.20},
            {'N_efficiency': 0.00},
        ],
        'xlabels': ['100%','80%','60%','40%','20%','0%'],
    },
    {
        'id':     'actuator_W',
        'label':  'Irrigation Pump Efficiency',
        'levels': [
            {'W_efficiency': 1.00},
            {'W_efficiency': 0.80},
            {'W_efficiency': 0.60},
            {'W_efficiency': 0.40},
            {'W_efficiency': 0.20},
            {'W_efficiency': 0.00},
        ],
        'xlabels': ['100%','80%','60%','40%','20%','0%'],
    },
]


# ══════════════════════════════════════════════════════════════════════
# Rollout one episode with LSTM
# ══════════════════════════════════════════════════════════════════════
def run_episode(actor, env, w_corner, fault_kwargs):
    """Returns (rv, sos_info) for one episode."""
    env.current_w   = w_corner.copy()
    env._max_anfer  = float(np.clip(200.0 * (1.0 - w_corner[1])**2, 2.0, 200.0))
    env._max_amir   = float(np.clip( 50.0 * (1.0 - w_corner[2])**2, 3.0,  50.0))
    env._fault_state = _FaultStateV3(fault_kwargs)

    # Reset DSSAT directly (avoid re-sampling w and fault in outer env.reset)
    sos_obs = env._sos_env.reset()
    raw_obs = env._encode(sos_obs)
    obs19   = env._fault_state.corrupt(raw_obs)
    obs19[-N_OBJ:] = w_corner

    # Fresh LSTM hidden state at episode start
    h = torch.zeros(1, 1, HIDDEN, device=DEVICE)
    c = torch.zeros(1, 1, HIDDEN, device=DEVICE)

    done      = False
    last_info = {}

    while not done:
        obs16          = obs19[:CROP_MASK_DIM]
        action, h, c   = actor.step(obs16, w_corner, h, c)
        action         = np.clip(action,
                                 [0.0, 0.0],
                                 [env._max_anfer, env._max_amir])
        obs19, _, done, _, last_info = env.step(action)
        obs19[-N_OBJ:] = w_corner

    rv  = np.array(last_info.get('reward_vec', [0, 0, 0]), dtype=np.float32)
    sos = last_info.get('sos_state', {})
    return rv, sos


# ══════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════
def evaluate(actor):
    results = []
    env = CAPQLRobustEnvV2(dssat_seed=1)
    env.curriculum_phase = 3
    env.reset()

    total_runs = sum(len(s['levels']) for s in SCENARIOS) * len(CORNERS) * N_EVAL_EPS
    done_runs  = 0
    t0 = time.time()

    for scen in SCENARIOS:
        sid    = scen['id']
        slabel = scen['label']

        for level_idx, fault_kwargs in enumerate(scen['levels']):
            for corner_name, w_corner in CORNERS.items():
                rv_list, sos_list = [], []

                for seed in range(7000, 7000 + N_EVAL_EPS):
                    try:
                        env._sos_env._seed = seed
                    except Exception:
                        pass
                    rv, sos = run_episode(actor, env, w_corner, fault_kwargs)
                    rv_list.append(rv)
                    sos_list.append(sos)
                    done_runs += 1

                rv_mean = np.mean(rv_list, axis=0)
                results.append({
                    'scenario':      sid,
                    'scenario_label':slabel,
                    'level_idx':     level_idx,
                    'level_label':   scen['xlabels'][level_idx],
                    'corner':        corner_name,
                    'R_yield':       float(rv_mean[0]),
                    'R_ane':         float(rv_mean[1]),
                    'R_water_eff':   float(rv_mean[2]),
                    'yield_kg_ha':   float(np.mean([s.get('grnwt', 0) for s in sos_list])),
                    'total_N_kg_ha': float(np.mean([s.get('total_nitrogen', 0) for s in sos_list])),
                    'total_W_mm':    float(np.mean([s.get('total_water', 0) for s in sos_list])),
                })

                elapsed = (time.time() - t0) / 60
                eta     = elapsed / done_runs * (total_runs - done_runs)
                print(f'  [{done_runs:>4}/{total_runs}]  '
                      f'{sid:<18} L{level_idx}  {corner_name:<10} '
                      f'yield={rv_mean[0]:+.3f}  N={rv_mean[1]:+.3f}  '
                      f'W={rv_mean[2]:+.3f}  '
                      f'yield_kg={results[-1]["yield_kg_ha"]:.0f}  '
                      f'ETA {eta:.1f}m',
                      flush=True)

    env.close()
    return results


# ══════════════════════════════════════════════════════════════════════
# Print table
# ══════════════════════════════════════════════════════════════════════
def print_table(results):
    print('\n' + '=' * 100)
    print('  CAPQL-Robust LSTM — Results per fault scenario × corner')
    print('  (yield_kg_ha | N_kg_ha | W_mm  for each corner)')
    print('=' * 100)

    hdr = (f"  {'Scenario':<26} {'Level':<10} "
           f"{'── YIELD corner ──':^32} "
           f"{'── N-EFF corner ──':^32} "
           f"{'── WATER corner ──':^32} "
           f"{'── BALANCED ──':^32}")
    col = '  yield(kg)    N(kg)   W(mm)'
    subhdr = f"  {'':26} {'':10}  {col}  {col}  {col}  {col}"
    print(hdr)
    print(subhdr)
    print('  ' + '─' * 98)

    for scen in SCENARIOS:
        sid = scen['id']
        for li, xlabel in enumerate(scen['xlabels']):
            row_parts = [f"  {scen['label'][:25]:<26} {xlabel:<10}"]
            for corner in ['yield', 'n_eff', 'water', 'balanced']:
                sub = [r for r in results
                       if r['scenario'] == sid
                       and r['level_idx'] == li
                       and r['corner'] == corner]
                if sub:
                    r = sub[0]
                    row_parts.append(
                        f"  {r['yield_kg_ha']:>7.0f}"
                        f" {r['total_N_kg_ha']:>6.0f}"
                        f" {r['total_W_mm']:>6.0f}"
                    )
                else:
                    row_parts.append('  ' + ' ' * 22)
            print(''.join(row_parts))
        print()

    print('=' * 100 + '\n')


def print_reward_table(results):
    print('\n' + '=' * 100)
    print('  CAPQL-Robust LSTM — Reward components per fault × corner')
    print('  (R_yield | R_ane | R_water  for each corner)')
    print('=' * 100)

    for scen in SCENARIOS:
        sid = scen['id']
        print(f"\n  {scen['label']}")
        print(f"  {'Level':<12} "
              f"{'yield-R_y':>10} {'yield-R_n':>10} {'yield-R_w':>10}  "
              f"{'neff-R_y':>10} {'neff-R_n':>10} {'neff-R_w':>10}  "
              f"{'water-R_y':>10} {'water-R_n':>10} {'water-R_w':>10}  "
              f"{'bal-R_y':>10} {'bal-R_n':>10} {'bal-R_w':>10}")
        print('  ' + '─' * 95)

        for li, xlabel in enumerate(scen['xlabels']):
            row = [f"  {xlabel:<12}"]
            for corner in ['yield', 'n_eff', 'water', 'balanced']:
                sub = [r for r in results
                       if r['scenario'] == sid
                       and r['level_idx'] == li
                       and r['corner'] == corner]
                if sub:
                    r = sub[0]
                    row.append(
                        f" {r['R_yield']:>+10.3f}"
                        f" {r['R_ane']:>+10.3f}"
                        f" {r['R_water_eff']:>+10.3f} "
                    )
                else:
                    row.append(' ' * 34)
            print(''.join(row))

    print('\n' + '=' * 100 + '\n')


# ══════════════════════════════════════════════════════════════════════
# Figure: yield heatmap across faults and corners
# ══════════════════════════════════════════════════════════════════════
def plot_yield_table(results):
    corner_list = ['yield', 'n_eff', 'water', 'balanced']
    corner_label = {'yield': 'Yield corner', 'n_eff': 'N-Eff corner',
                    'water': 'Water corner', 'balanced': 'Balanced'}

    # One subplot per corner, rows = scenarios, cols = severity levels
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    max_levels = max(len(s['levels']) for s in SCENARIOS)
    row_labels = [s['label'] for s in SCENARIOS]

    for ax, corner in zip(axes_flat, corner_list):
        mat_yield = np.full((len(SCENARIOS), max_levels), np.nan)
        mat_N     = np.full((len(SCENARIOS), max_levels), np.nan)
        mat_W     = np.full((len(SCENARIOS), max_levels), np.nan)

        for r, scen in enumerate(SCENARIOS):
            for li in range(len(scen['levels'])):
                sub = [x for x in results
                       if x['scenario'] == scen['id']
                       and x['level_idx'] == li
                       and x['corner'] == corner]
                if sub:
                    mat_yield[r, li] = sub[0]['yield_kg_ha']
                    mat_N[r, li]     = sub[0]['total_N_kg_ha']
                    mat_W[r, li]     = sub[0]['total_W_mm']

        im = ax.imshow(mat_yield, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=12000)

        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xticks(range(max_levels))
        ax.set_xticklabels([f'L{i}' for i in range(max_levels)], fontsize=8)
        ax.set_title(corner_label[corner], fontsize=10, fontweight='bold')
        ax.set_xlabel('Fault severity level →', fontsize=8)

        # Annotate: yield / N / W
        for r in range(len(SCENARIOS)):
            n_lev = len(SCENARIOS[r]['levels'])
            for li in range(n_lev):
                yld = mat_yield[r, li]
                n_v = mat_N[r, li]
                w_v = mat_W[r, li]
                if not np.isnan(yld):
                    txt_color = 'white' if yld < 4000 else 'black'
                    ax.text(li, r,
                            f'{yld:.0f}\nN:{n_v:.0f}\nW:{w_v:.0f}',
                            ha='center', va='center', fontsize=5.5,
                            color=txt_color, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Yield (kg/ha)', fraction=0.03, pad=0.02)

    fig.suptitle(
        'CAPQL-Robust LSTM — Yield / N / Water per Fault Scenario × Preference Corner\n'
        '(green = high yield, red = low yield  |  each cell: yield kg/ha / N kg/ha / W mm)',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {OUT_FIG}')


# ══════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('15b_resilience_eval_lstm.py — CAPQL-Robust LSTM fault evaluation')
    print('=' * 70)

    actor_path = os.path.join(MODEL_DIR, 'actor.pt')
    if not os.path.exists(actor_path):
        print(f'  ERROR: actor not found at {actor_path}')
        sys.exit(1)

    actor = LSTMActor().to(DEVICE)
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    actor.eval()
    print(f'  ✓  Loaded LSTM actor from {actor_path}')

    total_scenarios = sum(len(s['levels']) for s in SCENARIOS)
    print(f'  Scenarios : {len(SCENARIOS)}  |  Severity levels total : {total_scenarios}')
    print(f'  Corners   : {len(CORNERS)}  |  Episodes per cell : {N_EVAL_EPS}')
    print(f'  Total rollouts : {total_scenarios * len(CORNERS) * N_EVAL_EPS}\n')

    results = evaluate(actor)

    # Save CSV
    fields = ['scenario','scenario_label','level_idx','level_label','corner',
              'R_yield','R_ane','R_water_eff','yield_kg_ha','total_N_kg_ha','total_W_mm']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f'\n  ✓  Results saved → {OUT_CSV}')

    print_table(results)
    print_reward_table(results)
    plot_yield_table(results)

    print('Done.')


if __name__ == '__main__':
    main()
