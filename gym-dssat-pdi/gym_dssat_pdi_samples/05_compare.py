"""
Run all 4 rule-based baselines then compare against saved PPO eval results.

Output:
  /workspace/baselines_results.csv   — per-episode rows for all baselines
  /workspace/comparison.csv          — summary comparison table
  Console                            — formatted comparison table

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 -u 05_compare.py
"""
import csv
import os
import numpy as np
from importlib import import_module

SmartFarmSoSEnv = import_module('02_smart_farm_env').SmartFarmSoSEnv

EVAL_CSV        = '/workspace/eval_results.csv'
BASELINES_CSV   = '/workspace/baselines_results.csv'
COMPARISON_CSV  = '/workspace/comparison.csv'
N_EPISODES      = 20   # per baseline — ~4 min total

# ================================================================== #
# Helpers (copied from 01_rule_based_baselines.py)                    #
# ================================================================== #
TAW_MAIZE     = 150.0
P_DEPLETION   = 0.50
MAX_IRRIG_EVT = 25.0


def kc_maize(dap):
    if dap < 20:   return 0.30
    elif dap < 50: return 0.30 + (1.20 - 0.30) * (dap - 20) / 30
    elif dap < 100:return 1.20
    elif dap < 140:return 1.20 - (1.20 - 0.60) * (dap - 100) / 40
    else:          return 0.60


def get_crop(obs, key, default=0.0):
    if obs is None: return float(default)
    val = obs.get(f'crop_{key}', default)
    if val is None: return float(default)
    try:    return float(val)
    except: return float(default)


def run_episode(env, policy_fn):
    obs  = env.reset()
    done = False
    info = {}
    cum_r = 0.0
    state = {}
    while not done:
        action = policy_fn(obs, state)
        obs, R, done, info = env.step(action)
        cum_r += float(R)
    sos = info.get('sos_state', {})
    c   = info.get('reward_components', {})
    return {
        'cum_reward':    round(cum_r, 4),
        'yield_kg_ha':   round(float(sos.get('grnwt',         0.0)), 1),
        'total_N_kg_ha': round(float(sos.get('total_nitrogen', 0.0)), 1),
        'total_W_mm':    round(float(sos.get('total_water',    0.0)), 1),
        'total_rain_mm': round(float(sos.get('total_rain',     0.0)), 1),
        'R_seasonal':    round(float(c.get('R_seasonal', 0.0)), 4),
        'R_yield':       round(float(c.get('R_yield',    0.0)), 4),
        'R_hiad':        round(float(c.get('R_hiad',     0.0)), 4),
        'R_ane':         round(float(c.get('R_ane',      0.0)), 4),
    }


# ================================================================== #
# Policy functions                                                     #
# ================================================================== #
_rng = np.random.default_rng(0)

def policy_random(obs, state):
    return {'anfer': float(_rng.uniform(0, 40)),
            'amir':  float(_rng.uniform(0, 10))}


def policy_fixed(obs, state):
    day = state.get('day', 0) + 1
    state['day'] = day
    return {
        'anfer': 40.0 if day in (30, 60, 90) else 0.0,
        'amir':  8.0  if (day % 5 == 0)      else 0.0,
    }


def policy_stage(obs, state):
    vstage = get_crop(obs, 'vstage', 0.0)
    dap    = get_crop(obs, 'dap',    0.0)
    last   = state.get('vstage', -1.0)
    anfer  = 0.0
    for t in (3.0, 6.0, 10.0):
        if last < t <= vstage:
            anfer = 50.0; break
    state['vstage'] = vstage
    amir = 10.0 if (vstage > 1.0 and int(dap) % 3 == 0) else 0.0
    return {'anfer': anfer, 'amir': amir}


def policy_fao56(obs, state):
    raw = TAW_MAIZE * P_DEPLETION
    day        = state.get('day', 0) + 1
    depletion  = state.get('depletion', 0.0)
    state['day'] = day

    et0  = get_crop(obs, 'eo',   4.0)
    rain = get_crop(obs, 'rain', 0.0)
    kc   = kc_maize(day)
    depletion = max(0.0, depletion + kc * et0 - rain)

    if depletion > raw:
        amir = min(depletion, MAX_IRRIG_EVT)
        depletion -= amir
    else:
        amir = 0.0
    state['depletion'] = depletion

    anfer = 0.0
    for d, dose in zip((35, 65, 95), (40, 40, 40)):
        if day == d: anfer = float(dose); break
    return {'anfer': anfer, 'amir': float(amir)}


# ================================================================== #
# Run baselines                                                        #
# ================================================================== #
BASELINES = [
    ('Random',        policy_random),
    ('Fixed-schedule',policy_fixed),
    ('Stage-based',   policy_stage),
    ('FAO-56',        policy_fao56),
]

FIELDS = ['name', 'episode', 'cum_reward', 'yield_kg_ha', 'total_N_kg_ha',
          'total_W_mm', 'total_rain_mm', 'R_seasonal', 'R_yield', 'R_hiad', 'R_ane']


def run_baselines():
    all_rows = []
    summaries = {}

    with open(BASELINES_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for name, policy_fn in BASELINES:
            print(f"\n  [{name}]  {N_EPISODES} episodes …")
            env = SmartFarmSoSEnv(mode='all', seed=123,
                                  run_dssat_location='run_dssat',
                                  enable_faults=False)
            ep_rows = []
            for ep in range(N_EPISODES):
                row = run_episode(env, policy_fn)
                row['name']    = name
                row['episode'] = ep + 1
                writer.writerow(row)
                f.flush()
                ep_rows.append(row)
                if (ep + 1) % 5 == 0:
                    y = np.mean([r['yield_kg_ha']   for r in ep_rows])
                    n = np.mean([r['total_N_kg_ha']  for r in ep_rows])
                    w = np.mean([r['total_W_mm']      for r in ep_rows])
                    print(f"    ep {ep+1:>3}: yield={y:6.0f} kg/ha  "
                          f"N={n:5.0f}  W={w:5.0f} mm")
            env.close()

            def ms(k): return np.mean([r[k] for r in ep_rows]), np.std([r[k] for r in ep_rows])
            summaries[name] = {k: ms(k) for k in
                ['cum_reward','yield_kg_ha','total_N_kg_ha','total_W_mm',
                 'R_seasonal','R_yield','R_hiad','R_ane']}
            all_rows.extend(ep_rows)

    print(f"\n✓ Baseline episodes saved → {BASELINES_CSV}")
    return summaries


# ================================================================== #
# Load PPO eval results                                                #
# ================================================================== #
def load_ppo_summary():
    if not os.path.exists(EVAL_CSV):
        print(f"  ⚠  PPO eval file not found: {EVAL_CSV}")
        return None

    rows = []
    with open(EVAL_CSV) as f:
        for row in csv.DictReader(f):
            if row['episode'] == 'MEAN':
                continue
            try:
                rows.append({
                    'cum_reward':    float(row['cum_reward']),
                    'yield_kg_ha':   float(row['yield_kg_ha']),
                    'total_N_kg_ha': float(row['total_N_kg_ha']),
                    'total_W_mm':    float(row['total_W_mm']),
                    'R_seasonal':    float(row['R_seasonal']),
                    'R_yield':       float(row['R_yield']),
                    'R_hiad':        float(row['R_hiad']),
                    'R_ane':         float(row['R_ane']),
                })
            except (ValueError, KeyError):
                continue

    def ms(k): return np.mean([r[k] for r in rows]), np.std([r[k] for r in rows])
    return {k: ms(k) for k in
            ['cum_reward','yield_kg_ha','total_N_kg_ha','total_W_mm',
             'R_seasonal','R_yield','R_hiad','R_ane']}


# ================================================================== #
# Print comparison table                                               #
# ================================================================== #
def print_comparison(summaries, ppo):
    W = 78
    print("\n" + "=" * W)
    print("COMPARISON — Rule-based Baselines  vs  PPO (1M steps)")
    print("=" * W)

    metrics = [
        ('yield_kg_ha',   'Yield  (kg/ha)', '{:7.0f}'),
        ('total_N_kg_ha', 'N      (kg/ha)', '{:7.0f}'),
        ('total_W_mm',    'Water    (mm)',  '{:7.0f}'),
        ('R_yield',       'R_yield',        '{:+7.4f}'),
        ('R_hiad',        'R_hiad',         '{:+7.4f}'),
        ('R_ane',         'R_ane',          '{:+7.4f}'),
        ('R_seasonal',    'R_seasonal',     '{:+7.4f}'),
        ('cum_reward',    'cum_reward',     '{:+7.4f}'),
    ]

    names = [n for n, _ in BASELINES] + (['PPO (1M)'] if ppo else [])
    col   = 14

    # header
    print(f"\n  {'Metric':<16}", end='')
    for n in names:
        print(f"  {n:>{col}}", end='')
    print()
    print(f"  {'-'*16}", end='')
    for _ in names:
        print(f"  {'-'*col}", end='')
    print()

    for key, label, fmt in metrics:
        print(f"  {label:<16}", end='')
        for n in [n for n, _ in BASELINES]:
            m, s = summaries[n][key]
            val  = fmt.format(m)
            print(f"  {val:>{col}}", end='')
        if ppo:
            m, s = ppo[key]
            val  = fmt.format(m)
            print(f"  {val:>{col}}", end='')
        print()

    print("=" * W)

    # Save comparison CSV
    comp_fields = ['metric'] + names
    comp_rows = []
    for key, label, fmt in metrics:
        row = {'metric': label}
        for n in [n for n, _ in BASELINES]:
            m, _ = summaries[n][key]
            row[n] = round(m, 4)
        if ppo:
            m, _ = ppo[key]
            row['PPO (1M)'] = round(m, 4)
        comp_rows.append(row)

    with open(COMPARISON_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=comp_fields)
        writer.writeheader()
        writer.writerows(comp_rows)
    print(f"\n✓ Comparison table saved → {COMPARISON_CSV}")


# ================================================================== #
# Main                                                                 #
# ================================================================== #
if __name__ == '__main__':
    print("=" * 78)
    print(f"BASELINE EVALUATION — {N_EPISODES} episodes per policy")
    print("=" * 78)

    summaries = run_baselines()
    ppo       = load_ppo_summary()
    print_comparison(summaries, ppo)
