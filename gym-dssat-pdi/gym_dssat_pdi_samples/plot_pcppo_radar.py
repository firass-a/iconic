"""
PC-PPO v5 — reward-profile radar (same style as CAPQL pareto_v2_radar).

Uses v5 strong-eval corner means (eval_20260605_201221 / logs/eval_20260605_204802.log)
and maps agronomic outcomes to R_yield, R_ane, R_water_eff via the same formulas as
02_smart_farm_env_pcppo.py so the chart is comparable to CAPQL.

Outputs:
  figures/pcppo_v5_radar.png
  figures/pcppo_v5_radar.csv

Run:
  python plot_pcppo_radar.py
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'figures'
OUT_PNG = OUT_DIR / 'pcppo_v5_radar.png'
OUT_CSV = OUT_DIR / 'pcppo_v5_radar.csv'

# v5 strong eval — 20-episode means (logs/eval_20260605_204802.log)
V5_CORNERS = {
    'yield':   {'grnwt': 10866.0, 'tot_N': 718.0, 'tot_W': 1081.0},
    'water':   {'grnwt': 10314.0, 'tot_N': 384.0, 'tot_W': 456.0},
    'fert':    {'grnwt': 9108.0,  'tot_N': 246.0, 'tot_W': 465.0},
    'uniform': {'grnwt': 9926.0,  'tot_N': 329.0, 'tot_W': 699.0},
}

CORNERS = ['yield', 'water', 'fert', 'uniform']
COLOR = {
    'yield':   '#27ae60',
    'water':   '#2980b9',
    'fert':    '#e67e22',
    'uniform': '#8e44ad',
}
LABEL = {
    'yield':   'Yield-Only [1, 0, 0]',
    'water':   'Water-Only [0, 1, 0]',
    'fert':    'Fert-Only [0, 0, 1]',
    'uniform': 'Balanced [1/3, 1/3, 1/3]',
}

BASELINE_YIELD = 7620.0
MAX_EXPECTED_YIELD = 11430.0
BASELINE_ANE = 40.0
MAX_ANE = 80.0
MIN_VIABLE_YIELD = 5000.0
WATER_TARGET = 400.0

METRICS = ['R_yield', 'R_ane', 'R_water_eff']
METRIC_LABELS = ['$R_{yield}$', '$R_{ane}$', '$R_{water}$']


def reward_vec(grnwt, tot_N, tot_W):
    R_yield = float(np.clip(
        (grnwt - BASELINE_YIELD) / (MAX_EXPECTED_YIELD - BASELINE_YIELD),
        -1.0, 1.0,
    ))
    if grnwt >= MIN_VIABLE_YIELD:
        ane = grnwt / max(tot_N, 1.0)
        R_ane = float(np.clip(
            (ane - BASELINE_ANE) / (MAX_ANE - BASELINE_ANE),
            -1.0, 1.0,
        ))
    else:
        R_ane = -1.0
    if grnwt < MIN_VIABLE_YIELD:
        R_water_eff = -1.0
    else:
        R_water_eff = float(np.clip(
            (WATER_TARGET - tot_W) / WATER_TARGET,
            -1.0, 1.0,
        ))
    return {'R_yield': R_yield, 'R_ane': R_ane, 'R_water_eff': R_water_eff}


def load_from_eval_csv(path):
    """Optional: load means if models/eval/eval_20260605_201221.csv exists."""
    if not path.exists():
        return None
    import csv as csvmod
    rows = list(csvmod.DictReader(open(path, newline='', encoding='utf-8')))
    name_map = {
        'corner_yield': 'yield',
        'corner_water': 'water',
        'corner_fert': 'fert',
        'uniform': 'uniform',
    }
    buckets = {v: [] for v in name_map.values()}
    for r in rows:
        key = name_map.get(r.get('pref_name', ''))
        if key is None:
            continue
        buckets[key].append(r)
    if not any(buckets.values()):
        return None
    out = {}
    for key, rs in buckets.items():
        if not rs:
            continue
        if 'R_yield' in rs[0]:
            out[key] = {
                m: float(np.mean([float(x[m]) for x in rs]))
                for m in METRICS
            }
        else:
            out[key] = reward_vec(
                float(np.mean([float(x['grnwt']) for x in rs])),
                float(np.mean([float(x['N_applied']) for x in rs])),
                float(np.mean([float(x['tot_W']) for x in rs])),
            )
    return out if out else None


def build_means():
    csv_path = HERE / 'models' / 'eval' / 'eval_20260605_201221.csv'
    means = load_from_eval_csv(csv_path)
    if means:
        return means
    return {
        c: reward_vec(d['grnwt'], d['tot_N'], d['tot_W'])
        for c, d in V5_CORNERS.items()
    }


def plot_radar(eval_means, out_path):
    n = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle(
        'PC-PPO v5 — Reward Profile per Preference Corner',
        fontsize=13, fontweight='bold',
    )

    for c in CORNERS:
        if c not in eval_means:
            continue
        raw = [eval_means[c][k] for k in METRICS]
        vals = [(v + 1) / 2 for v in raw]
        vals = vals + vals[:1]

        ax.plot(angles, vals, color=COLOR[c], lw=2.5, label=LABEL[c])
        ax.fill(angles, vals, color=COLOR[c], alpha=0.14)
        ax.scatter(angles[:-1], [(v + 1) / 2 for v in raw],
                   s=65, color=COLOR[c], zorder=5)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['-0.5', '0', '+0.5', '+1.0'],
                       fontsize=8, color='gray')
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta, np.full(200, 0.5),
            color='#555555', lw=1.2, ls='--', alpha=0.5, zorder=0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15),
              fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'OK  {out_path}')


def save_csv(eval_means, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['corner', 'label'] + METRICS)
        for c in CORNERS:
            if c not in eval_means:
                continue
            w.writerow([c, LABEL[c]] + [f"{eval_means[c][m]:.4f}" for m in METRICS])
    print(f'OK  {out_path}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    means = build_means()
    print('PC-PPO v5 reward profile (computed):')
    for c in CORNERS:
        m = means[c]
        print(f"  {LABEL[c]:30s}  R_yield={m['R_yield']:+.3f}  "
              f"R_ane={m['R_ane']:+.3f}  R_water={m['R_water_eff']:+.3f}")
    plot_radar(means, OUT_PNG)
    save_csv(means, OUT_CSV)


if __name__ == '__main__':
    main()
