"""
09_plot_pareto_v2.py — Pareto front exploration for CAPQL v2.

Loads ALL training episodes from capql_v2_episode_log.csv.
Each episode is one point in objective space (R_yield, R_ane, R_water_eff).
Color = preference vector w → interpolated RGB so you can see which
preference leads to which outcome.

Generates 2 figures in /workspace/:
  pareto_v2_exploration.png  — training-episode scatter + Pareto frontier
  pareto_v2_radar.png        — radar chart of the 4 eval corners

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 09_plot_pareto_v2.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ====================================================================== #
# Paths                                                                    #
# ====================================================================== #
EP_LOG   = '/workspace/capql_v2_episode_log.csv'
EVAL_CSV = '/workspace/capql_v2_eval_results.csv'
OUT_DIR  = '/workspace/'

# ====================================================================== #
# Corner styling (used for eval markers and radar)                        #
# ====================================================================== #
CORNERS = ['yield', 'n_eff', 'water', 'balanced']
COLOR   = {'yield': '#27ae60', 'n_eff': '#e67e22',
           'water': '#2980b9', 'balanced': '#8e44ad'}
MARKER  = {'yield': 'o', 'n_eff': 's', 'water': '^', 'balanced': 'D'}
LABEL   = {'yield': 'Yield', 'n_eff': 'N-Efficiency',
           'water': 'Water', 'balanced': 'Balanced'}

# RGB arrays for w→color interpolation
_C = {
    'yield': np.array([0x27, 0xae, 0x60]) / 255.0,   # green
    'n_eff': np.array([0xe6, 0x7e, 0x22]) / 255.0,   # orange
    'water': np.array([0x29, 0x80, 0xb9]) / 255.0,   # blue
}


# ====================================================================== #
# Helpers                                                                  #
# ====================================================================== #
def w_to_rgb(wy, wn, ww):
    """Blend the 3 corner colours using the preference weights as mix ratios."""
    rgb = wy * _C['yield'] + wn * _C['n_eff'] + ww * _C['water']
    return np.clip(rgb, 0.0, 1.0)


def pareto_front_2d(xs, ys):
    """
    Return sorted indices of non-dominated points (maximise both axes).
    O(n log n): sort by x descending, keep a point if it improves max-y.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(-xs)          # descending x
    front = []
    max_y = -np.inf
    for i in order:
        if ys[i] > max_y:
            front.append(i)
            max_y = ys[i]
    # sort front by x ascending so we can draw a step curve
    front = sorted(front, key=lambda i: xs[i])
    return np.array(front)


def load_episodes(path):
    """Load episode log → arrays of objectives and preferences."""
    if not os.path.exists(path):
        print(f'  ⚠  Not found: {path}')
        return None
    rows = list(csv.DictReader(open(path)))
    keys = ['R_yield', 'R_ane', 'R_water_eff', 'w_yield', 'w_neff', 'w_water']
    data = {k: np.array([float(r[k]) for r in rows]) for k in keys}
    print(f'  Loaded {len(rows):,} training episodes from {path}')
    return data


def load_eval(path):
    """Load eval CSV → {corner: mean-dict}."""
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path)))
    d = {}
    for r in rows:
        d.setdefault(r['corner'], []).append(r)
    keys = ['R_yield', 'R_ane', 'R_water_eff']
    return {c: {k: np.mean([float(r[k]) for r in rs]) for k in keys}
            for c, rs in d.items()}


# ====================================================================== #
# Figure 1 — Exploration scatter + Pareto frontier                        #
# ====================================================================== #
def plot_exploration(ep, eval_means, out_path):
    """
    3 pairwise 2D projections of the objective space.
    Every training episode is one dot; colour = w vector.
    The Pareto frontier is drawn in each projection.
    Eval corners are overlaid as large labelled markers.
    """
    ry  = ep['R_yield']
    rn  = ep['R_ane']
    rw  = ep['R_water_eff']
    wy  = ep['w_yield']
    wn  = ep['w_neff']
    ww  = ep['w_water']

    colors = np.array([w_to_rgb(wy[i], wn[i], ww[i]) for i in range(len(ry))])

    pairs = [
        (ry, rn,  'R_yield  (grain weight)',    'R_ane  (N efficiency)',
         'Yield  ↔  N-Efficiency'),
        (ry, rw,  'R_yield  (grain weight)',    'R_water  (water efficiency)',
         'Yield  ↔  Water'),
        (rn, rw,  'R_ane  (N efficiency)',      'R_water  (water efficiency)',
         'N-Efficiency  ↔  Water'),
    ]
    eval_xy = {
        (0, 1): ('R_yield', 'R_ane'),
        (0, 2): ('R_yield', 'R_water_eff'),
        (1, 2): ('R_ane',   'R_water_eff'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        'CAPQL v2 — Pareto Front Exploration\n'
        'Each dot = one training episode  •  Colour = preference vector w\n'
        '● Green = yield weight   ● Orange = N-eff weight   ● Blue = water weight',
        fontsize=11, fontweight='bold', y=1.04,
    )

    for col_idx, (ax, (xs, ys, xlabel, ylabel, title)) in enumerate(
            zip(axes, pairs)):

        # ── All training episodes ──────────────────────────────────────
        ax.scatter(xs, ys, c=colors, s=6, alpha=0.18, linewidths=0,
                   zorder=1, rasterized=True)

        # ── Pareto frontier ───────────────────────────────────────────
        pf_idx = pareto_front_2d(xs, ys)
        pf_x   = xs[pf_idx]
        pf_y   = ys[pf_idx]
        # staircase step-line along the frontier
        ax.step(np.append(pf_x, pf_x[-1]),
                np.append(pf_y[0], pf_y),
                where='post',
                color='black', lw=1.6, alpha=0.7, zorder=3,
                label='Pareto frontier')
        ax.scatter(pf_x, pf_y, s=18, color='black', alpha=0.55,
                   zorder=4, linewidths=0)

        # ── Eval corner markers ───────────────────────────────────────
        ek_pair = list(eval_xy.values())[col_idx]
        for c in CORNERS:
            if c not in eval_means:
                continue
            ex = eval_means[c][ek_pair[0]]
            ey = eval_means[c][ek_pair[1]]
            ax.scatter(ex, ey,
                       marker=MARKER[c], s=180,
                       color=COLOR[c], edgecolors='white',
                       linewidths=1.8, zorder=6)
            ax.annotate(
                LABEL[c], xy=(ex, ey),
                xytext=(6, 5), textcoords='offset points',
                fontsize=8, color=COLOR[c], fontweight='bold',
                zorder=7,
            )

        # ── Ideal point ───────────────────────────────────────────────
        ax.scatter(1.0, 1.0, marker='*', s=250, color='gold',
                   edgecolors='#f39c12', linewidths=1.2, zorder=8)

        # ── Formatting ────────────────────────────────────────────────
        ax.axhline(0, color='#bbbbbb', lw=0.7, ls='--', alpha=0.6)
        ax.axvline(0, color='#bbbbbb', lw=0.7, ls='--', alpha=0.6)
        ax.set_xlim(-1.1, 1.15)
        ax.set_ylim(-1.1, 1.15)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')

    # ── Legend ────────────────────────────────────────────────────────
    corner_handles = [
        mpatches.Patch(facecolor=COLOR[c], label=f'{LABEL[c]} corner (eval)')
        for c in CORNERS
    ]
    extra_handles = [
        Line2D([0], [0], color='black', lw=1.6, label='Pareto frontier'),
        Line2D([0], [0], marker='*', color='gold', markeredgecolor='#f39c12',
               markersize=12, lw=0, label='Ideal [1, 1]'),
    ]
    fig.legend(
        handles=corner_handles + extra_handles,
        loc='lower center', ncol=6, fontsize=8.5,
        bbox_to_anchor=(0.5, -0.07), framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  {out_path}')


# ====================================================================== #
# Figure 2 — Radar chart (v2 eval corners only)                          #
# ====================================================================== #
def plot_radar(eval_means, out_path):
    metrics       = ['R_yield', 'R_ane', 'R_water_eff']
    metric_labels = ['R_yield', 'R_ane', 'R_water']
    n = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle(
        'CAPQL v2 — Pareto Front (Radar)\nAll Preference Corners',
        fontsize=13, fontweight='bold',
    )

    for c in CORNERS:
        if c not in eval_means:
            continue
        raw  = [eval_means[c][k] for k in metrics]
        vals = [(v + 1) / 2 for v in raw]    # shift [-1,1] → [0,1]
        vals = vals + vals[:1]

        ax.plot(angles, vals, color=COLOR[c], lw=2.5, label=LABEL[c])
        ax.fill(angles, vals, color=COLOR[c], alpha=0.14)
        ax.scatter(angles[:-1], [(v + 1) / 2 for v in raw],
                   s=65, color=COLOR[c], zorder=5)

    # Gridlines: show original [-1, +1] scale on normalised [0, 1] axis
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['-0.5', '0', '+0.5', '+1.0'],
                       fontsize=8, color='gray')

    # Zero reference circle (at normalised 0.5)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(theta, np.full(200, 0.5),
            color='#555555', lw=1.2, ls='--', alpha=0.5, zorder=0)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.15),
              fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  {out_path}')


# ====================================================================== #
# Main                                                                     #
# ====================================================================== #
def main():
    print('=' * 60)
    print('09_plot_pareto_v2.py — CAPQL v2 Pareto exploration')
    print('=' * 60 + '\n')

    ep         = load_episodes(EP_LOG)
    eval_means = load_eval(EVAL_CSV)

    if ep is None:
        print('❌  Episode log not found — cannot draw exploration plot.')
    else:
        plot_exploration(
            ep, eval_means,
            os.path.join(OUT_DIR, 'pareto_v2_exploration.png'),
        )

    if not eval_means:
        print('❌  Eval CSV not found — cannot draw radar.')
    else:
        print(f'  Eval corners: {sorted(eval_means.keys())}')
        plot_radar(
            eval_means,
            os.path.join(OUT_DIR, 'pareto_v2_radar.png'),
        )

    print('\n✓  Done. Plots saved to', OUT_DIR)
    print('=' * 60)


if __name__ == '__main__':
    main()
