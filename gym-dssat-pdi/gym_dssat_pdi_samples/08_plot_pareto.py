"""
08_plot_pareto.py — Pareto front visualizations for CAPQL v1 vs v2.

Reads eval CSVs saved by 06_capql_train.py and 07_capql_train_v2.py.
Generates 3 figures in /workspace/:

  pareto_2d.png        — 2D pairwise Pareto projections (3 objective pairs)
  pareto_radar.png     — Radar chart: all corners, v1 vs v2 side-by-side
  pareto_resources.png — Agricultural outcomes (yield, N, water) v1 vs v2

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 08_plot_pareto.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ====================================================================== #
# Paths                                                                    #
# ====================================================================== #
V1_CSV  = '/workspace/capql_eval_results.csv'
V2_CSV  = '/workspace/capql_v2_eval_results.csv'
OUT_DIR = '/workspace/'

# ====================================================================== #
# Styling                                                                  #
# ====================================================================== #
CORNERS = ['yield', 'n_eff', 'water', 'balanced']
COLOR   = {
    'yield':    '#27ae60',
    'n_eff':    '#e67e22',
    'water':    '#2980b9',
    'balanced': '#8e44ad',
}
MARKER  = {'yield': 'o', 'n_eff': 's', 'water': '^', 'balanced': 'D'}
LABEL   = {'yield': 'Yield', 'n_eff': 'N-Efficiency',
           'water': 'Water', 'balanced': 'Balanced'}

# ====================================================================== #
# Data loading                                                             #
# ====================================================================== #
def load_csv(path):
    """Load eval CSV → {corner: [row_dicts]}."""
    if not os.path.exists(path):
        print(f'  ⚠  Not found: {path}')
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    d = {}
    for r in rows:
        d.setdefault(r['corner'], []).append(r)
    return d


def aggregate(data):
    """Compute mean ± std per corner for all numeric metrics."""
    keys = ['R_yield', 'R_ane', 'R_water_eff',
            'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm']
    out = {}
    for corner, rows in data.items():
        vals = {k: [float(r[k]) for r in rows] for k in keys}
        out[corner] = {k: float(np.mean(v)) for k, v in vals.items()}
        out[corner].update({k + '_std': float(np.std(v)) for k, v in vals.items()})
    return out


# ====================================================================== #
# Figure 1 — 2D Pareto projections                                        #
# ====================================================================== #
def plot_pareto_2d(m1, m2, out_path):
    pairs = [
        ('R_yield',     'R_ane',       'Yield  vs  N-Efficiency'),
        ('R_yield',     'R_water_eff', 'Yield  vs  Water'),
        ('R_ane',       'R_water_eff', 'N-Efficiency  vs  Water'),
    ]
    xlabels = {
        'R_yield':     'R_yield  (grain weight)',
        'R_ane':       'R_ane  (N efficiency)',
        'R_water_eff': 'R_water  (water efficiency)',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'CAPQL — Pareto Front (2D Projections)\n'
        'hollow = v1 (Dirichlet only)   •   filled = v2 (corner injection)',
        fontsize=12, fontweight='bold', y=1.03,
    )

    for ax, (xk, yk, title) in zip(axes, pairs):

        for c in CORNERS:
            col = COLOR[c]
            mk  = MARKER[c]

            # v1 — hollow markers
            if c in m1:
                x1 = m1[c][xk];  y1 = m1[c][yk]
                xe1 = m1[c][xk + '_std'];  ye1 = m1[c][yk + '_std']
                ax.errorbar(x1, y1, xerr=xe1, yerr=ye1,
                            fmt=mk, color=col,
                            mfc='white', mec=col, ms=10, mew=2,
                            capsize=3, elinewidth=1, alpha=0.85,
                            zorder=3)

            # v2 — filled markers
            if c in m2:
                x2 = m2[c][xk];  y2 = m2[c][yk]
                xe2 = m2[c][xk + '_std'];  ye2 = m2[c][yk + '_std']
                ax.errorbar(x2, y2, xerr=xe2, yerr=ye2,
                            fmt=mk, color=col,
                            mfc=col, mec=col, ms=12,
                            capsize=3, elinewidth=1,
                            label=LABEL[c], zorder=4)

                # Arrow v1 → v2
                if c in m1:
                    ax.annotate(
                        '', xy=(x2, y2),
                        xytext=(m1[c][xk], m1[c][yk]),
                        arrowprops=dict(arrowstyle='->', color=col,
                                        lw=1.5, alpha=0.45),
                        zorder=2,
                    )

            # Corner label next to v2 point
            if c in m2:
                ax.annotate(
                    LABEL[c],
                    xy=(m2[c][xk], m2[c][yk]),
                    xytext=(6, 4), textcoords='offset points',
                    fontsize=7, color=col, fontweight='bold',
                )

        # Ideal point
        ax.scatter(1.0, 1.0, marker='*', s=220, color='gold',
                   edgecolors='#f39c12', linewidths=1.2,
                   zorder=6, label='Ideal [1,1]')

        # Zero-reference lines
        ax.axhline(0, color='#aaaaaa', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(0, color='#aaaaaa', lw=0.8, ls='--', alpha=0.6)

        ax.set_xlim(-1.1, 1.15)
        ax.set_ylim(-1.1, 1.15)
        ax.set_xlabel(xlabels[xk], fontsize=10)
        ax.set_ylabel(xlabels[yk], fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.legend(fontsize=7.5, loc='lower right', framealpha=0.85)
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  {out_path}')


# ====================================================================== #
# Figure 2 — Radar chart                                                  #
# ====================================================================== #
def plot_radar(m1, m2, out_path):
    """
    Side-by-side radar for v1 and v2.
    Values are normalised from [-1,+1] → [0,1] for the polar display.
    Gridlines are labelled with original [-1,+1] values.
    """
    metrics       = ['R_yield', 'R_ane', 'R_water_eff']
    metric_labels = ['R_yield', 'R_ane', 'R_water']
    n = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])   # close polygon

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7),
        subplot_kw=dict(polar=True),
    )
    fig.suptitle(
        'CAPQL — Radar Chart (Pareto Corners)\n'
        'Radial scale: −1 (center) → +1 (edge)',
        fontsize=13, fontweight='bold',
    )

    titles = [
        'v1  —  Dirichlet([1,1,1])',
        'v2  —  Corner Injection (20 %)',
    ]

    for ax, m, title in zip(axes, [m1, m2], titles):

        for c in CORNERS:
            if c not in m:
                continue
            # Normalise: [-1,1] → [0,1]  (so −1 → 0, 0 → 0.5, +1 → 1)
            raw  = [m[c][k] for k in metrics]
            vals = [(v + 1) / 2 for v in raw]
            vals = vals + vals[:1]

            ax.plot(angles, vals, color=COLOR[c], lw=2.2, label=LABEL[c])
            ax.fill(angles, vals, color=COLOR[c], alpha=0.12)
            # Mark each vertex
            ax.scatter(angles[:-1],
                       [(v + 1) / 2 for v in raw],
                       s=50, color=COLOR[c], zorder=5)

        # Gridlines at normalised 0.25, 0.5, 0.75, 1.0
        # → correspond to original −0.5, 0, +0.5, +1.0
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(['-0.5', '0', '+0.5', '+1.0'],
                           fontsize=7.5, color='gray')

        # Dashed reference circle at 0 (normalised 0.5)
        theta_ref = np.linspace(0, 2 * np.pi, 200)
        ax.plot(theta_ref, np.full(200, 0.5),
                color='#555555', lw=1.0, ls='--', alpha=0.5, zorder=0)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.45, 1.15),
            fontsize=9.5,
            framealpha=0.85,
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  {out_path}')


# ====================================================================== #
# Figure 3 — Agricultural resource comparison                             #
# ====================================================================== #
def plot_resources(m1, m2, out_path):
    """
    Grouped bar chart showing yield, N applied, and water applied
    for each preference corner across v1 and v2.
    """
    metrics = [
        ('yield_kg_ha',   'Grain Yield  (kg/ha)',    7620,  'Baseline 7620 kg/ha'),
        ('total_N_kg_ha', 'N Applied  (kg/ha)',       None,  None),
        ('total_W_mm',    'Water Applied  (mm)',      400,   'Target 400 mm'),
    ]

    n_corners = len(CORNERS)
    x = np.arange(n_corners)
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        'CAPQL — Agricultural Outcomes per Preference Corner  (v1 vs v2)',
        fontsize=13, fontweight='bold',
    )

    for ax, (key, ylabel, ref_val, ref_label) in zip(axes, metrics):
        vals1 = [m1.get(c, {}).get(key, 0)            for c in CORNERS]
        vals2 = [m2.get(c, {}).get(key, 0)            for c in CORNERS]
        err1  = [m1.get(c, {}).get(key + '_std', 0)   for c in CORNERS]
        err2  = [m2.get(c, {}).get(key + '_std', 0)   for c in CORNERS]
        colors = [COLOR[c] for c in CORNERS]

        # v1 — hatched, lighter
        ax.bar(x - w / 2, vals1, w, yerr=err1,
               color=colors, alpha=0.40, hatch='//',
               capsize=4, ecolor='#555555',
               edgecolor='white', linewidth=0.5)

        # v2 — solid
        ax.bar(x + w / 2, vals2, w, yerr=err2,
               color=colors, alpha=0.88,
               capsize=4, ecolor='#555555',
               edgecolor='white', linewidth=0.5)

        # Reference line
        if ref_val is not None:
            ax.axhline(ref_val, color='#c0392b', lw=1.8, ls='--', alpha=0.8,
                       label=ref_label)

        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[c] for c in CORNERS], fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_axisbelow(True)

        if ref_val is not None:
            ax.legend(fontsize=8, loc='best', framealpha=0.85)

    # Shared legend for v1 / v2 and corners
    legend_handles = [
        mpatches.Patch(facecolor='#aaaaaa', alpha=0.40,
                       hatch='//', label='v1  (Dirichlet only)'),
        mpatches.Patch(facecolor='#aaaaaa', alpha=0.88,
                       label='v2  (corner injection)'),
    ] + [
        mpatches.Patch(facecolor=COLOR[c], label=LABEL[c])
        for c in CORNERS
    ]
    fig.legend(handles=legend_handles,
               loc='lower center', ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, -0.08),
               framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'✓  {out_path}')


# ====================================================================== #
# Main                                                                     #
# ====================================================================== #
def main():
    print('=' * 60)
    print('08_plot_pareto.py — CAPQL Pareto visualizations')
    print('=' * 60)

    d1 = load_csv(V1_CSV)
    d2 = load_csv(V2_CSV)

    if not d1 and not d2:
        print('❌  No eval data found. Run training first.')
        return

    m1 = aggregate(d1)
    m2 = aggregate(d2)

    print(f'\n  v1 corners loaded : {sorted(m1.keys())}')
    print(f'  v2 corners loaded : {sorted(m2.keys())}')
    print()

    plot_pareto_2d(m1, m2,
                   os.path.join(OUT_DIR, 'pareto_2d.png'))

    plot_radar(m1, m2,
               os.path.join(OUT_DIR, 'pareto_radar.png'))

    plot_resources(m1, m2,
                   os.path.join(OUT_DIR, 'pareto_resources.png'))

    print('\n✓  All plots saved to', OUT_DIR)
    print('=' * 60)


if __name__ == '__main__':
    main()
