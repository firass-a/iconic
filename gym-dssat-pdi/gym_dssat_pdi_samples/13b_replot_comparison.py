"""
13b_replot_comparison.py  —  regenerate comparison plots from existing CSVs.

Run this instead of re-running the full 13_resilience_eval_robust.py
if you already have resilience_robust_log.csv and resilience_v2_log.csv.

    /opt/gym_dssat_pdi/bin/python3 13b_replot_comparison.py
"""
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V2_LOG      = '/workspace/resilience_v2_log.csv'
RB_LOG      = '/workspace/resilience_robust_log.csv'
PLOT_CURVES     = '/workspace/resilience_robust_curves.png'
PLOT_CMP_YIELD  = '/workspace/resilience_comparison.png'
PLOT_CMP_FULL   = '/workspace/resilience_comparison_full.png'

METRICS      = ['R_yield', 'R_ane', 'R_water_eff']
METRIC_LABEL = {'R_yield': 'R_yield', 'R_ane': 'R_ane', 'R_water_eff': 'R_water'}

CORNER_ORDER = ['yield', 'n_eff', 'water', 'balanced']
COLOR  = {'yield': '#27ae60', 'n_eff': '#e67e22',
          'water': '#2980b9', 'balanced': '#8e44ad'}
LABEL  = {'yield': 'Yield', 'n_eff': 'N-Eff',
          'water': 'Water', 'balanced': 'Balanced'}
MARKER = {'yield': 'o', 'n_eff': 's', 'water': '^', 'balanced': 'D'}

SCENARIOS = [
    ('dropout_soil',  'Soil Moisture Dropout',        5, ['0%','5%','10%','20%','50%']),
    ('dropout_all',   'All Sensors Dropout',           5, ['0%','5%','10%','20%','50%']),
    ('packet_loss',   'Packet Loss (Type 1)',           5, ['0%','5%','10%','20%','50%']),
    ('stuck_soil',    'Stuck Sensor (Type 2)',          6, ['No fault','Day 120','Day 90','Day 60','Day 30','Day 10']),
    ('bias_soil',     'Soil Bias (Type 2)',             6, ['0','-0.20','-0.10','+0.10','+0.20','+0.30']),
    ('actuator_N',    'N Spreader Efficiency',         6, ['100%','80%','60%','40%','20%','0%']),
    ('actuator_W',    'Irrigation Pump Efficiency',    6, ['100%','80%','60%','40%','20%','0%']),
]


def load(path):
    if not os.path.exists(path):
        print(f'  ✗  Not found: {path}')
        return None
    df = pd.read_csv(path)
    print(f'  ✓  Loaded {len(df)} rows from {path}')
    return df


def plot_curves(df_rb):
    n_s = len(SCENARIOS)
    n_m = len(METRICS)
    fig, axes = plt.subplots(n_s, n_m, figsize=(5*n_m, 3.5*n_s), squeeze=False)
    fig.suptitle('CAPQL-Robust — Resilience under Realistic Faults',
                 fontsize=13, fontweight='bold')

    for row, (sid, slabel, n_levels, xlabels) in enumerate(SCENARIOS):
        sub = df_rb[df_rb['scenario'] == sid]
        for col, metric in enumerate(METRICS):
            ax = axes[row][col]
            for corner in CORNER_ORDER:
                csub = sub[sub['corner'] == corner]
                vals = [
                    float(csub[csub['level_idx'] == li][metric].values[0])
                    if len(csub[csub['level_idx'] == li]) else np.nan
                    for li in range(n_levels)
                ]
                ax.plot(range(n_levels), vals, color=COLOR[corner],
                        marker=MARKER[corner], lw=1.8, ms=6, label=LABEL[corner])
            ax.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
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
    plt.savefig(PLOT_CURVES, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {PLOT_CURVES}')


def plot_comparison(df_v2, df_rb, metric='R_yield'):
    n_s = len(SCENARIOS)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False)
    axes_flat = [axes[r][c] for r in range(2) for c in range(4)]

    fig.suptitle(
        f'CAPQL v2  vs  CAPQL-Robust  —  {metric}  under Realistic Faults\n'
        'Solid = CAPQL-Robust   |   Dashed = CAPQL v2',
        fontsize=13, fontweight='bold')

    for idx, (sid, slabel, n_levels, xlabels) in enumerate(SCENARIOS):
        ax = axes_flat[idx]
        sub_v2 = df_v2[df_v2['scenario'] == sid]
        sub_rb = df_rb[df_rb['scenario'] == sid]

        for corner in CORNER_ORDER:
            cv2 = sub_v2[sub_v2['corner'] == corner]
            crb = sub_rb[sub_rb['corner'] == corner]

            v2_vals = [
                float(cv2[cv2['level_idx'] == li][metric].values[0])
                if len(cv2[cv2['level_idx'] == li]) else np.nan
                for li in range(n_levels)
            ]
            rb_vals = [
                float(crb[crb['level_idx'] == li][metric].values[0])
                if len(crb[crb['level_idx'] == li]) else np.nan
                for li in range(n_levels)
            ]

            ax.plot(range(n_levels), rb_vals, color=COLOR[corner],
                    marker=MARKER[corner], lw=2.0, ms=6, ls='-')
            ax.plot(range(n_levels), v2_vals, color=COLOR[corner],
                    marker=MARKER[corner], lw=1.5, ms=5, ls='--', alpha=0.7)

        ax.axhline(0, color='#aaa', lw=0.8, ls=':', alpha=0.5)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha='right')
        ax.set_ylim(-1.1, 1.15)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(slabel, fontsize=9, fontweight='bold', pad=4)
        ax.grid(True, alpha=0.2)

    ax_leg = axes_flat[7]
    ax_leg.axis('off')
    handles = []
    for corner in CORNER_ORDER:
        h_rb, = ax_leg.plot([], [], color=COLOR[corner], ls='-', lw=2.0,
                            marker=MARKER[corner], ms=6, label=f'{LABEL[corner]} (Robust)')
        h_v2, = ax_leg.plot([], [], color=COLOR[corner], ls='--', lw=1.5,
                            marker=MARKER[corner], ms=5, alpha=0.7,
                            label=f'{LABEL[corner]} (v2)')
        handles += [h_rb, h_v2]
    ax_leg.legend(handles=handles, fontsize=8.5, loc='center',
                  framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(PLOT_CMP_YIELD, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {PLOT_CMP_YIELD}')


def plot_comparison_full(df_v2, df_rb):
    n_m = len(METRICS)
    n_s = len(SCENARIOS)
    fig, axes = plt.subplots(n_m, n_s, figsize=(4*n_s, 4*n_m), squeeze=False)
    fig.suptitle(
        'CAPQL v2  vs  CAPQL-Robust  —  All Metrics  (balanced corner)\n'
        'Solid = CAPQL-Robust   |   Dashed = CAPQL v2',
        fontsize=13, fontweight='bold')

    for col, (sid, slabel, n_levels, xlabels) in enumerate(SCENARIOS):
        sub_v2 = df_v2[(df_v2['scenario'] == sid) & (df_v2['corner'] == 'balanced')]
        sub_rb = df_rb[(df_rb['scenario'] == sid) & (df_rb['corner'] == 'balanced')]

        for row, metric in enumerate(METRICS):
            ax = axes[row][col]

            rb_vals = [
                float(sub_rb[sub_rb['level_idx'] == li][metric].values[0])
                if len(sub_rb[sub_rb['level_idx'] == li]) else np.nan
                for li in range(n_levels)
            ]
            v2_vals = [
                float(sub_v2[sub_v2['level_idx'] == li][metric].values[0])
                if len(sub_v2[sub_v2['level_idx'] == li]) else np.nan
                for li in range(n_levels)
            ]

            ax.plot(range(n_levels), rb_vals, color='#c0392b', marker='o',
                    lw=2.0, ms=6, ls='-', label='Robust')
            ax.plot(range(n_levels), v2_vals, color='#2980b9', marker='s',
                    lw=1.5, ms=5, ls='--', alpha=0.8, label='v2')

            x = list(range(n_levels))
            fill = [not (np.isnan(a) or np.isnan(b)) for a, b in zip(v2_vals, rb_vals)]
            ax.fill_between(x, v2_vals, rb_vals, where=fill,
                            alpha=0.15, color='#27ae60', label='Δ resilience')

            ax.axhline(0, color='#aaa', lw=0.8, ls=':', alpha=0.5)
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
    plt.savefig(PLOT_CMP_FULL, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {PLOT_CMP_FULL}')


def print_summary(df_v2, df_rb):
    print('\n' + '=' * 72)
    print('  Resilience summary — mean R_yield degradation'
          ' (clean → worst fault level)')
    print(f"  {'Scenario':<22}  {'v2 clean':>10}  {'v2 worst':>10}"
          f"  {'rb clean':>10}  {'rb worst':>10}  {'Δ improvement':>14}")
    print('  ' + '─' * 70)

    for sid, slabel, n_levels, xlabels in SCENARIOS:
        sub_v2 = df_v2[df_v2['scenario'] == sid]
        sub_rb = df_rb[df_rb['scenario'] == sid]

        def mean_r(sub, li):
            rows = sub[sub['level_idx'] == li]['R_yield']
            return float(rows.mean()) if len(rows) else np.nan

        v2_clean = mean_r(sub_v2, 0)
        v2_worst = min((mean_r(sub_v2, li) for li in range(n_levels)),
                       default=np.nan)
        rb_clean = mean_r(sub_rb, 0)
        rb_worst = min((mean_r(sub_rb, li) for li in range(n_levels)),
                       default=np.nan)

        v2_deg      = v2_clean - v2_worst
        rb_deg      = rb_clean - rb_worst
        improvement = v2_deg - rb_deg

        print(
            f"  {sid:<22}  {v2_clean:>+10.3f}  {v2_worst:>+10.3f}"
            f"  {rb_clean:>+10.3f}  {rb_worst:>+10.3f}"
            f"  {improvement:>+14.3f}"
        )
    print('=' * 72 + '\n')


def main():
    print('=' * 70)
    print('13b_replot_comparison.py  —  regenerate plots from saved CSVs')
    print('=' * 70)

    df_rb = load(RB_LOG)
    df_v2 = load(V2_LOG)

    if df_rb is None:
        print('ERROR: robust log missing — run 13_resilience_eval_robust.py first')
        return

    plot_curves(df_rb)

    if df_v2 is not None:
        print_summary(df_v2, df_rb)
        plot_comparison(df_v2, df_rb, metric='R_yield')
        plot_comparison_full(df_v2, df_rb)
    else:
        print('  v2 log not found — only robust curves plotted')

    print('Done.')


if __name__ == '__main__':
    main()
