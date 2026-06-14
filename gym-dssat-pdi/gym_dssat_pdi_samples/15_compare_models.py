"""
15_compare_models.py  —  Full comparison: CAPQL v2 vs CAPQL-Robust MLP.

Loads existing CSVs (no re-running episodes) and generates:
  Table 4.3  — clean corner performance (printed to console)
  Table 4.4  — fault degradation summary (printed to console)
  Fig 4.8    — resilience curves per fault type (Robust only)
  Fig 4.9    — v2 vs Robust degradation under faults (R_yield)
  Fig 4.10   — v2 vs Robust all metrics, balanced corner
  Fig 4.11   — degradation heatmap (v2 vs Robust side by side)

Run inside Docker:
    /opt/gym_dssat_pdi/bin/python3 15_compare_models.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Input files ─────────────────────────────────────────────────────
V2_RESILIENCE   = '/workspace/resilience_v2_log.csv'
RB_RESILIENCE   = '/workspace/resilience_robust_log.csv'
RB_EVAL_CORNERS = '/workspace/capql_robust_eval_results.csv'
V2_EVAL_CORNERS = '/workspace/capql_v2_eval_results.csv'   # may not exist

# ── Output files ─────────────────────────────────────────────────────
FIG_CURVES      = '/workspace/fig_4_8_resilience_curves.png'
FIG_COMPARE     = '/workspace/fig_4_9_comparison_ryield.png'
FIG_FULL        = '/workspace/fig_4_10_comparison_full.png'
FIG_HEATMAP     = '/workspace/fig_4_11_degradation_heatmap.png'

# ── Scenarios (must match resilience eval scripts) ───────────────────
SCENARIOS = [
    ('dropout_soil', 'Soil Moisture\nDropout',      5, ['0%','5%','10%','20%','50%']),
    ('dropout_all',  'All Sensors\nDropout',         5, ['0%','5%','10%','20%','50%']),
    ('packet_loss',  'Packet Loss\n(Type 1)',         5, ['0%','5%','10%','20%','50%']),
    ('stuck_soil',   'Stuck Sensor\n(Type 2)',        6, ['No fault','Day 120','Day 90','Day 60','Day 30','Day 10']),
    ('bias_soil',    'Soil Bias\n(Type 2)',           6, ['0','-0.20','-0.10','+0.10','+0.20','+0.30']),
    ('actuator_N',   'N Spreader\nEfficiency',        6, ['100%','80%','60%','40%','20%','0%']),
    ('actuator_W',   'Irrigation Pump\nEfficiency',   6, ['100%','80%','60%','40%','20%','0%']),
]

METRICS      = ['R_yield', 'R_ane', 'R_water_eff']
METRIC_LABEL = {'R_yield': 'R_yield', 'R_ane': 'R_ane', 'R_water_eff': 'R_water'}

CORNERS = ['yield', 'n_eff', 'water', 'balanced']
COLOR   = {'yield': '#27ae60', 'n_eff': '#e67e22',
           'water': '#2980b9', 'balanced': '#8e44ad'}
LABEL   = {'yield': 'Yield', 'n_eff': 'N-Eff',
           'water': 'Water', 'balanced': 'Balanced'}
MARKER  = {'yield': 'o', 'n_eff': 's', 'water': '^', 'balanced': 'D'}


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def load(path, label):
    if not os.path.exists(path):
        print(f'  ✗  Not found: {path}  [{label}]')
        return None
    df = pd.read_csv(path)
    print(f'  ✓  {label}: {len(df)} rows from {path}')
    return df


def _get(df, scenario, corner, level_idx, metric):
    """Safe single-value lookup from resilience log."""
    sub = df[(df['scenario'] == scenario) &
             (df['corner']   == corner)   &
             (df['level_idx']== level_idx)]
    if len(sub) == 0:
        return np.nan
    return float(sub[metric].mean())


def scenario_vals(df, sid, corner, n_levels, metric):
    return [_get(df, sid, corner, li, metric) for li in range(n_levels)]


# ══════════════════════════════════════════════════════════════════════
# Table 4.3 — clean corner performance
# ══════════════════════════════════════════════════════════════════════
def print_corner_table(df_rb_corners, df_v2_corners):
    print('\n' + '=' * 80)
    print('  Table 4.3 — Clean corner evaluation (mean over 5 episodes)')
    header = (f"  {'Corner':<12} {'Model':<16} {'R_yield':>9} "
              f"{'R_ane':>9} {'R_water':>9} {'Yield kg/ha':>12} "
              f"{'N kg/ha':>9} {'Water mm':>10}")
    print(header)
    print('  ' + '─' * 78)

    for corner in CORNERS:
        for df, label in [(df_v2_corners, 'CAPQL v2'),
                          (df_rb_corners,  'CAPQL-Robust')]:
            if df is None:
                continue
            sub = df[df['corner'] == corner]
            if len(sub) == 0:
                continue
            ry  = sub['R_yield'].mean()
            ra  = sub['R_ane'].mean()
            rw  = sub['R_water_eff'].mean()
            yld = sub['yield_kg_ha'].mean()
            n   = sub['total_N_kg_ha'].mean()
            w   = sub['total_W_mm'].mean()
            print(f"  {corner:<12} {label:<16} "
                  f"{ry:>+9.3f} {ra:>+9.3f} {rw:>+9.3f} "
                  f"{yld:>12.0f} {n:>9.0f} {w:>10.0f}")
        print()
    print('=' * 80 + '\n')


# ══════════════════════════════════════════════════════════════════════
# Table 4.4 — fault degradation summary
# ══════════════════════════════════════════════════════════════════════
def print_degradation_table(df_v2, df_rb):
    print('\n' + '=' * 78)
    print('  Table 4.4 — Fault degradation (R_yield): clean → worst fault level')
    print(f"  {'Scenario':<22} {'v2 clean':>10} {'v2 worst':>10} "
          f"{'v2 drop':>9} {'Rb clean':>10} {'Rb worst':>10} {'Rb drop':>9} "
          f"{'Improvement':>12}")
    print('  ' + '─' * 76)

    for sid, slabel, n_levels, _ in SCENARIOS:
        # average over all 4 corners for the summary
        v2_clean = np.nanmean([_get(df_v2, sid, c, 0,            'R_yield') for c in CORNERS])
        v2_worst = np.nanmin( [_get(df_v2, sid, c, li, 'R_yield')
                                for c in CORNERS for li in range(n_levels)])
        rb_clean = np.nanmean([_get(df_rb, sid, c, 0,            'R_yield') for c in CORNERS])
        rb_worst = np.nanmin( [_get(df_rb, sid, c, li, 'R_yield')
                                for c in CORNERS for li in range(n_levels)])

        v2_drop  = v2_clean - v2_worst
        rb_drop  = rb_clean - rb_worst
        impr     = v2_drop  - rb_drop

        slabel_short = slabel.replace('\n', ' ')
        print(f"  {slabel_short:<22} "
              f"{v2_clean:>+10.3f} {v2_worst:>+10.3f} {v2_drop:>+9.3f} "
              f"{rb_clean:>+10.3f} {rb_worst:>+10.3f} {rb_drop:>+9.3f} "
              f"{impr:>+12.3f}")
    print('=' * 78 + '\n')


# ══════════════════════════════════════════════════════════════════════
# Fig 4.8 — resilience curves (Robust only, 7 scenarios × 3 metrics)
# ══════════════════════════════════════════════════════════════════════
def fig_resilience_curves(df_rb):
    n_s = len(SCENARIOS)
    n_m = len(METRICS)
    fig, axes = plt.subplots(n_s, n_m, figsize=(5 * n_m, 3.2 * n_s), squeeze=False)
    fig.suptitle('Figure 4.8 — CAPQL-Robust Resilience under Realistic Faults',
                 fontsize=13, fontweight='bold')

    for row, (sid, slabel, n_levels, xlabels) in enumerate(SCENARIOS):
        for col, metric in enumerate(METRICS):
            ax = axes[row][col]
            for corner in CORNERS:
                vals = scenario_vals(df_rb, sid, corner, n_levels, metric)
                ax.plot(range(n_levels), vals, color=COLOR[corner],
                        marker=MARKER[corner], lw=1.8, ms=5, label=LABEL[corner])
            ax.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, fontsize=7.5, rotation=30, ha='right')
            ax.set_ylim(-1.15, 1.20)
            ax.set_ylabel(METRIC_LABEL[metric], fontsize=9)
            ax.set_title(f'{slabel.replace(chr(10), " ")} — {METRIC_LABEL[metric]}',
                         fontsize=8.5, fontweight='bold')
            ax.grid(True, alpha=0.18)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc='lower left', framealpha=0.85)

    plt.tight_layout()
    plt.savefig(FIG_CURVES, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {FIG_CURVES}')


# ══════════════════════════════════════════════════════════════════════
# Fig 4.9 — v2 vs Robust, R_yield per scenario
# ══════════════════════════════════════════════════════════════════════
def fig_comparison_ryield(df_v2, df_rb):
    n_s = len(SCENARIOS)
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    fig.suptitle(
        'Figure 4.9 — CAPQL v2 vs CAPQL-Robust  |  R_yield under Faults\n'
        'Solid = Robust  |  Dashed = v2  (per preference corner)',
        fontsize=12, fontweight='bold')

    for idx, (sid, slabel, n_levels, xlabels) in enumerate(SCENARIOS):
        ax = axes_flat[idx]
        for corner in CORNERS:
            v2_vals = scenario_vals(df_v2, sid, corner, n_levels, 'R_yield')
            rb_vals = scenario_vals(df_rb, sid, corner, n_levels, 'R_yield')
            ax.plot(range(n_levels), rb_vals, color=COLOR[corner],
                    marker=MARKER[corner], lw=2.0, ms=5, ls='-')
            ax.plot(range(n_levels), v2_vals, color=COLOR[corner],
                    marker=MARKER[corner], lw=1.3, ms=4, ls='--', alpha=0.65)

        ax.axhline(0, color='#aaa', lw=0.8, ls=':', alpha=0.5)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=7.5, rotation=30, ha='right')
        ax.set_ylim(-1.15, 1.20)
        ax.set_ylabel('R_yield', fontsize=9)
        ax.set_title(slabel.replace('\n', ' '), fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.18)

    # Legend panel
    ax_leg = axes_flat[7]
    ax_leg.axis('off')
    handles = []
    for corner in CORNERS:
        h1, = ax_leg.plot([], [], color=COLOR[corner], ls='-',  lw=2.0,
                          marker=MARKER[corner], ms=5, label=f'{LABEL[corner]} — Robust')
        h2, = ax_leg.plot([], [], color=COLOR[corner], ls='--', lw=1.3,
                          marker=MARKER[corner], ms=4, alpha=0.65,
                          label=f'{LABEL[corner]} — v2')
        handles += [h1, h2]
    ax_leg.legend(handles=handles, fontsize=8, loc='center',
                  framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(FIG_COMPARE, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {FIG_COMPARE}')


# ══════════════════════════════════════════════════════════════════════
# Fig 4.10 — all metrics, balanced corner, v2 vs Robust
# ══════════════════════════════════════════════════════════════════════
def fig_comparison_full(df_v2, df_rb):
    n_m = len(METRICS)
    n_s = len(SCENARIOS)
    fig, axes = plt.subplots(n_m, n_s, figsize=(4 * n_s, 3.8 * n_m), squeeze=False)
    fig.suptitle(
        'Figure 4.10 — CAPQL v2 vs CAPQL-Robust  |  All Metrics  (Balanced corner)\n'
        'Solid red = Robust  |  Dashed blue = v2  |  Green fill = Δ resilience gain',
        fontsize=12, fontweight='bold')

    for col, (sid, slabel, n_levels, xlabels) in enumerate(SCENARIOS):
        sub_v2 = df_v2[(df_v2['scenario'] == sid) & (df_v2['corner'] == 'balanced')]
        sub_rb = df_rb[(df_rb['scenario'] == sid) & (df_rb['corner'] == 'balanced')]

        for row, metric in enumerate(METRICS):
            ax = axes[row][col]
            v2_vals = [_get(df_v2, sid, 'balanced', li, metric) for li in range(n_levels)]
            rb_vals = [_get(df_rb, sid, 'balanced', li, metric) for li in range(n_levels)]
            x = list(range(n_levels))

            ax.plot(x, rb_vals, color='#c0392b', marker='o', lw=2.0, ms=5,
                    ls='-',  label='Robust')
            ax.plot(x, v2_vals, color='#2980b9', marker='s', lw=1.5, ms=4,
                    ls='--', alpha=0.8, label='v2')

            # Fill where Robust is better
            fill_mask = [not (np.isnan(a) or np.isnan(b))
                         for a, b in zip(v2_vals, rb_vals)]
            if any(fill_mask):
                ax.fill_between(x, v2_vals, rb_vals, where=fill_mask,
                                alpha=0.15, color='#27ae60', label='Δ gain')

            ax.axhline(0, color='#aaa', lw=0.7, ls=':', alpha=0.5)
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, fontsize=7, rotation=30, ha='right')
            ax.set_ylim(-1.15, 1.20)
            ax.set_ylabel(METRIC_LABEL[metric], fontsize=9)
            if row == 0:
                ax.set_title(slabel.replace('\n', ' '), fontsize=8,
                             fontweight='bold')
            ax.grid(True, alpha=0.18)
            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, loc='lower left', framealpha=0.85)

    plt.tight_layout()
    plt.savefig(FIG_FULL, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {FIG_FULL}')


# ══════════════════════════════════════════════════════════════════════
# Fig 4.11 — degradation heatmap (v2 vs Robust)
# ══════════════════════════════════════════════════════════════════════
def fig_heatmap(df_v2, df_rb):
    """
    Heatmap: rows = scenarios, cols = severity levels.
    Color = R_yield at that level (averaged over 4 corners).
    Two panels: v2 (left) | Robust (right).
    """
    max_levels = max(n for _, _, n, _ in SCENARIOS)

    def build_matrix(df):
        mat = np.full((len(SCENARIOS), max_levels), np.nan)
        for r, (sid, _, n_levels, _) in enumerate(SCENARIOS):
            for li in range(n_levels):
                vals = [_get(df, sid, c, li, 'R_yield') for c in CORNERS]
                mat[r, li] = np.nanmean(vals)
        return mat

    mat_v2 = build_matrix(df_v2)
    mat_rb = build_matrix(df_rb)

    cmap = LinearSegmentedColormap.from_list(
        'ry', ['#c0392b', '#f39c12', '#f1c40f', '#2ecc71'], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    row_labels = [s.replace('\n', ' ') for _, s, _, _ in SCENARIOS]

    for ax, mat, title in [
        (axes[0], mat_v2, 'CAPQL v2'),
        (axes[1], mat_rb, 'CAPQL-Robust'),
    ]:
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=-1.0, vmax=1.0)
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel('Fault severity level →', fontsize=9)
        ax.set_title(f'{title}\n(R_yield, mean over 4 corners)',
                     fontsize=10, fontweight='bold')
        ax.set_xticks(range(max_levels))
        ax.set_xticklabels([f'L{i}' for i in range(max_levels)], fontsize=8)

        # Add text values
        for r in range(len(SCENARIOS)):
            n_lev = SCENARIOS[r][2]
            for li in range(n_lev):
                v = mat[r, li]
                if not np.isnan(v):
                    ax.text(li, r, f'{v:+.2f}', ha='center', va='center',
                            fontsize=7, color='white' if abs(v) > 0.5 else 'black',
                            fontweight='bold')

    plt.colorbar(im, ax=axes, label='R_yield (−1 = worst, +1 = best)',
                 fraction=0.02, pad=0.04)
    fig.suptitle('Figure 4.11 — Fault Degradation Heatmap\n'
                 '(darker green = more resilient; red = severe degradation)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_HEATMAP, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {FIG_HEATMAP}')


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('15_compare_models.py  —  CAPQL v2 vs CAPQL-Robust comparison')
    print('=' * 70)

    df_v2 = load(V2_RESILIENCE,   'CAPQL v2 resilience')
    df_rb = load(RB_RESILIENCE,   'CAPQL-Robust resilience')
    df_rb_corners = load(RB_EVAL_CORNERS, 'CAPQL-Robust corner eval')
    df_v2_corners = load(V2_EVAL_CORNERS, 'CAPQL v2 corner eval')

    if df_v2 is None or df_rb is None:
        print('\n  ERROR: Need both resilience CSVs. '
              'Run 11_resilience_eval.py and 13_resilience_eval_robust.py first.')
        return

    # Tables
    if df_rb_corners is not None or df_v2_corners is not None:
        print_corner_table(df_rb_corners, df_v2_corners)

    print_degradation_table(df_v2, df_rb)

    # Figures
    print('\n  Generating figures...')
    fig_resilience_curves(df_rb)
    fig_comparison_ryield(df_v2, df_rb)
    fig_comparison_full(df_v2, df_rb)
    fig_heatmap(df_v2, df_rb)

    print('\n  All figures saved to /workspace/')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
