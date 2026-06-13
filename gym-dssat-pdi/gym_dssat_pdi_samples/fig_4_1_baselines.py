"""
fig_4_1_baselines.py  —  Figure 4.1 + Table 4.1 for thesis Chapter 4.

Generates:
  /workspace/fig_4_1_baselines.png   — grouped bar chart (4 metrics × 4 baselines)
  /workspace/fig_4_1b_radar.png      — radar chart (normalised profile per baseline)

Run inside Docker:
    /opt/gym_dssat_pdi/bin/python3 fig_4_1_baselines.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths ────────────────────────────────────────────────────────────
CSV_IN   = '/workspace/baselines_results.csv'
OUT_BAR  = '/workspace/fig_4_1_baselines.png'
OUT_RAD  = '/workspace/fig_4_1b_radar.png'

# ── style ────────────────────────────────────────────────────────────
BASELINE_ORDER  = ['Random', 'Fixed Schedule', 'Stage-based', 'FAO-56']
BASELINE_COLORS = ['#95a5a6', '#3498db', '#2ecc71', '#e67e22']
BASELINE_LABELS = ['Random', 'Fixed\nSchedule', 'Stage-\nbased', 'FAO-56']

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'legend.fontsize':  9,
    'figure.dpi':       150,
})

# ═══════════════════════════════════════════════════════════════════════
# Load & summarise
# ═══════════════════════════════════════════════════════════════════════
def load_and_summarise(path):
    df = pd.read_csv(path)

    # Normalise baseline names (strip whitespace, fix casing)
    df['name'] = df['name'].str.strip()

    metrics = ['yield_kg_ha', 'total_N_kg_ha', 'total_W_mm',
               'R_yield', 'R_ane', 'R_hiad', 'R_seasonal', 'cum_reward']

    summary = {}
    for name in df['name'].unique():
        sub = df[df['name'] == name]
        summary[name] = {
            m: {'mean': sub[m].mean(), 'std': sub[m].std()}
            for m in metrics if m in sub.columns
        }
    return df, summary


# ═══════════════════════════════════════════════════════════════════════
# Table 4.1 — printed to console
# ═══════════════════════════════════════════════════════════════════════
def print_table(summary):
    metrics = ['yield_kg_ha', 'total_N_kg_ha', 'total_W_mm', 'R_yield', 'R_ane', 'R_hiad']
    header  = f"{'Baseline':<18} {'Yield (kg/ha)':>14} {'N (kg/ha)':>11} {'Water (mm)':>11} {'R_yield':>9} {'R_ane':>9} {'R_hiad':>9}"
    sep     = '─' * len(header)

    print('\nTable 4.1 — Rule-based baseline summary (mean ± std, 20 episodes each)')
    print(sep)
    print(header)
    print(sep)
    for name in BASELINE_ORDER:
        if name not in summary:
            continue
        s = summary[name]
        def v(m):
            return f"{s[m]['mean']:+.1f}±{s[m]['std']:.1f}" if m in s else '—'
        def vf(m, fmt='.3f'):
            return f"{s[m]['mean']:+{fmt}}±{s[m]['std']:.3f}" if m in s else '—'
        print(
            f"{name:<18} "
            f"{s['yield_kg_ha']['mean']:>8.0f}±{s['yield_kg_ha']['std']:.0f}   "
            f"{s['total_N_kg_ha']['mean']:>6.0f}±{s['total_N_kg_ha']['std']:.0f}   "
            f"{s['total_W_mm']['mean']:>6.0f}±{s['total_W_mm']['std']:.0f}   "
            f"{s['R_yield']['mean']:>+7.3f}   "
            f"{s['R_ane']['mean']:>+7.3f}   "
            f"{s['R_hiad']['mean']:>+7.3f}"
        )
    print(sep + '\n')


# ═══════════════════════════════════════════════════════════════════════
# Figure 4.1 — grouped bar chart (2×2)
# ═══════════════════════════════════════════════════════════════════════
def plot_bar(summary, out_path):
    fig = plt.figure(figsize=(12, 8))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    panels = [
        ('yield_kg_ha',   'Grain Yield (kg ha⁻¹)',         None,   'a'),
        ('total_N_kg_ha', 'Total N Applied (kg ha⁻¹)',      None,   'b'),
        ('total_W_mm',    'Total Irrigation (mm)',           None,   'c'),
        ('R_ane',         'N-Efficiency Score (R_ane)',      [-1,1], 'd'),
    ]

    names_present = [n for n in BASELINE_ORDER if n in summary]
    x   = np.arange(len(names_present))
    w   = 0.55

    for idx, (metric, ylabel, ylim, panel_letter) in enumerate(panels):
        ax   = fig.add_subplot(gs[idx // 2, idx % 2])
        means = [summary[n][metric]['mean'] for n in names_present]
        stds  = [summary[n][metric]['std']  for n in names_present]
        colors = [BASELINE_COLORS[BASELINE_ORDER.index(n)] for n in names_present]

        bars = ax.bar(x, means, width=w, yerr=stds, capsize=4,
                      color=colors, edgecolor='white', linewidth=0.8,
                      error_kw=dict(elinewidth=1.2, ecolor='#555'))

        # Value labels on top of each bar
        for bar, mean in zip(bars, means):
            va  = 'bottom' if mean >= 0 else 'top'
            off = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0]) if ax.get_ylim()[1] != ax.get_ylim()[0] else 50
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + (abs(off) if mean >= 0 else -abs(off)),
                    f'{mean:.0f}' if abs(mean) > 10 else f'{mean:.2f}',
                    ha='center', va=va, fontsize=8.5, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(
            [BASELINE_LABELS[BASELINE_ORDER.index(n)] for n in names_present],
            fontsize=9
        )
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.6)
        ax.grid(axis='y', alpha=0.25, lw=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'({panel_letter})', loc='left', fontsize=10, fontweight='bold')

        # Add a reference line for yield panel (Stage-based anchor ~7620 kg/ha)
        if metric == 'yield_kg_ha':
            ax.axhline(7620, color='#c0392b', lw=1.2, ls=':', alpha=0.7,
                       label='Stage-based anchor\n(7 620 kg ha⁻¹)')
            ax.legend(fontsize=8, loc='upper left', framealpha=0.85)

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=BASELINE_COLORS[i], label=BASELINE_ORDER[i])
        for i in range(len(BASELINE_ORDER))
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=9.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        'Figure 4.1 — Rule-Based Baseline Performance\n'
        '(mean ± std over 20 episodes)',
        fontsize=12, fontweight='bold', y=0.98
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {out_path}')


# ═══════════════════════════════════════════════════════════════════════
# Figure 4.1b — radar chart
# ═══════════════════════════════════════════════════════════════════════
def plot_radar(summary, out_path):
    """
    Normalised spider chart: 5 axes, each baseline = one polygon.
    All metrics normalised to [0,1] relative to the best baseline on that axis.
    """
    axes_cfg = [
        ('yield_kg_ha',   'Yield',     True),   # higher = better
        ('total_N_kg_ha', '−N applied', False),  # lower = better
        ('total_W_mm',    '−Water',    False),   # lower = better
        ('R_ane',         'N-Efficiency', True), # higher = better
        ('R_hiad',        'Harv.Index', True),   # higher = better
    ]

    names = [n for n in BASELINE_ORDER if n in summary]
    n_ax  = len(axes_cfg)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    # Collect raw values
    raw = {}
    for name in names:
        raw[name] = [summary[name][m]['mean'] for m, _, _ in axes_cfg]

    # Normalise: map each axis to [0,1] (0=worst, 1=best among baselines)
    def normalise(vals, higher_better):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [0.5] * len(vals)
        normed = [(v - lo) / (hi - lo) for v in vals]
        return normed if higher_better else [1 - n for n in normed]

    norm_per_axis = []
    for ax_idx, (_, _, hb) in enumerate(axes_cfg):
        col = [raw[n][ax_idx] for n in names]
        norm_per_axis.append(normalise(col, hb))

    norm_per_baseline = {}
    for i, name in enumerate(names):
        norm_per_baseline[name] = [norm_per_axis[ax][i] for ax in range(n_ax)]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for name in names:
        vals   = norm_per_baseline[name] + norm_per_baseline[name][:1]
        color  = BASELINE_COLORS[BASELINE_ORDER.index(name)]
        ax.plot(angles, vals, color=color, lw=2.0, label=name)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [lbl for _, lbl, _ in axes_cfg], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7.5)
    ax.grid(color='#ccc', lw=0.8)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=9.5, framealpha=0.9)
    ax.set_title(
        'Figure 4.1b — Normalised Performance Profile\n(1.0 = best baseline on that axis)',
        fontsize=10, fontweight='bold', pad=20
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {out_path}')


# ═══════════════════════════════════════════════════════════════════════
def main():
    print('Generating Figure 4.1 — Baseline Reference...')
    df, summary = load_and_summarise(CSV_IN)
    print(f'  Loaded {len(df)} rows, baselines: {list(summary.keys())}')
    print_table(summary)
    plot_bar(summary, OUT_BAR)
    plot_radar(summary, OUT_RAD)
    print('Done.')

if __name__ == '__main__':
    main()
