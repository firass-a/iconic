"""
fig_4_2_ppo.py  —  Figures 4.2 & 4.3 for thesis Chapter 4, Section 4.2.

Generates:
  /workspace/fig_4_2_training_curves.png   — 6-panel PPO training convergence
  /workspace/fig_4_2b_ppo_losses.png       — PPO loss curves (policy/value/entropy)
  /workspace/fig_4_3_variants_bar.png      — grouped bar: base vs water vs N-min PPO
  /workspace/fig_4_3b_tradeoff.png         — yield vs N vs water trade-off scatter

Run inside Docker:
    /opt/gym_dssat_pdi/bin/python3 fig_4_2_ppo.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths ─────────────────────────────────────────────────────────────
EP_LOG_BASE  = '/workspace/episode_log.csv'
EP_LOG_WATER = '/workspace/episode_log_water.csv'
EP_LOG_N     = '/workspace/episode_log_nitrogen_v3.csv'

EVAL_BASE    = '/workspace/eval_results.csv'
EVAL_WATER   = '/workspace/eval_results_water.csv'
EVAL_N       = '/workspace/eval_results_nitrogen.csv'
EVAL_N_V3    = '/workspace/eval_results_nitrogen_v3.csv'
METRICS_BASE = '/workspace/training_metrics.csv'

BASELINE_CSV = '/workspace/baselines_results.csv'

OUT_CURVES   = '/workspace/fig_4_2_training_curves.png'
OUT_LOSSES   = '/workspace/fig_4_2b_ppo_losses.png'
OUT_BAR      = '/workspace/fig_4_3_variants_bar.png'
OUT_TRADE    = '/workspace/fig_4_3b_tradeoff.png'

ROLL = 80   # rolling mean window (episodes)

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi':     150,
})


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════
def roll(series, w=ROLL):
    return series.rolling(w, min_periods=max(1, w // 4)).mean()

def load(path, required_cols=None):
    try:
        df = pd.read_csv(path)
        if required_cols:
            for c in required_cols:
                if c not in df.columns:
                    return None
        return df
    except Exception:
        return None

def mean_row(eval_df):
    """Return the MEAN summary row or compute it."""
    if eval_df is None:
        return {}
    mask = eval_df['episode'].astype(str).str.upper() == 'MEAN'
    if mask.any():
        row = eval_df[mask].iloc[0]
    else:
        row = eval_df.mean(numeric_only=True)
    return row.to_dict()


# ══════════════════════════════════════════════════════════════════════
# Figure 4.2 — Training convergence (6 panels)
# ══════════════════════════════════════════════════════════════════════
def plot_training_curves(df, out_path):
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

    COLOR_ROLL = '#2980b9'
    COLOR_RAW  = '#bdc3c7'

    panels = [
        ('cum_reward',    'Cumulative Episode Reward',  '(a)', None),
        ('R_seasonal',    'Seasonal Reward  R_seasonal','(b)', [-1.2, 1.2]),
        ('R_yield',       'Yield Reward  R_yield',      '(c)', [-1.2, 1.2]),
        ('R_ane',         'N-Efficiency Reward  R_ane', '(d)', [-1.2, 1.2]),
        ('yield_kg_ha',   'Grain Yield  (kg ha⁻¹)',     '(e)', None),
        ('total_N_kg_ha', 'Total N Applied  (kg ha⁻¹)', '(f)', None),
    ]

    for idx, (col, ylabel, letter, ylim) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        if col not in df.columns:
            ax.text(0.5, 0.5, f'{col}\nnot found', ha='center',
                    transform=ax.transAxes, fontsize=9, color='red')
            continue

        x = (df['timestep'] / 1e6).to_numpy()

        # raw (faint)
        ax.plot(x, df[col].to_numpy(), color=COLOR_RAW, lw=0.5, alpha=0.4)
        # rolling mean
        ax.plot(x, roll(df[col]).to_numpy(), color=COLOR_ROLL, lw=1.8,
                label=f'Rolling mean (n={ROLL})')

        # overlay total_W_mm on the N panel
        if col == 'total_N_kg_ha' and 'total_W_mm' in df.columns:
            ax2 = ax.twinx()
            ax2.plot(x, roll(df['total_W_mm']).to_numpy(),
                     color='#27ae60', lw=1.8, ls='--', label='Water (mm)')
            ax2.set_ylabel('Irrigation (mm)', color='#27ae60', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='#27ae60')
            ax2.legend(loc='upper right', fontsize=8)

        ax.set_xlabel('Training Steps (×10⁶)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(ylim)
            ax.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.5)
        ax.grid(alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(letter, loc='left', fontsize=10, fontweight='bold')

        if col == 'cum_reward':
            ax.legend(fontsize=8, loc='lower right')

        # annotate final rolling mean value
        final_val = roll(df[col]).iloc[-1]
        ax.annotate(f'{final_val:.2f}',
                    xy=(x[-1], final_val),
                    xytext=(-30, 8), textcoords='offset points',
                    fontsize=8, color=COLOR_ROLL,
                    arrowprops=dict(arrowstyle='->', color=COLOR_ROLL, lw=0.8))

    fig.suptitle(
        'Figure 4.2 — PPO Training Convergence (1 M steps)\n'
        f'Raw signal (grey) and {ROLL}-episode rolling mean (blue)',
        fontsize=12, fontweight='bold', y=0.99
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {out_path}')


# ══════════════════════════════════════════════════════════════════════
# Figure 4.2b — PPO loss curves
# ══════════════════════════════════════════════════════════════════════
def plot_losses(metrics_path, out_path):
    df = load(metrics_path)
    if df is None:
        print(f'  ⚠  Metrics log not found: {metrics_path}')
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Figure 4.2b — PPO Loss Curves',
                 fontsize=12, fontweight='bold')

    loss_panels = [
        ('policy_loss',  'Policy Loss',  '#e74c3c'),
        ('value_loss',   'Value Loss',   '#3498db'),
        ('entropy_loss', 'Entropy Loss', '#2ecc71'),
    ]

    # x axis — use update number or timesteps
    x_col = 'timestep' if 'timestep' in df.columns else df.index

    for ax, (col, title, color) in zip(axes, loss_panels):
        if col not in df.columns:
            ax.text(0.5, 0.5, f'{col}\nnot found', ha='center',
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(title)
            continue
        x = (df[x_col] / 1e6).to_numpy() if 'timestep' in df.columns else np.arange(len(df))
        ax.plot(x, df[col].to_numpy(), color=color, lw=1.2, alpha=0.9)
        ax.plot(x, df[col].rolling(10, min_periods=1).mean().to_numpy(),
                color='black', lw=1.8, ls='--', label='Rolling mean')
        ax.set_xlabel('Steps (×10⁶)' if 'timestep' in df.columns else 'Update')
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {out_path}')


# ══════════════════════════════════════════════════════════════════════
# Figure 4.3 — PPO variants grouped bar
# ══════════════════════════════════════════════════════════════════════
def plot_variants_bar(baselines_df, out_path):
    # Load eval summaries
    base_m  = mean_row(load(EVAL_BASE))
    water_m = mean_row(load(EVAL_WATER))
    n_m     = mean_row(load(EVAL_N_V3))   # use v3 (better version)

    # FAO-56 from baselines
    fao = baselines_df[baselines_df['name'] == 'FAO-56'].mean(numeric_only=True)

    agents = ['FAO-56\n(rule)', 'Base PPO\n(balanced)', 'Water-min\nPPO', 'N-min\nPPO (v3)']
    colors = ['#e67e22', '#2980b9', '#27ae60', '#9b59b6']

    metrics = [
        ('yield_kg_ha',   'Grain Yield (kg ha⁻¹)',       None),
        ('total_N_kg_ha', 'Total N Applied (kg ha⁻¹)',    None),
        ('total_W_mm',    'Total Irrigation (mm)',         None),
        ('R_ane',         'N-Efficiency Score (R_ane)',   [-1, 1]),
    ]

    def v(row, key, fallback=0):
        if isinstance(row, dict):
            return float(row.get(key, fallback))
        try:
            return float(getattr(row, key, fallback))
        except Exception:
            return fallback

    data = {
        'yield_kg_ha':   [v(fao,'yield_kg_ha'), v(base_m,'yield_kg_ha'),
                          v(water_m,'yield_kg_ha'), v(n_m,'yield_kg_ha')],
        'total_N_kg_ha': [v(fao,'total_N_kg_ha'), v(base_m,'total_N_kg_ha'),
                          v(water_m,'total_N_kg_ha'), v(n_m,'total_N_kg_ha')],
        'total_W_mm':    [v(fao,'total_W_mm'), v(base_m,'total_W_mm'),
                          v(water_m,'total_W_mm'), v(n_m,'total_W_mm')],
        'R_ane':         [v(fao,'R_ane'), v(base_m,'R_ane'),
                          v(water_m,'R_ane'), v(n_m,'R_ane')],
    }

    fig = plt.figure(figsize=(13, 8))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    for idx, (key, ylabel, ylim) in enumerate(metrics):
        ax  = fig.add_subplot(gs[idx // 2, idx % 2])
        x   = np.arange(len(agents))
        vals = data[key]

        bars = ax.bar(x, vals, width=0.55, color=colors,
                      edgecolor='white', linewidth=0.8)

        for bar, val in zip(bars, vals):
            va  = 'bottom' if val >= 0 else 'top'
            offset = max(abs(max(vals)) * 0.02, 5)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (offset if val >= 0 else -offset),
                    f'{val:.0f}' if abs(val) > 5 else f'{val:.2f}',
                    ha='center', va=va, fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.axhline(0, color='#aaa', lw=0.8, ls='--', alpha=0.6)
        ax.grid(axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        panel_letter = ['(a)', '(b)', '(c)', '(d)'][idx]
        ax.set_title(panel_letter, loc='left', fontsize=10, fontweight='bold')

        # Highlight improvement vs FAO-56 on yield panel
        if key == 'yield_kg_ha':
            ax.axhline(vals[0], color='#e67e22', lw=1.2, ls=':',
                       alpha=0.7, label='FAO-56 reference')
            ax.legend(fontsize=8)

    legend_patches = [mpatches.Patch(color=c, label=a.replace('\n', ' '))
                      for c, a in zip(colors, agents)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=9.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        'Figure 4.3 — PPO Variants: Evaluation Performance Comparison\n'
        '(mean over 4 evaluation episodes)',
        fontsize=12, fontweight='bold', y=0.99
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {out_path}')


# ══════════════════════════════════════════════════════════════════════
# Figure 4.3b — Yield vs N trade-off scatter
# ══════════════════════════════════════════════════════════════════════
def plot_tradeoff(baselines_df, out_path):
    base_m  = mean_row(load(EVAL_BASE))
    water_m = mean_row(load(EVAL_WATER))
    n_m     = mean_row(load(EVAL_N_V3))
    fao     = baselines_df[baselines_df['name'] == 'FAO-56'].mean(numeric_only=True)
    stg     = baselines_df[baselines_df['name'] == 'Stage-based'].mean(numeric_only=True)

    def v(row, key):
        if isinstance(row, dict):
            return float(row.get(key, 0))
        try:
            return float(getattr(row, key, 0))
        except Exception:
            return 0

    points = [
        ('FAO-56',          v(fao,   'yield_kg_ha'), v(fao,   'total_N_kg_ha'), v(fao,   'total_W_mm'), '#e67e22', 's', 120),
        ('Stage-based',     v(stg,   'yield_kg_ha'), v(stg,   'total_N_kg_ha'), v(stg,   'total_W_mm'), '#95a5a6', 's', 120),
        ('Base PPO',        v(base_m,'yield_kg_ha'), v(base_m,'total_N_kg_ha'), v(base_m,'total_W_mm'), '#2980b9', 'o', 160),
        ('Water-min PPO',   v(water_m,'yield_kg_ha'),v(water_m,'total_N_kg_ha'),v(water_m,'total_W_mm'),'#27ae60', '^', 160),
        ('N-min PPO (v3)',  v(n_m,   'yield_kg_ha'), v(n_m,   'total_N_kg_ha'), v(n_m,   'total_W_mm'), '#9b59b6', 'D', 160),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Figure 4.3b — Agricultural Trade-off Analysis',
                 fontsize=12, fontweight='bold')

    # Left: Yield vs N
    ax = axes[0]
    for label, y, n, w, color, marker, ms in points:
        ax.scatter(n, y, color=color, marker=marker, s=ms,
                   zorder=5, label=label, edgecolors='white', linewidth=0.8)
        ax.annotate(label, (n, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=8.5)
    ax.set_xlabel('Total N Applied (kg ha⁻¹)', fontsize=10)
    ax.set_ylabel('Grain Yield (kg ha⁻¹)', fontsize=10)
    ax.set_title('(a)  Yield  vs  N Application', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: Yield vs Water
    ax = axes[1]
    for label, y, n, w, color, marker, ms in points:
        ax.scatter(w, y, color=color, marker=marker, s=ms,
                   zorder=5, label=label, edgecolors='white', linewidth=0.8)
        ax.annotate(label, (w, y), textcoords='offset points',
                    xytext=(6, 4), fontsize=8.5)
    ax.set_xlabel('Total Irrigation (mm)', fontsize=10)
    ax.set_ylabel('Grain Yield (kg ha⁻¹)', fontsize=10)
    ax.set_title('(b)  Yield  vs  Irrigation', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles = [plt.scatter([], [], color=c, marker=m, s=100,
                           label=lbl, edgecolors='white')
               for lbl, _, _, _, c, m, _ in points]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  {out_path}')


# ══════════════════════════════════════════════════════════════════════
# Console summary table
# ══════════════════════════════════════════════════════════════════════
def print_summary():
    base_m  = mean_row(load(EVAL_BASE))
    water_m = mean_row(load(EVAL_WATER))
    n_v1_m  = mean_row(load(EVAL_N))
    n_v3_m  = mean_row(load(EVAL_N_V3))

    def v(row, key, fmt='.1f'):
        val = row.get(key, float('nan')) if isinstance(row, dict) else float('nan')
        return f'{val:{fmt}}'

    rows = [
        ('Base PPO',       base_m),
        ('Water-min PPO',  water_m),
        ('N-min PPO (v1)', n_v1_m),
        ('N-min PPO (v3)', n_v3_m),
    ]
    sep = '─' * 90
    print('\nTable 4.2 — PPO Variants Evaluation Summary (mean, 4 episodes)')
    print(sep)
    print(f"  {'Model':<20} {'Yield (kg/ha)':>14} {'N (kg/ha)':>11} {'Water (mm)':>11}"
          f" {'R_yield':>9} {'R_ane':>9} {'R_hiad':>9}")
    print(sep)
    for name, row in rows:
        print(f"  {name:<20} {v(row,'yield_kg_ha','.0f'):>14} "
              f"{v(row,'total_N_kg_ha','.0f'):>11} "
              f"{v(row,'total_W_mm','.0f'):>11} "
              f"{v(row,'R_yield','+.3f'):>9} "
              f"{v(row,'R_ane','+.3f'):>9} "
              f"{v(row,'R_hiad','+.3f'):>9}")
    print(sep + '\n')


# ══════════════════════════════════════════════════════════════════════
def main():
    print('Generating Section 4.2 figures...')

    df_base = load(EP_LOG_BASE)
    if df_base is None:
        df_base = load('/workspace/episode_log_rw.csv')
    if df_base is not None:
        print(f'  Training log: {len(df_base)} episodes')
        plot_training_curves(df_base, OUT_CURVES)
        plot_losses(METRICS_BASE, OUT_LOSSES)
    else:
        print('  ⚠  episode_log.csv not found')

    baselines_df = load(BASELINE_CSV)
    if baselines_df is not None:
        plot_variants_bar(baselines_df, OUT_BAR)
        plot_tradeoff(baselines_df, OUT_TRADE)

    print_summary()
    print('Done.')

if __name__ == '__main__':
    main()
