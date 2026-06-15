"""
Thesis figures from training log + strong-eval CSV.

Produces 8 PNGs in figures/ :
    fig_training_curves.png       — r_mean, R_components, mean_N, mean_W over training
    fig_baseline_comparison.png   — agent corners + uniform vs baselines (yield/N/water)
    fig_corners_boxplot.png       — 20-episode distributions for each corner
    fig_preference_response.png   — does behaviour change with preference weights?
    fig_pareto.png                — tot_N vs tot_W coloured by yield (the MORL plot)
    fig_action_profile.png        — one season's daily actions + grain accumulation
                                    (requires figures/daily_actions.csv)
    fig_corner_overlay.png        — cumulative N/W/grain for all 3 corners, same weather
                                    (requires figures/daily_actions_{yield,water,fert}.csv)
    fig_ternary_simplex.png       — preference simplex coloured by N/W/yield
                                    (requires mpltern; pip-installed by make_plots.ps1)

Run inside Docker (via make_plots.ps1) or locally with matplotlib + pandas + seaborn (+ mpltern).
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SAMPLES = Path(__file__).resolve().parent
LOGS_DIR = SAMPLES / 'logs'
EVAL_DIR = SAMPLES / 'models' / 'eval'
FIGURES = SAMPLES / 'figures'
FIGURES.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid', context='paper', font_scale=1.05)
TITLE_KW = dict(fontsize=12, fontweight='bold')
SUPTITLE_KW = dict(fontsize=13, fontweight='bold')


def latest_file(directory, pattern):
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def parse_training_log(path):
    """Extract per-update metrics from a training log."""
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    blocks = re.split(r'--- update\s+(\d+)/\d+', text)
    rows = []
    for i in range(1, len(blocks), 2):
        try:
            update_num = int(blocks[i])
        except ValueError:
            continue
        body = blocks[i + 1] if i + 1 < len(blocks) else ''
        m_r = re.search(r'r_mean=([+-]?[\d.]+)', body)
        m_rvec = re.search(
            r'yield=([+-]?[\d.]+)\s+water=([+-]?[\d.]+)\s+fert=([+-]?[\d.]+)', body
        )
        m_n = re.search(r'mean_N=\s*([\d.]+)', body)
        m_w = re.search(r'mean_W=\s*([\d.]+)', body)
        if not m_r:
            continue
        rows.append({
            'update': update_num,
            'r_mean': float(m_r.group(1)),
            'R_yield': float(m_rvec.group(1)) if m_rvec else np.nan,
            'R_water': float(m_rvec.group(2)) if m_rvec else np.nan,
            'R_fert': float(m_rvec.group(3)) if m_rvec else np.nan,
            'mean_N': float(m_n.group(1)) if m_n else np.nan,
            'mean_W': float(m_w.group(1)) if m_w else np.nan,
        })
    return pd.DataFrame(rows)


def plot_training_curves(df, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    ax.plot(df['update'], df['r_mean'], color='#1976D2', lw=1.4)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title('Scalar reward (r_mean) per update', **TITLE_KW)
    ax.set_xlabel('Update')
    ax.set_ylabel('r_mean')

    ax = axes[0, 1]
    ax.plot(df['update'], df['R_yield'], label='R_yield', color='#2E7D32', lw=1.4)
    ax.plot(df['update'], df['R_water'], label='R_water', color='#1976D2', lw=1.4)
    ax.plot(df['update'], df['R_fert'],  label='R_fert',  color='#F57C00', lw=1.4)
    ax.axhline(0, color='gray', lw=0.5)
    ax.legend(loc='lower right')
    ax.set_title('Per-objective reward components (rollout mean)', **TITLE_KW)
    ax.set_xlabel('Update')
    ax.set_ylabel('R component')

    ax = axes[1, 0]
    ax.plot(df['update'], df['mean_N'], color='#C62828', lw=1.4)
    ax.set_title('Mean N applied per day (rollout)', **TITLE_KW)
    ax.set_xlabel('Update')
    ax.set_ylabel('kg N / ha / day')

    ax = axes[1, 1]
    ax.plot(df['update'], df['mean_W'], color='#1976D2', lw=1.4)
    ax.set_title('Mean water applied per day (rollout)', **TITLE_KW)
    ax.set_xlabel('Update')
    ax.set_ylabel('mm / day')

    fig.suptitle('PC-PPO training progression', **SUPTITLE_KW)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def plot_baseline_comparison(eval_df, out_path):
    """Headline bar chart: agent corners + uniform vs baselines."""
    targets = [
        ('rainfed',        'rainfed',       '#9E9E9E'),
        ('corner_yield',   'agent\n(yield)', '#2E7D32'),
        ('corner_water',   'agent\n(water)', '#1976D2'),
        ('corner_fert',    'agent\n(fert)',  '#F57C00'),
        ('uniform',        'agent\n(uniform)', '#7B1FA2'),
        ('moderate',       'moderate\n(15N, 8mm)', '#455A64'),
        ('high_input',     'high_input\n(50N, 25mm)', '#212121'),
    ]
    rows = []
    for key, label, color in targets:
        if key in ('rainfed', 'moderate', 'high_input'):
            sub = eval_df[eval_df['policy'] == key]
        else:
            sub = eval_df[eval_df['pref_name'] == key]
        if len(sub) == 0:
            continue
        rows.append({
            'label': label, 'color': color,
            'grnwt_m': sub['grnwt'].mean(), 'grnwt_s': sub['grnwt'].std(),
            'N_m': sub['tot_N'].mean(),     'N_s': sub['tot_N'].std(),
            'W_m': sub['tot_W'].mean(),     'W_s': sub['tot_W'].std(),
        })
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    colors = list(df['color'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, key, ylabel, title in zip(
        axes,
        ['grnwt', 'N', 'W'],
        ['Grain weight (kg/ha)', 'Total N applied (kg/ha)', 'Total water applied (mm)'],
        ['Yield', 'Nitrogen use', 'Water use'],
    ):
        ax.bar(
            x, df[f'{key}_m'], yerr=df[f'{key}_s'],
            color=colors, capsize=4, edgecolor='black', linewidth=0.7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(df['label'], rotation=0, fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, **TITLE_KW)

    fig.suptitle('Agent vs baselines — 20 episodes each (random weather)', **SUPTITLE_KW)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def plot_corner_boxplots(eval_df, out_path):
    corners = ['corner_yield', 'corner_water', 'corner_fert']
    palette = {'corner_yield': '#2E7D32', 'corner_water': '#1976D2', 'corner_fert': '#F57C00'}
    sub = eval_df[eval_df['pref_name'].isin(corners)].copy()
    sub['pref_name'] = pd.Categorical(sub['pref_name'], categories=corners, ordered=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, col, ylabel, title in zip(
        axes,
        ['grnwt', 'tot_N', 'tot_W'],
        ['Grain weight (kg/ha)', 'Total N applied (kg/ha)', 'Total water applied (mm)'],
        ['Yield distribution', 'N distribution', 'Water distribution'],
    ):
        sns.boxplot(data=sub, x='pref_name', y=col, ax=ax,
                    hue='pref_name', palette=palette, legend=False,
                    width=0.55, fliersize=3, linewidth=1.0)
        sns.stripplot(data=sub, x='pref_name', y=col, ax=ax,
                      color='black', alpha=0.45, size=3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Preference corner')
        ax.set_title(title, **TITLE_KW)
        labels = ['yield\n[1,0,0]', 'water\n[0,1,0]', 'fert\n[0,0,1]']
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)

    fig.suptitle('Corner-preference distributions across 20 weather years', **SUPTITLE_KW)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def plot_preference_response(eval_df, out_path):
    """Scatter w_x vs target_x for all 27 agent preferences."""
    agent = eval_df[~eval_df['policy'].isin(['rainfed', 'moderate', 'high_input'])].copy()
    summary = agent.groupby('pref_name').agg(
        w_y=('w0', 'first'),
        w_w=('w1', 'first'),
        w_f=('w2', 'first'),
        grnwt_m=('grnwt', 'mean'), grnwt_s=('grnwt', 'std'),
        N_m=('tot_N', 'mean'),     N_s=('tot_N', 'std'),
        W_m=('tot_W', 'mean'),     W_s=('tot_W', 'std'),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ('w_w', 'W_m', 'W_s', '#1976D2',
         'w_water  (weight on water-saving)',
         'Total water applied (mm)',
         'Water-saving preference → less water'),
        ('w_f', 'N_m', 'N_s', '#F57C00',
         'w_fert  (weight on fertilizer-saving)',
         'Total N applied (kg/ha)',
         'Fert-saving preference → less N'),
        ('w_y', 'grnwt_m', 'grnwt_s', '#2E7D32',
         'w_yield  (weight on yield)',
         'Grain weight (kg/ha)',
         'Yield-priority → grain (capped near 10.8 t/ha)'),
    ]

    for ax, (wx, ym, ys, color, xlab, ylab, title) in zip(axes, panels):
        ax.errorbar(summary[wx], summary[ym], yerr=summary[ys],
                    fmt='o', color=color, capsize=3, alpha=0.8,
                    markersize=6, markeredgecolor='black', markeredgewidth=0.5)
        # Linear fit
        if summary[wx].nunique() > 1:
            z = np.polyfit(summary[wx], summary[ym], 1)
            xs = np.linspace(0, 1, 50)
            ax.plot(xs, z[0] * xs + z[1], '--', color='gray', alpha=0.7,
                    label=f'slope = {z[0]:.0f}')
            ax.legend(loc='best', fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, **TITLE_KW)

    fig.suptitle('Preference responsiveness — 27 preference points x 20 episodes', **SUPTITLE_KW)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def plot_pareto(eval_df, out_path):
    """Tot_N vs tot_W coloured by yield + baseline markers."""
    agent = eval_df[~eval_df['policy'].isin(['rainfed', 'moderate', 'high_input'])].copy()
    summary = agent.groupby('pref_name').agg(
        N_m=('tot_N', 'mean'), W_m=('tot_W', 'mean'), Y_m=('grnwt', 'mean'),
    ).reset_index()
    bl = (eval_df[eval_df['policy'].isin(['rainfed', 'moderate', 'high_input'])]
          .groupby('policy').agg(
              N_m=('tot_N', 'mean'), W_m=('tot_W', 'mean'), Y_m=('grnwt', 'mean'),
          ).reset_index())

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sc = ax.scatter(summary['N_m'], summary['W_m'], c=summary['Y_m'],
                    s=120, cmap='viridis', edgecolor='black', linewidth=0.6,
                    alpha=0.9, zorder=3, label='agent (27 preferences)')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Grain weight (kg/ha)')

    x_max = max(summary['N_m'].max(), bl['N_m'].max())
    for _, r in bl.iterrows():
        ax.scatter(r['N_m'], r['W_m'], s=240, marker='X',
                   facecolor='white', edgecolor='#C62828', linewidth=2.0, zorder=5)
        right_side = r['N_m'] > 0.6 * x_max
        ax.annotate(
            f"{r['policy']}\n({r['Y_m']:.0f} kg/ha)",
            (r['N_m'], r['W_m']),
            xytext=(-10 if right_side else 10, 8),
            textcoords='offset points',
            ha='right' if right_side else 'left',
            fontsize=9, color='#C62828', fontweight='bold',
        )

    # Annotate corners specifically
    name_to_label = {'corner_yield': 'Y', 'corner_water': 'W', 'corner_fert': 'F'}
    for _, r in summary.iterrows():
        if r['pref_name'] in name_to_label:
            ax.annotate(name_to_label[r['pref_name']],
                        (r['N_m'], r['W_m']), xytext=(7, 7),
                        textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_xlabel('Total N applied (kg/ha)')
    ax.set_ylabel('Total water applied (mm)')
    ax.set_title('Input frontier — agent (27 prefs) vs baselines', **TITLE_KW)
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def plot_action_profile(daily_csv, out_path):
    daily_csv = Path(daily_csv)
    if not daily_csv.is_file():
        print(f'  SKIPPED {out_path.name} — no {daily_csv.name} (run make_plots.ps1)')
        return
    df = pd.read_csv(daily_csv)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(df['day'], df['action_N'], color='#C62828', width=0.9, alpha=0.85)
    axes[0].set_ylabel('N (kg/ha/day)')
    axes[0].set_title('Daily nitrogen application', **TITLE_KW)

    axes[1].bar(df['day'], df['action_W'], color='#1976D2', width=0.9, alpha=0.85)
    axes[1].set_ylabel('Water (mm/day)')
    axes[1].set_title('Daily irrigation', **TITLE_KW)

    axes[2].plot(df['day'], df['grnwt'], color='#2E7D32', lw=2)
    axes[2].fill_between(df['day'], 0, df['grnwt'], color='#2E7D32', alpha=0.2)
    axes[2].set_ylabel('Grain weight (kg/ha)')
    axes[2].set_xlabel('Days after planting')
    axes[2].set_title('Grain accumulation', **TITLE_KW)

    fig.suptitle('One representative season — uniform preference [1/3, 1/3, 1/3]', **SUPTITLE_KW)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def plot_corner_overlay(out_path):
    """Cumulative N, W, grain for all 3 corners under the SAME weather."""
    files = {
        'corner_yield': (FIGURES / 'daily_actions_yield.csv', '#2E7D32'),
        'corner_water': (FIGURES / 'daily_actions_water.csv', '#1976D2'),
        'corner_fert':  (FIGURES / 'daily_actions_fert.csv',  '#F57C00'),
    }
    avail = {k: (p, c) for k, (p, c) in files.items() if p.is_file()}
    if not avail:
        print(f'  SKIPPED {out_path.name} — no per-corner CSVs')
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for name, (p, c) in avail.items():
        df = pd.read_csv(p)
        axes[0].plot(df['day'], df['action_N'].cumsum(), color=c, lw=2.0, label=name)
        axes[1].plot(df['day'], df['action_W'].cumsum(), color=c, lw=2.0, label=name)
        axes[2].plot(df['day'], df['grnwt'],            color=c, lw=2.0, label=name)

    axes[0].set_ylabel('Cumulative N (kg/ha)')
    axes[0].set_title('Cumulative nitrogen', **TITLE_KW)
    axes[0].legend(loc='upper left')

    axes[1].set_ylabel('Cumulative water (mm)')
    axes[1].set_title('Cumulative water', **TITLE_KW)

    axes[2].set_ylabel('Grain weight (kg/ha)')
    axes[2].set_xlabel('Days after planting')
    axes[2].set_title('Grain accumulation', **TITLE_KW)

    fig.suptitle('Same weather year, three preferences — how the agent substitutes', **SUPTITLE_KW)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def plot_ternary_simplex(eval_df, out_path):
    """3-panel preference simplex coloured by N, W, yield."""
    try:
        import mpltern  # noqa: F401
    except ImportError:
        print(f'  SKIPPED {out_path.name} — mpltern not installed')
        return

    agent = eval_df[~eval_df['policy'].isin(['rainfed', 'moderate', 'high_input'])].copy()
    summary = agent.groupby('pref_name').agg(
        w_y=('w0', 'first'), w_w=('w1', 'first'), w_f=('w2', 'first'),
        N_m=('tot_N', 'mean'),
        W_m=('tot_W', 'mean'),
        Y_m=('grnwt', 'mean'),
    ).reset_index()

    panels = [
        ('N_m', 'Reds',   'Total N (kg/ha)'),
        ('W_m', 'Blues',  'Total water (mm)'),
        ('Y_m', 'Greens', 'Grain weight (kg/ha)'),
    ]

    fig = plt.figure(figsize=(17, 5.5))
    for i, (col, cmap, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection='ternary')
        sc = ax.scatter(
            summary['w_y'], summary['w_w'], summary['w_f'],
            c=summary[col], cmap=cmap, s=160,
            edgecolor='black', linewidth=0.5, zorder=3,
        )
        ax.set_tlabel('w_yield')
        ax.set_llabel('w_water')
        ax.set_rlabel('w_fert')
        ax.taxis.set_label_position('tick1')
        ax.laxis.set_label_position('tick1')
        ax.raxis.set_label_position('tick1')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, **TITLE_KW)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.12)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Preference simplex — agent behaviour across the 3-objective space',
                 **SUPTITLE_KW)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path.name}')


def main():
    print('=' * 70)
    print('PC-PPO thesis figures')
    print('=' * 70)

    train_log = latest_file(LOGS_DIR, 'train_*.log')
    eval_csv  = latest_file(EVAL_DIR, 'eval_*.csv')
    if train_log is None:
        sys.exit('ERROR: no training log found in logs/')
    if eval_csv is None:
        sys.exit('ERROR: no eval CSV found in models/eval/')

    print(f'  train log : {train_log.name}')
    print(f'  eval CSV  : {eval_csv.name}')
    print(f'  output    : {FIGURES}')
    print()

    print('[1/8] Training curves...')
    train_df = parse_training_log(train_log)
    print(f'  parsed {len(train_df)} updates')
    plot_training_curves(train_df, FIGURES / 'fig_training_curves.png')

    print('\n[2/8] Loading eval CSV...')
    eval_df = pd.read_csv(eval_csv)
    print(f'  loaded {len(eval_df)} episodes ({eval_df["pref_name"].nunique()} preferences)')

    print('\n[3/8] Agent vs baselines...')
    plot_baseline_comparison(eval_df, FIGURES / 'fig_baseline_comparison.png')

    print('\n[4/8] Corner boxplots...')
    plot_corner_boxplots(eval_df, FIGURES / 'fig_corners_boxplot.png')

    print('\n[5/8] Preference response...')
    plot_preference_response(eval_df, FIGURES / 'fig_preference_response.png')

    print('\n[6/8] Pareto / input frontier...')
    plot_pareto(eval_df, FIGURES / 'fig_pareto.png')

    print('\n[7/8] Daily action profile (uniform)...')
    plot_action_profile(FIGURES / 'daily_actions.csv', FIGURES / 'fig_action_profile.png')

    print('\n[7b/8] Corner action overlay...')
    plot_corner_overlay(FIGURES / 'fig_corner_overlay.png')

    print('\n[8/8] Ternary preference simplex...')
    plot_ternary_simplex(eval_df, FIGURES / 'fig_ternary_simplex.png')

    print()
    print('Done. Figures saved to:')
    print(f'  {FIGURES}')


if __name__ == '__main__':
    main()
