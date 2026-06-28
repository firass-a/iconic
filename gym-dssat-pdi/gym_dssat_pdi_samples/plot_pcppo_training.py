"""
PC-PPO v5 training curves — CAPQL-style (light raw + rolling(30) overlay).

Parses logs/train_20260605_182258.log (or any compatible custom PC-PPO log).
Reconstructs per-episode points from logged `episode returns` and
`harvest yields` tails; estimates season total N as mean_N × 161 days.

Outputs:
  figures/pcppo_v5_training_curves.png   — 1×3 panel (matches CAPQL slide style)
  figures/pcppo_v5_training_curves_full.png — 2×3 extended panel
  figures/pcppo_v5_episode_reconstructed.csv

Run:
  python plot_pcppo_training.py
"""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LOG_DEFAULT = HERE / 'logs' / 'train_20260605_182258.log'
EPISODE_CSV = HERE / 'pc_ppo_episode_log.csv'
OUT_DIR = HERE / 'figures'
OUT_MAIN = OUT_DIR / 'pcppo_v5_training_curves.png'
OUT_FULL = OUT_DIR / 'pcppo_v5_training_curves_full.png'
OUT_CSV = OUT_DIR / 'pcppo_v5_episode_reconstructed.csv'

EP_LEN = 161   # typical maize season length in DSSAT runs
ROLL_W = 30


def _panel(ax, x, y, title, color, w=ROLL_W):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.plot(x, y, alpha=0.2, color=color, linewidth=0.5)
    if len(y) >= w:
        roll = np.convolve(y, np.ones(w) / w, mode='valid')
        ax.plot(x[w - 1:], roll, color=color, linewidth=1.8, label=f'rolling({w})')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Episode', fontsize=8)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)


def parse_training_log(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    blocks = re.split(r'(?=--- update\s+\d+/)', text)

    ret_eps, ret_vals = [], []
    yld_eps, yld_vals = [], []
    n_eps, n_vals = [], []

    for block in blocks:
        m_up = re.search(
            r'--- update\s+(\d+)/(\d+).*?total_seasons=(\d+)',
            block, re.DOTALL,
        )
        if not m_up:
            continue
        total_seasons = int(m_up.group(3))

        m_ret = re.search(r'episode returns \(last\): \[(.+?)\]', block)
        if m_ret:
            vals = [float(v.strip()) for v in m_ret.group(1).split(',')]
            start = total_seasons - len(vals) + 1
            for i, v in enumerate(vals):
                ret_eps.append(start + i)
                ret_vals.append(v)

        m_yld = re.search(r'harvest yields \(last\): \[(.+?)\] kg/ha', block)
        if m_yld:
            vals = [float(v.strip()) for v in m_yld.group(1).split(',')]
            start = total_seasons - len(vals) + 1
            for i, v in enumerate(vals):
                yld_eps.append(start + i)
                yld_vals.append(v)

        m_n = re.search(r'mean_N=\s+([\d.]+)', block)
        m_ret2 = re.search(r'episode returns \(last\): \[(.+?)\]', block)
        if m_n and m_ret2:
            tot_n = float(m_n.group(1)) * EP_LEN
            n_count = len(m_ret2.group(1).split(','))
            start = total_seasons - n_count + 1
            for i in range(n_count):
                n_eps.append(start + i)
                n_vals.append(tot_n)

    return {
        'return': (ret_eps, ret_vals),
        'yield': (yld_eps, yld_vals),
        'total_N': (n_eps, n_vals),
    }


def load_episode_csv(path):
    if not Path(path).exists():
        return None
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    if not rows:
        return None
    eps = [int(r['episode']) for r in rows]
    series = {
        'return': (eps, [float(r['cum_reward']) for r in rows]),
        'yield': (eps, [float(r['yield_kg_ha']) for r in rows]),
        'total_N': (eps, [float(r['total_N_kg_ha']) for r in rows]),
    }
    if 'total_W_mm' in rows[0]:
        series['total_W'] = (eps, [float(r['total_W_mm']) for r in rows])
    if 'R_yield' in rows[0]:
        series['R_yield'] = (eps, [float(r['R_yield']) for r in rows])
        series['R_water'] = (eps, [float(r['R_water']) for r in rows])
        series['R_fert'] = (eps, [float(r['R_fert']) for r in rows])
    return series


def load_series(log_path=None):
    series = load_episode_csv(EPISODE_CSV)
    if series:
        print(f'  dense episode log: {EPISODE_CSV} ({len(series["return"][0])} episodes)')
        return series
    log_path = Path(log_path or LOG_DEFAULT)
    if not log_path.exists():
        raise FileNotFoundError(f'No {EPISODE_CSV} and no log at {log_path}')
    print(f'  sparse fallback: {log_path.name}')
    return parse_training_log(log_path)


def save_csv(series, out_path):
    keys = sorted(set(series['return'][0]) | set(series['yield'][0]) | set(series['total_N'][0]))
    lookup = {
        'return': dict(zip(*series['return'])),
        'yield': dict(zip(*series['yield'])),
        'total_N': dict(zip(*series['total_N'])),
    }
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'cum_reward', 'yield_kg_ha', 'total_N_kg_ha'])
        for ep in keys:
            w.writerow([
                ep,
                f"{lookup['return'].get(ep, '')}",
                f"{lookup['yield'].get(ep, '')}",
                f"{lookup['total_N'].get(ep, '')}",
            ])


def plot_main(series, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [
        (axes[0], series['return'][0], series['return'][1], 'Cumulative Reward', 'tab:blue'),
        (axes[1], series['yield'][0], series['yield'][1], 'Yield (kg/ha)', 'tab:green'),
        (axes[2], series['total_N'][0], series['total_N'][1], 'Total N (kg/ha)', 'tab:orange'),
    ]
    for ax, x, y, title, color in panels:
        if x and y:
            _panel(ax, x, y, title, color)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'OK  {out_path}')


def plot_full(series, out_path):
    """Extended 2×3 layout like CAPQL training script."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        (axes[0, 0], series['return'][0], series['return'][1], 'Cumulative Reward', 'tab:blue'),
        (axes[0, 1], series['yield'][0], series['yield'][1], 'Yield (kg/ha)', 'tab:green'),
        (axes[0, 2], series['total_N'][0], series['total_N'][1], 'Total N (kg/ha)', 'tab:orange'),
        (axes[1, 0], *series.get('total_W', ([], [])), 'Total Water (mm)', 'tab:cyan'),
        (axes[1, 1], *series.get('R_yield', ([], [])), 'R_yield (terminal)', 'tab:purple'),
        (axes[1, 2], *series.get('R_water', ([], [])), 'R_water (terminal)', 'tab:red'),
    ]
    for ax, x, y, title, color in panels:
        if x and y:
            _panel(ax, x, y, title, color)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, f'{title}\n(not available)', ha='center', va='center',
                    fontsize=10, color='#666', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'OK  {out_path}')


def main(log_path=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series = load_series(log_path)
    n_ret = len(series['return'][0])
    print(f'  episodes plotted: {n_ret}')
    if not n_ret:
        raise RuntimeError('No episode data found.')

    save_csv(series, OUT_CSV)
    plot_main(series, OUT_MAIN)
    plot_full(series, OUT_FULL)


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
