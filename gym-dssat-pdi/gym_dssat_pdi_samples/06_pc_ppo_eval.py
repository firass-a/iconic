"""
Strong PC-PPO evaluation — preference grid, baselines, CSV export.

Env overrides:
    MODEL_PATH, EPISODES_PER_PREF, N_RANDOM_PREFS, DSSAT_SEED,
    OUTPUT_DIR, LOG_FILE

Run in Docker:
    python3 06_pc_ppo_eval.py
"""
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

try:
    import torch
except ImportError:
    print('Install PyTorch: pip install torch')
    raise

from importlib import import_module
pc = import_module('pc_env')
PCSmartFarmEnv = pc.PCSmartFarmEnv
ACTION_LOW = pc.ACTION_LOW
ACTION_HIGH = pc.ACTION_HIGH
OBS_DIM = pc.OBS_DIM

from ppo_agent import PPOAgent

MODEL_PATH = os.environ.get('MODEL_PATH', '/models/pc_ppo_custom_best.pt')
EPISODES_PER_PREF = int(os.environ.get('EPISODES_PER_PREF', '20'))
N_RANDOM_PREFS = int(os.environ.get('N_RANDOM_PREFS', '20'))
DSSAT_SEED = int(os.environ.get('DSSAT_SEED', '123'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/models/eval')
LOG_FILE = os.environ.get('LOG_FILE', '')

_log_handle = None


def _log_sink():
    global _log_handle
    if not LOG_FILE:
        return None
    if _log_handle is None:
        os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
        _log_handle = open(LOG_FILE, 'a', encoding='utf-8', buffering=1)
    return _log_handle


def log(msg):
    print(msg, flush=True)
    fh = _log_sink()
    if fh:
        fh.write(msg + '\n')
        fh.flush()


def close_log():
    global _log_handle
    if _log_handle is not None:
        _log_handle.close()
        _log_handle = None


def build_preference_grid(n_random=20, seed=42):
    """Corners + edges + uniform + random Dirichlet preferences."""
    rng = np.random.default_rng(seed)
    grid = [
        ('corner_yield', np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ('corner_water', np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ('corner_fert', np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ('edge_yield_water', np.array([0.5, 0.5, 0.0], dtype=np.float32)),
        ('edge_yield_fert', np.array([0.5, 0.0, 0.5], dtype=np.float32)),
        ('edge_water_fert', np.array([0.0, 0.5, 0.5], dtype=np.float32)),
        ('uniform', np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)),
    ]
    for i in range(n_random):
        w = rng.dirichlet(np.ones(3)).astype(np.float32)
        grid.append((f'random_{i:02d}', w))
    return grid


def run_episode(env, policy_fn, seed):
    """Run one season; policy_fn(obs) -> action array."""
    obs, _ = env.reset(seed=seed)
    actions = []
    cum_r = np.zeros(3, dtype=np.float64)
    scalar_return = 0.0
    done = False
    # Cumulative pools DSSAT exposes only monotonically grow; the terminal
    # step often returns an empty / reset state dict, so we keep the running
    # max across every step instead of reading the final state alone.
    peak_uptake = 0.0
    peak_leached = 0.0
    peak_denit = 0.0
    last_grnwt = 0.0
    last_topwt = 0.0
    while not done:
        action = np.asarray(policy_fn(obs), dtype=np.float32).flatten()
        action[0] = np.clip(action[0], ACTION_LOW[0], ACTION_HIGH[0])
        action[1] = np.clip(action[1], ACTION_LOW[1], ACTION_HIGH[1])
        obs, scalar_r, term, trunc, info = env.step(action)
        actions.append(action)
        cum_r += np.asarray(info['reward_vector'], dtype=np.float64)
        scalar_return += float(scalar_r)
        fs_step = info.get('full_state', {}) or {}
        peak_uptake = max(peak_uptake, float(fs_step.get('wtnup', 0.0) or 0.0))
        peak_leached = max(peak_leached, float(fs_step.get('cleach', 0.0) or 0.0))
        peak_denit = max(peak_denit, float(fs_step.get('cnox', 0.0) or 0.0))
        grnwt_step = float(fs_step.get('grnwt', 0.0) or 0.0)
        topwt_step = float(fs_step.get('topwt', 0.0) or 0.0)
        if grnwt_step > 0:
            last_grnwt = grnwt_step
        if topwt_step > 0:
            last_topwt = topwt_step
        done = term or trunc

    actions = np.array(actions)
    fs = info.get('full_state', {})
    w = info.get('preference', np.array([0.33, 0.33, 0.34], dtype=np.float32))
    # Cumulative N pools straight from DSSAT (kg N/ha, season totals)
    #   wtnup   = N taken up by the plant   (the agronomic real number)
    #   cleach  = N lost to deep leaching   (groundwater pollution)
    #   cnox    = N lost to denitrification (N2O / atmospheric loss)
    n_applied = float(info['totals']['nitrogen'])
    n_uptake = max(peak_uptake, float(fs.get('wtnup', 0.0) or 0.0))
    n_leached = max(peak_leached, float(fs.get('cleach', 0.0) or 0.0))
    n_denit = max(peak_denit, float(fs.get('cnox', 0.0) or 0.0))
    # Same trick for grnwt/topwt: terminal step can blank the state out.
    grnwt_final = max(last_grnwt, float(fs.get('grnwt', 0.0) or 0.0))
    topwt_final = max(last_topwt, float(fs.get('topwt', 0.0) or 0.0))
    n_waste = max(0.0, n_applied - n_uptake)
    waste_ratio = (n_applied / n_uptake) if n_uptake > 1.0 else float('nan')
    nue = (n_uptake / n_applied) if n_applied > 1.0 else float('nan')

    return {
        'grnwt': grnwt_final,
        'topwt': topwt_final,
        'tot_N': n_applied,
        'tot_N_uptake': n_uptake,
        'tot_N_leached': n_leached,
        'tot_N_denit': n_denit,
        'tot_N_waste': n_waste,
        'waste_ratio': waste_ratio,
        'NUE': nue,
        'tot_W': float(info['totals']['water']),
        'cum_R_yield': float(cum_r[0]),
        'cum_R_water': float(cum_r[1]),
        'cum_R_fert': float(cum_r[2]),
        'scalar_score': float(np.dot(w, cum_r)),
        'episode_return': scalar_return,
        'mean_N': float(actions[:, 0].mean()) if len(actions) else 0.0,
        'max_N': float(actions[:, 0].max()) if len(actions) else 0.0,
        'mean_W': float(actions[:, 1].mean()) if len(actions) else 0.0,
        'max_W': float(actions[:, 1].max()) if len(actions) else 0.0,
        'n_days': len(actions),
    }


def evaluate_agent(agent, pref_grid, episodes_per_pref, base_seed=2000):
    rows = []
    total = len(pref_grid) * episodes_per_pref
    done_eps = 0
    t0 = time.time()

    for pref_name, w in pref_grid:
        for ep in range(episodes_per_pref):
            seed = base_seed + done_eps
            # Vary dssat_seed per episode so each one draws a distinct weather year.
            # Without this every episode replays the same season (std=0 in summary).
            env = PCSmartFarmEnv(
                mode='all', dssat_seed=DSSAT_SEED + ep,
                preference=w.copy(), rng_seed=seed,
                random_weather=True,
            )
            try:
                metrics = run_episode(
                    env,
                    policy_fn=lambda obs, ag=agent: ag.predict(obs, deterministic=True),
                    seed=seed,
                )
            finally:
                env.close()

            rows.append(_row('pc_ppo', pref_name, w, ep, seed, metrics))
            done_eps += 1
            if done_eps % 10 == 0 or done_eps == total:
                elapsed = time.time() - t0
                eta = elapsed / done_eps * (total - done_eps) / 60
                log(f'  agent eval {done_eps}/{total}  ({elapsed/60:.1f}m elapsed, ~{eta:.0f}m left)')

    return rows


def evaluate_baselines(episodes_per_pref, base_seed=9000):
    """Fixed-action baselines with neutral preference (not used for action choice)."""
    neutral_w = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    baselines = [
        ('rainfed', lambda obs: np.array([0.0, 0.0], dtype=np.float32)),
        ('moderate', lambda obs: np.array([15.0, 8.0], dtype=np.float32)),
        ('high_input', lambda obs: np.array([50.0, 25.0], dtype=np.float32)),
    ]
    rows = []
    for bname, policy_fn in baselines:
        for ep in range(episodes_per_pref):
            seed = base_seed + len(rows)
            env = PCSmartFarmEnv(
                mode='all', dssat_seed=DSSAT_SEED + ep,
                preference=neutral_w.copy(), rng_seed=seed,
                random_weather=True,
            )
            try:
                metrics = run_episode(env, policy_fn, seed)
            finally:
                env.close()
            rows.append(_row(bname, bname, neutral_w, ep, seed, metrics))
    return rows


def _row(policy, pref_name, w, ep, seed, m):
    return {
        'policy': policy,
        'pref_name': pref_name,
        'w0': float(w[0]),
        'w1': float(w[1]),
        'w2': float(w[2]),
        'episode': ep,
        'seed': seed,
        'grnwt': m['grnwt'],
        'topwt': m['topwt'],
        'tot_N': m['tot_N'],
        'tot_N_uptake': m['tot_N_uptake'],
        'tot_N_leached': m['tot_N_leached'],
        'tot_N_denit': m['tot_N_denit'],
        'tot_N_waste': m['tot_N_waste'],
        'waste_ratio': m['waste_ratio'],
        'NUE': m['NUE'],
        'tot_W': m['tot_W'],
        'cum_R_yield': m['cum_R_yield'],
        'cum_R_water': m['cum_R_water'],
        'cum_R_fert': m['cum_R_fert'],
        'scalar_score': m['scalar_score'],
        'mean_N': m['mean_N'],
        'max_N': m['max_N'],
        'mean_W': m['mean_W'],
        'max_W': m['max_W'],
        'n_days': m['n_days'],
    }


def summarize(rows, group_key='pref_name'):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)

    summary = {}
    for name, items in sorted(groups.items()):
        y = [x['grnwt'] for x in items]
        n_app = [x['tot_N'] for x in items]
        n_upt = [x['tot_N_uptake'] for x in items]
        n_lch = [x['tot_N_leached'] for x in items]
        n_dnt = [x['tot_N_denit'] for x in items]
        w = [x['tot_W'] for x in items]
        # NUE averaged only over episodes where applied > 0 (rainfed corner-case)
        nue_vals = [x['NUE'] for x in items if np.isfinite(x['NUE'])]
        summary[name] = {
            'n': len(items),
            'grnwt_mean': float(np.mean(y)),
            'grnwt_std': float(np.std(y)),
            'tot_N_mean': float(np.mean(n_app)),
            'tot_N_std': float(np.std(n_app)),
            'tot_N_uptake_mean': float(np.mean(n_upt)),
            'tot_N_uptake_std': float(np.std(n_upt)),
            'tot_N_leached_mean': float(np.mean(n_lch)),
            'tot_N_denit_mean': float(np.mean(n_dnt)),
            'NUE_mean': float(np.mean(nue_vals)) if nue_vals else float('nan'),
            'tot_W_mean': float(np.mean(w)),
            'tot_W_std': float(np.std(w)),
        }
    return summary


def check_monotonicity(agent_summary):
    """Verify corner preferences behave as expected."""
    checks = []
    corners = {
        'yield': agent_summary.get('corner_yield'),
        'water': agent_summary.get('corner_water'),
        'fert': agent_summary.get('corner_fert'),
    }
    if not all(corners.values()):
        return checks

    y_y, y_w, y_f = corners['yield'], corners['water'], corners['fert']
    checks.append((
        'yield-priority highest grnwt among corners',
        y_y['grnwt_mean'] >= y_w['grnwt_mean'] and y_y['grnwt_mean'] >= y_f['grnwt_mean'],
        f"yield={y_y['grnwt_mean']:.0f}  water={y_w['grnwt_mean']:.0f}  fert={y_f['grnwt_mean']:.0f}",
    ))
    checks.append((
        'water-priority lowest tot_W among corners',
        y_w['tot_W_mean'] <= y_y['tot_W_mean'] and y_w['tot_W_mean'] <= y_f['tot_W_mean'],
        f"yield_W={y_y['tot_W_mean']:.0f}  water_W={y_w['tot_W_mean']:.0f}  fert_W={y_f['tot_W_mean']:.0f}",
    ))
    checks.append((
        'fert-priority lowest tot_N among corners',
        y_f['tot_N_mean'] <= y_y['tot_N_mean'] and y_f['tot_N_mean'] <= y_w['tot_N_mean'],
        f"yield_N={y_y['tot_N_mean']:.0f}  water_N={y_w['tot_N_mean']:.0f}  fert_N={y_f['tot_N_mean']:.0f}",
    ))
    return checks


def save_results(rows, summary, checks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'eval_{stamp}.csv')
    json_path = os.path.join(out_dir, f'eval_{stamp}_summary.json')

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        'model_path': MODEL_PATH,
        'episodes_per_pref': EPISODES_PER_PREF,
        'n_random_prefs': N_RANDOM_PREFS,
        'timestamp': stamp,
        'summary': summary,
        'monotonicity_checks': [
            {'test': t, 'passed': bool(p), 'detail': d} for t, p, d in checks
        ],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def load_agent(path):
    agent = PPOAgent(
        obs_dim=OBS_DIM,
        act_dim=2,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        device='cpu',
    )
    agent.load(path)
    return agent


def print_summary_table(title, summary):
    log('')
    log(title)
    log('-' * 120)
    log(
        f'  {"name":<18s}  {"n":>3s}  {"grnwt":>14s}  '
        f'{"N_applied":>11s}  {"N_uptake":>10s}  {"N_leach":>8s}  {"NUE%":>6s}  {"tot_W":>10s}'
    )
    log('-' * 120)
    for name, s in sorted(summary.items()):
        nue_pct = s['NUE_mean'] * 100 if np.isfinite(s['NUE_mean']) else float('nan')
        nue_str = f'{nue_pct:5.1f}' if np.isfinite(nue_pct) else '  n/a'
        log(
            f'  {name:<18s}  {s["n"]:3d}  '
            f'{s["grnwt_mean"]:6.0f} ± {s["grnwt_std"]:4.0f}  '
            f'{s["tot_N_mean"]:5.0f}±{s["tot_N_std"]:4.0f}  '
            f'{s["tot_N_uptake_mean"]:5.0f}±{s["tot_N_uptake_std"]:3.0f}  '
            f'{s["tot_N_leached_mean"]:6.1f}  '
            f'{nue_str}  '
            f'{s["tot_W_mean"]:5.0f}±{s["tot_W_std"]:3.0f}'
        )


def main():
    log('=' * 95)
    log('STRONG PC-PPO EVAL')
    log('=' * 95)
    log(f'  model           : {MODEL_PATH}')
    log(f'  episodes/pref   : {EPISODES_PER_PREF}')
    log(f'  random prefs    : {N_RANDOM_PREFS}')
    log(f'  preference grid : {7 + N_RANDOM_PREFS} points (corners + edges + uniform + random)')
    log(f'  total agent eps : {(7 + N_RANDOM_PREFS) * EPISODES_PER_PREF}')
    log(f'  baseline eps    : {3 * EPISODES_PER_PREF}  (rainfed, moderate, high_input)')
    log('=' * 95)

    if not os.path.isfile(MODEL_PATH):
        log(f'ERROR: model not found: {MODEL_PATH}')
        sys.exit(1)

    agent = load_agent(MODEL_PATH)
    pref_grid = build_preference_grid(n_random=N_RANDOM_PREFS)
    t0 = time.time()

    log('')
    log('AGENT — preference-conditioned policy')
    agent_rows = evaluate_agent(agent, pref_grid, EPISODES_PER_PREF)
    agent_summary = summarize(agent_rows)

    log('')
    log('BASELINES — fixed actions')
    baseline_rows = evaluate_baselines(EPISODES_PER_PREF)
    baseline_summary = summarize(baseline_rows, group_key='policy')

    all_rows = agent_rows + baseline_rows
    checks = check_monotonicity(agent_summary)
    csv_path, json_path = save_results(all_rows, {**agent_summary, **baseline_summary}, checks, OUTPUT_DIR)

    print_summary_table('AGENT SUMMARY (mean ± std)', agent_summary)
    print_summary_table('BASELINE SUMMARY (mean ± std)', baseline_summary)

    log('')
    log('MONOTONICITY CHECKS (corner preferences)')
    log('-' * 95)
    for test, passed, detail in checks:
        status = 'PASS' if passed else 'FAIL'
        log(f'  [{status}] {test}')
        log(f'         {detail}')

    log('')
    log(f'Eval time       : {(time.time()-t0)/60:.1f} min')
    log(f'CSV saved       : {csv_path}')
    log(f'Summary JSON    : {json_path}')
    log('=' * 95)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f'\nERROR: {type(e).__name__}: {e}')
        raise
    finally:
        close_log()
