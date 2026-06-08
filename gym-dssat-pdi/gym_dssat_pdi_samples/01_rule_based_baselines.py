"""
Rule-based baselines for SmartFarmSoSEnv — hierarchical scalar reward design.

Every baseline runs through the SAME wrapper that PPO/PC-PPO agents will use,
so the scalar reward and its components are computed identically. This gives a
directly comparable baseline for evaluating trained agents.

Reward returned per step is a scalar float (hierarchical daily + seasonal).
Components are exposed via info['reward_components'] every step.

Baselines included:
    1. Random            — random fertilization and irrigation each day
    2. Fixed Schedule    — textbook 3-split N + periodic irrigation
    3. Stage-based       — V-stage-triggered N + developmentally-gated irrig
    4. FAO-56            — soil-water-balance irrigation + 3-split N

Run inside Docker:
    python3 /workspace/gym-dssat-pdi/gym_dssat_pdi_samples/01_rule_based_baselines.py
"""
import numpy as np

from importlib import import_module
SmartFarmSoSEnv = import_module('02_smart_farm_env').SmartFarmSoSEnv


# ============================================================
# FAO-56 helper functions
# ============================================================
TAW_MAIZE      = 150.0
P_DEPLETION    = 0.50
MAX_IRRIG_EVT  = 25.0


def kc_maize(days_after_planting):
    """Crop coefficient Kc for maize at a given DAP, per FAO-56 Table 12."""
    dap = days_after_planting
    if dap < 20:
        return 0.30
    elif dap < 50:
        return 0.30 + (1.20 - 0.30) * (dap - 20) / 30
    elif dap < 100:
        return 1.20
    elif dap < 140:
        return 1.20 - (1.20 - 0.60) * (dap - 100) / 40
    else:
        return 0.60


# ============================================================
# Observation helpers — wrapper prefixes crop vars with "crop_"
# ============================================================
def get_crop(obs, key, default=0.0):
    if obs is None:
        return float(default)
    val = obs.get(f'crop_{key}', default)
    if val is None:
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def fresh_results():
    """Empty result template for one episode."""
    return {
        # Cumulative scalar reward
        'cum_reward':    0.0,
        # Accumulated daily sub-rewards
        'R_water_sum':   0.0,
        'R_fert_sum':    0.0,
        'R_resource_sum': 0.0,
        'R_losses_sum':  0.0,
        # Terminal seasonal components (populated at episode end)
        'R_seasonal':    0.0,
        'R_yield':       0.0,
        'R_hiad':        0.0,
        'R_ane':         0.0,
        'R_penalty':     0.0,
        # Physical metrics
        'yield_kg_ha':   0.0,
        'total_N_kg_ha': 0.0,
        'total_W_mm':    0.0,
        'total_rain_mm': 0.0,
        'final_energy':  0.0,
        'comm_quality':  0.0,
    }


def _accumulate_step(rec, scalar_r, info):
    """Add one step's scalar reward and daily sub-reward components."""
    rec['cum_reward'] += float(scalar_r)
    c = info.get('reward_components', {})
    rec['R_water_sum']    += float(c.get('R_water',    0.0))
    rec['R_fert_sum']     += float(c.get('R_fert',     0.0))
    rec['R_resource_sum'] += float(c.get('R_resource', 0.0))
    rec['R_losses_sum']   += float(c.get('R_losses',   0.0))


def finalize_episode(rec, info):
    """Pull seasonal components and physical outcomes from the terminal step."""
    c   = info.get('reward_components', {})
    sos = info.get('sos_state', {})

    rec['R_seasonal'] = float(c.get('R_seasonal', 0.0))
    rec['R_yield']    = float(c.get('R_yield',    0.0))
    rec['R_hiad']     = float(c.get('R_hiad',     0.0))
    rec['R_ane']      = float(c.get('R_ane',      0.0))
    rec['R_penalty']  = float(c.get('R_penalty',  0.0))

    rec['yield_kg_ha']   = float(sos.get('grnwt',          0.0))
    rec['total_N_kg_ha'] = float(sos.get('total_nitrogen',  0.0))
    rec['total_W_mm']    = float(sos.get('total_water',     0.0))
    rec['total_rain_mm'] = float(sos.get('total_rain',      0.0))
    rec['final_energy']  = float(sos.get('energy_budget',   0.0))
    rec['comm_quality']  = float(sos.get('comm_quality',    0.0))


# ============================================================
# Baseline 1: Random agent
# ============================================================
def random_agent(env, n_episodes=50, seed=0):
    """Random fertilization in [0, 40] kg N/ha and irrigation in [0, 10] mm."""
    rng = np.random.default_rng(seed)
    all_results = []
    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        rec  = fresh_results()
        info = {}
        while not done:
            action = {
                'anfer': float(rng.uniform(0, 40)),
                'amir':  float(rng.uniform(0, 10)),
            }
            obs, R, done, info = env.step(action)
            _accumulate_step(rec, R, info)
        finalize_episode(rec, info)
        all_results.append(rec)
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:>3}: yield={rec['yield_kg_ha']:5.0f} kg/ha  "
                  f"N={rec['total_N_kg_ha']:4.0f} kg/ha  "
                  f"W={rec['total_W_mm']:5.0f} mm  "
                  f"cum_R={rec['cum_reward']:+7.3f}")
    return all_results


# ============================================================
# Baseline 2: Fixed-schedule agent
# ============================================================
def fixed_schedule_agent(env, n_episodes=50):
    """3-split N (40 kg on days 30/60/90) + periodic irrigation (8 mm every 5 days)."""
    all_results = []
    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        rec  = fresh_results()
        info = {}
        day  = 0
        while not done:
            day  += 1
            anfer = 40.0 if day in (30, 60, 90) else 0.0
            amir  = 8.0  if (day % 5 == 0)      else 0.0
            obs, R, done, info = env.step({'anfer': anfer, 'amir': amir})
            _accumulate_step(rec, R, info)
        finalize_episode(rec, info)
        all_results.append(rec)
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:>3}: yield={rec['yield_kg_ha']:5.0f} kg/ha  "
                  f"N={rec['total_N_kg_ha']:4.0f} kg/ha  "
                  f"W={rec['total_W_mm']:5.0f} mm  "
                  f"cum_R={rec['cum_reward']:+7.3f}")
    return all_results


# ============================================================
# Baseline 3: Stage-based agent
# ============================================================
def stage_based_agent(env, n_episodes=50):
    """
    Apply nitrogen at canonical maize V-stage triggers (V3, V6, V10), and
    irrigate every 3 days from emergence onward.
    """
    all_results = []
    for ep in range(n_episodes):
        obs         = env.reset()
        done        = False
        rec         = fresh_results()
        info        = {}
        last_vstage = -1.0
        while not done:
            vstage = get_crop(obs, 'vstage', 0.0)
            dap    = get_crop(obs, 'dap',    0.0)

            anfer = 0.0
            for trigger in (3.0, 6.0, 10.0):
                if last_vstage < trigger <= vstage:
                    anfer = 50.0
                    break
            last_vstage = vstage

            amir = 10.0 if (vstage > 1.0 and int(dap) % 3 == 0) else 0.0

            obs, R, done, info = env.step({'anfer': anfer, 'amir': float(amir)})
            _accumulate_step(rec, R, info)
        finalize_episode(rec, info)
        all_results.append(rec)
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:>3}: yield={rec['yield_kg_ha']:5.0f} kg/ha  "
                  f"N={rec['total_N_kg_ha']:4.0f} kg/ha  "
                  f"W={rec['total_W_mm']:5.0f} mm  "
                  f"cum_R={rec['cum_reward']:+7.3f}")
    return all_results


# ============================================================
# Baseline 4: FAO-56 agent
# ============================================================
def fao56_agent(env, n_episodes=50,
                taw=TAW_MAIZE, p_depletion=P_DEPLETION,
                fert_doses=(40, 40, 40),
                fert_days=(35, 65, 95)):
    """
    FAO-56 soil-water-balance irrigation + 3-split N at (35, 65, 95) DAP.

    D_r(t) = D_r(t-1) + ETc(t) - rain(t) - irrig(t-1)
    ETc = Kc(t) * ET0(t)
    Irrigate when D_r > p * TAW; refill capped at MAX_IRRIG_EVT mm/event.
    """
    raw = taw * p_depletion
    all_results = []

    for ep in range(n_episodes):
        obs        = env.reset()
        done       = False
        rec        = fresh_results()
        info       = {}
        depletion  = 0.0
        n_irrig_ev = 0
        day        = 0

        while not done:
            day += 1
            et0  = get_crop(obs, 'eo',   4.0)
            rain = get_crop(obs, 'rain', 0.0)
            kc   = kc_maize(day)
            etc  = kc * et0
            depletion = max(0.0, depletion + etc - rain)

            if depletion > raw:
                amir = min(depletion, MAX_IRRIG_EVT)
                depletion -= amir
                n_irrig_ev += 1
            else:
                amir = 0.0

            anfer = 0.0
            for d, dose in zip(fert_days, fert_doses):
                if day == d:
                    anfer = float(dose)
                    break

            obs, R, done, info = env.step({'anfer': anfer, 'amir': float(amir)})
            _accumulate_step(rec, R, info)

        finalize_episode(rec, info)
        rec['irrig_events'] = n_irrig_ev
        all_results.append(rec)
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1:>3}: yield={rec['yield_kg_ha']:5.0f} kg/ha  "
                  f"N={rec['total_N_kg_ha']:4.0f} kg/ha  "
                  f"W={rec['total_W_mm']:5.0f} mm ({n_irrig_ev:2d} ev)  "
                  f"cum_R={rec['cum_reward']:+7.3f}")
    return all_results


# ============================================================
# Reporting
# ============================================================
def summarize(name, results):
    """Print mean ± std of cumulative reward, sub-rewards, and physical metrics."""
    def ms(key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.std(vals)

    cr_m,  cr_s  = ms('cum_reward')
    ry_m,  ry_s  = ms('R_yield')
    rh_m,  rh_s  = ms('R_hiad')
    ra_m,  ra_s  = ms('R_ane')
    rs_m,  rs_s  = ms('R_seasonal')
    y_m,   y_s   = ms('yield_kg_ha')
    n_m,   n_s   = ms('total_N_kg_ha')
    w_m,   w_s   = ms('total_W_mm')

    print(f"\n  {name}")
    print(f"    cum_reward = {cr_m:+7.3f} ± {cr_s:5.3f}")
    print(f"    R_seasonal = {rs_m:+7.3f} ± {rs_s:5.3f}  "
          f"(R_yield={ry_m:+.3f}  R_hiad={rh_m:+.3f}  R_ane={ra_m:+.3f})")
    print(f"    yield      = {y_m:6.0f} ± {y_s:5.0f} kg/ha  "
          f"N = {n_m:5.0f} ± {n_s:4.0f} kg/ha  "
          f"W = {w_m:5.0f} ± {w_s:4.0f} mm")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    n_episodes = 50

    print("=" * 80)
    print("RULE-BASED BASELINES — SmartFarmSoSEnv  (mode='all')")
    print("Hierarchical scalar reward: daily shaping + seasonal terminal")
    print("=" * 80)

    print("\n[1/4] Random agent")
    env = SmartFarmSoSEnv(mode='all', seed=123,
                          run_dssat_location='run_dssat', enable_faults=False)
    random_results = random_agent(env, n_episodes)
    env.close()

    print("\n[2/4] Fixed-schedule agent")
    env = SmartFarmSoSEnv(mode='all', seed=123,
                          run_dssat_location='run_dssat', enable_faults=False)
    fixed_results = fixed_schedule_agent(env, n_episodes)
    env.close()

    print("\n[3/4] Stage-based agent")
    env = SmartFarmSoSEnv(mode='all', seed=123,
                          run_dssat_location='run_dssat', enable_faults=False)
    stage_results = stage_based_agent(env, n_episodes)
    env.close()

    print("\n[4/4] FAO-56 agent")
    env = SmartFarmSoSEnv(mode='all', seed=123,
                          run_dssat_location='run_dssat', enable_faults=False)
    fao_results = fao56_agent(env, n_episodes)
    env.close()

    print("\n" + "=" * 80)
    print(f"SUMMARY  (mean ± std over {n_episodes} episodes)")
    print("=" * 80)
    summarize("Random",           random_results)
    summarize("Fixed Schedule",   fixed_results)
    summarize("Stage-based",      stage_results)
    summarize("FAO-56",           fao_results)

    print("\n" + "=" * 80)
    print("Use these cum_reward values as the scalar-reward bar for PPO agents.")
    print("Use R_yield / N / W ratios as the multi-objective bar for PC-PPO.")
    print("=" * 80)
