"""
CAPQLEnv — Gymnasium adapter for CAPQL (Concave-augmented Pareto Q-Learning).

Differences from PCPPOEnv:
  - step() returns R_daily ONLY (no seasonal scalarization here)
  - info['reward_vec']   = [R_yield, R_ane, R_water_eff]  at terminal step
  - info['preference_w'] = current episode w  (also embedded in obs[-3:])
  - The training loop applies:
        r_aug = R_daily + done * ( w·R⃗  +  λ·mean(log(Rᵢ + 1 + ε)) )

Observation space (14 dims):
  [0-10]  crop + SoS features (normalized, same as pc_env_pcppo.py)
  [11-13] preference w = [w_yield, w_ane, w_water]  ~ Dirichlet([1,1,1])

Action space:
  Box([0, 0], [200, 50], float32)
    [0] anfer — nitrogen fertilizer  (kg N/ha)
    [1] amir  — irrigation depth     (mm)

Run smoke test inside Docker:
    python3 /workspace/gym-dssat-pdi/gym_dssat_pdi_samples/capql_env.py
"""
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from importlib import import_module

SmartFarmSoSEnv = import_module('02_smart_farm_env_pcppo').SmartFarmSoSEnv

N_OBJECTIVES = 3
CROP_DIM     = 11
OBS_DIM      = CROP_DIM + N_OBJECTIVES   # 14

NORM = {
    'topwt':      20000.0,
    'grnwt':      12000.0,
    'dap':          200.0,
    'vstage':        18.0,
    'xlai':           7.0,
    'cumsumfert':   300.0,
    'totir':       1000.0,
}

ACTION_LOW  = np.array([0.0,   0.0], dtype=np.float32)
ACTION_HIGH = np.array([200.0, 50.0], dtype=np.float32)


class CAPQLEnv(gym.Env):
    """
    Gymnasium wrapper for CAPQL.

    The seasonal reward vector is NOT scalarized here — it is returned raw
    in info['reward_vec'] so the training loop can apply the
    concave-augmented scalarization with the relabeled preference w.
    """

    metadata = {'render_modes': []}

    def __init__(self, mode='all', dssat_seed=123,
                 run_dssat_location='run_dssat',
                 enable_faults=False, fault_rate=0.02):
        super().__init__()

        self.sos_env = SmartFarmSoSEnv(
            mode=mode,
            seed=dssat_seed,
            run_dssat_location=run_dssat_location,
            enable_faults=enable_faults,
            fault_rate=fault_rate,
        )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.current_w  = np.full(N_OBJECTIVES, 1.0 / N_OBJECTIVES, dtype=np.float32)
        self._max_anfer = 200.0   # per-episode N cap, set at reset
        self._max_amir  =  50.0   # per-episode W cap, set at reset

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        self.current_w = np.random.dirichlet(np.ones(N_OBJECTIVES)).astype(np.float32)

        # Per-episode action caps driven by preference (squared law):
        #   preference=0.0 → full range unchanged
        #   preference=0.5 → 25% of range
        #   preference=1.0 → hard min (agronomically viable floor)
        #
        # N cap: at w_neff=1 → 2 kg/ha/day → ~320 kg/ha/season
        #        at w_neff=0.8 → ~8 kg/ha/day → ~1280 kg/ha/season
        #        → forces buffer to contain episodes near the agronomic optimum
        # W cap: at w_water=1 → 3 mm/day → ~480 mm/season (near 400mm target)
        w_neff  = float(self.current_w[1])
        w_water = float(self.current_w[2])
        self._max_anfer = float(np.clip(200.0 * (1.0 - w_neff) ** 2,  2.0, 200.0))
        self._max_amir  = float(np.clip( 50.0 * (1.0 - w_water) ** 2, 3.0,  50.0))

        sos_obs = self.sos_env.reset()
        return self._encode(sos_obs), {'preference_w': self.current_w.copy()}

    # ------------------------------------------------------------------ #
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        anfer  = float(np.clip(action[0], 0.0, self._max_anfer))
        amir   = float(np.clip(action[1], 0.0, self._max_amir))

        sos_obs, R_daily, done, sos_info = self.sos_env.step(
            {'anfer': anfer, 'amir': amir}
        )

        obs  = self._encode(sos_obs)
        info = dict(sos_info)
        info['preference_w']   = self.current_w.copy()
        info['action_applied'] = np.array([anfer, amir], dtype=np.float32)
        # reward_vec already set by SmartFarmSoSEnv_PCPPO in sos_info

        # Return R_daily only — training loop adds w·R⃗ + λ·g(R⃗) at terminal
        return obs, float(R_daily), bool(done), False, info

    # ------------------------------------------------------------------ #
    def close(self):
        self.sos_env.close()

    # ------------------------------------------------------------------ #
    def _encode(self, sos_obs):
        def _g(k):
            return float(sos_obs.get(f'crop_{k}', 0.0) or 0.0)

        sw_layers    = sos_obs.get('crop_sw', None)
        sw_mean      = float(np.mean(np.asarray(sw_layers, dtype=np.float32))) \
                       if sw_layers is not None else 0.0
        sensor_vals  = [float(sos_obs.get(f'sensor_{i}', 1.0) or 0.0)
                        for i in range(self.sos_env.n_sensors)]
        sensors_frac = float(np.mean(sensor_vals)) if sensor_vals else 1.0

        crop_feats = np.array([
            _g('topwt')      / NORM['topwt'],
            _g('grnwt')      / NORM['grnwt'],
            _g('dap')        / NORM['dap'],
            _g('vstage')     / NORM['vstage'],
            _g('xlai')       / NORM['xlai'],
            _g('cumsumfert') / NORM['cumsumfert'],
            _g('totir')      / NORM['totir'],
            sw_mean,
            float(sos_obs.get('energy_budget', 1.0) or 0.0),
            float(sos_obs.get('comm_quality',  1.0) or 0.0),
            sensors_frac,
        ], dtype=np.float32)

        return np.concatenate([crop_feats, self.current_w])


# ====================================================================== #
# Smoke test                                                               #
# ====================================================================== #
def _smoke_test():
    print("=" * 70)
    print("CAPQLEnv — smoke test")
    print("=" * 70)

    env = CAPQLEnv(mode='all', dssat_seed=123, run_dssat_location='run_dssat')
    assert env.action_space.shape      == (2,)
    assert env.observation_space.shape == (OBS_DIM,)
    print(f"  action_space      = {env.action_space}")
    print(f"  observation_space = {env.observation_space}")

    # ---- reset ----
    obs, info = env.reset()
    w = info['preference_w']
    print(f"\n  reset() — w={w.round(3)}  sum={w.sum():.3f}")
    print(f"  obs shape={obs.shape}  obs[-3:]={obs[-3:].round(3)}")
    assert obs.shape == (OBS_DIM,)
    assert np.allclose(obs[-3:], w, atol=1e-5), "w not in obs"
    print("  ✓ reset OK")

    # ---- single step ----
    obs, r, term, trunc, info = env.step(np.array([50.0, 8.0]))
    rv = info.get('reward_vec', None)
    assert rv is not None and rv.shape == (3,), "reward_vec missing"
    assert isinstance(r, float)
    print(f"\n  step() — R_daily={r:+.6f}  reward_vec={rv.round(4)}")
    print("  ✓ step OK")

    # ---- full episode ----
    obs, _ = env.reset()
    cum_daily = 0.0
    steps     = 0
    done      = False
    last_info = {}
    while not done:
        obs, r, done, _, last_info = env.step(env.action_space.sample())
        cum_daily += r
        steps     += 1

    sos = last_info.get('sos_state', {})
    rv  = last_info.get('reward_vec', np.zeros(3))
    w   = last_info.get('preference_w', np.ones(3) / 3)
    print(f"\n  Full episode — {steps} days")
    print(f"  sum R_daily = {cum_daily:+.4f}")
    print(f"  reward_vec  = {rv.round(4)}")
    print(f"  w·R⃗         = {float(np.dot(w, rv)):+.4f}")
    print(f"  yield       = {sos.get('grnwt', 0.):.0f} kg/ha")
    assert steps > 0
    print("  ✓ episode OK")

    env.close()
    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED.")
    print("=" * 70)


if __name__ == '__main__':
    try:
        _smoke_test()
    except AssertionError as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
