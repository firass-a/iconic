"""
CAPQLEnv v2 — same as capql_env.py with one change:

  Corner injection (Fix B):
    20% of episodes use a pure preference corner:
      ~6.7%  w = [1, 0, 0]  (yield only)
      ~6.7%  w = [0, 1, 0]  (N efficiency only)
      ~6.7%  w = [0, 0, 1]  (water only)
      80%    w ~ Dirichlet([1, 1, 1])  (random mix)

  This ensures the policy is explicitly trained on the exact corners
  that are used during evaluation, fixing the corner underrepresentation
  caused by Dirichlet([1,1,1]) having near-zero mass at the vertices.

Everything else — action caps, obs encoding, reward passthrough — is
identical to capql_env.py.
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

# Pure corners injected at eval and during training
_CORNERS = np.array([
    [1.0, 0.0, 0.0],   # yield
    [0.0, 1.0, 0.0],   # n_eff
    [0.0, 0.0, 1.0],   # water
], dtype=np.float32)

CORNER_PROB = 0.20   # 20% of episodes → pure corner  (6.7% each)


class CAPQLEnv(gym.Env):
    """
    Gymnasium wrapper for CAPQL v2 (with corner injection).

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
        self._max_anfer = 200.0
        self._max_amir  =  50.0

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        # Corner injection: 20% pure corners, 80% Dirichlet
        r = np.random.random()
        if r < CORNER_PROB:
            corner_idx = int(r / (CORNER_PROB / 3))          # 0, 1, or 2
            corner_idx = min(corner_idx, 2)
            self.current_w = _CORNERS[corner_idx].copy()
        else:
            self.current_w = np.random.dirichlet(
                np.ones(N_OBJECTIVES)
            ).astype(np.float32)

        # Per-episode action caps driven by preference (squared law):
        #   preference=0.0 → full range unchanged
        #   preference=0.5 → 25% of range
        #   preference=1.0 → hard min (agronomically viable floor)
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
