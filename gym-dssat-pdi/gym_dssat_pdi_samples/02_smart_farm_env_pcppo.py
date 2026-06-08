"""
SmartFarmSoSEnv_PCPPO — environment for Preference-Conditioned PPO.

Differences from base SmartFarmSoSEnv:
  - step() returns ONLY the daily shaping reward as scalar
  - At terminal step, info['reward_vec'] = [R_yield, R_ane, R_water_eff]
  - The preference-weighted scalarization of seasonal reward happens in
    the pc_env_pcppo.py adapter, NOT here

Reward vector (3 objectives, all ∈ [-1, +1], higher = better):
  R_yield      : grain yield relative to baseline (7620 kg/ha)
  R_ane        : N efficiency (grnwt / total_N), with yield floor at 5000 kg/ha
  R_water_eff  : water efficiency — reward for using LESS irrigation

Corner preferences map to:
  w=[1,0,0] → maximize yield       (similar to base PPO)
  w=[0,1,0] → maximize N efficiency (similar to N-min PPO)
  w=[0,0,1] → minimize water use    (similar to water-min PPO)
"""
import gym
import numpy as np
from collections import OrderedDict


class SmartFarmSoSEnv:

    # ------------------------------------------------------------------ #
    # Yield                                                                #
    # ------------------------------------------------------------------ #
    BASELINE_YIELD     = 7620.0
    MAX_EXPECTED_YIELD = 11430.0

    # ------------------------------------------------------------------ #
    # Harvest index                                                        #
    # ------------------------------------------------------------------ #
    BASELINE_HIAD = 0.45
    MAX_HIAD      = 0.60

    # ------------------------------------------------------------------ #
    # N efficiency                                                         #
    # ------------------------------------------------------------------ #
    BASELINE_ANE    = 40.0
    MAX_ANE         = 80.0
    MIN_VIABLE_YIELD = 5000.0   # below this → R_ane = -1 (crop failure)

    # ------------------------------------------------------------------ #
    # Water efficiency target                                              #
    # ------------------------------------------------------------------ #
    WATER_TARGET = 400.0   # mm: above → R_water_eff negative

    # ------------------------------------------------------------------ #
    # Moisture ratio                                                       #
    # ------------------------------------------------------------------ #
    MR_LOW  = 0.40
    MR_HIGH = 0.80

    # ------------------------------------------------------------------ #
    # Daily reward constants                                               #
    # ------------------------------------------------------------------ #
    TRNU_MAX   = 3.0
    RUNOFF_MAX = 20.0
    LEACH_MAX  =  5.0
    DENITR_MAX =  2.0
    DAILY_SCALE = 0.01

    # Daily weights — balanced base (same as base PPO)
    W_WATER    = 0.40
    W_FERT     = 0.30
    W_RESOURCE = 0.15
    W_LOSSES   = 0.15

    def __init__(self, mode='all', seed=None,
                 run_dssat_location='run_dssat',
                 enable_faults=False, fault_rate=0.02,
                 n_sensors=5, initial_energy=100.0):
        env_args = {
            'mode':                mode,
            'run_dssat_location':  run_dssat_location,
            'random_weather':      True,
        }
        if seed is not None:
            env_args['seed'] = seed
        self.env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)

        self.n_sensors      = n_sensors
        self.initial_energy = initial_energy
        self.enable_faults  = enable_faults
        self.fault_rate     = fault_rate

        self.sensor_health  = None
        self.energy_budget  = None
        self.comm_quality   = None
        self.total_water    = None
        self.total_rain     = None
        self.total_nitrogen = None
        self.day            = None
        self.done           = False
        self._last_crop_obs = {}

        self._ll    = None
        self._dul   = None
        self._dlayr = None

    def reset(self):
        obs = self.env.reset()
        if obs is None:
            obs = {}
        self._last_crop_obs = obs
        self._ll    = None
        self._dul   = None
        self._dlayr = None
        self.sensor_health  = np.ones(self.n_sensors, dtype=np.float32)
        self.energy_budget  = float(self.initial_energy)
        self.comm_quality   = 1.0
        self.total_water    = 0.0
        self.total_rain     = 0.0
        self.total_nitrogen = 0.0
        self.day            = 0
        self.done           = False
        return self._build_observation(obs)

    def step(self, action_dict):
        """
        Returns scalar = R_daily only (no seasonal).
        At terminal step, info['reward_vec'] = [R_yield, R_ane, R_water_eff].
        The adapter pc_env_pcppo.py adds w·reward_vec to get total reward.
        """
        crop_obs, _, done, info = self.env.step(action_dict)
        if info is None:
            info = {}
        if crop_obs is None or len(crop_obs) == 0:
            crop_obs = {}
            crop_obs_for_reward = self._last_crop_obs
        else:
            self._last_crop_obs = crop_obs
            crop_obs_for_reward = crop_obs

        self.done  = done
        self.day  += 1

        if self._ll is None and info:
            self._cache_soil_profile(info)

        anfer      = float(action_dict.get('anfer', 0.0))
        amir       = float(action_dict.get('amir',  0.0))
        rain_today = float(crop_obs.get('rain', 0.0) or 0.0)
        self.total_nitrogen += anfer
        self.total_water    += amir
        self.total_rain     += max(rain_today, 0.0)

        sensor_drain = float(np.sum(self.sensor_health)) * 0.1
        daily_cost   = sensor_drain + amir * 0.3 + anfer * 0.05
        self.energy_budget = float(np.clip(
            self.energy_budget - daily_cost + 2.0, 0.0, 100.0
        ))

        if self.enable_faults:
            self._inject_faults()

        # Daily shaping (fixed balanced weights)
        daily   = self._compute_daily_reward(crop_obs_for_reward, anfer, amir)
        R_daily = (
              self.W_WATER    * daily['R_water']
            + self.W_FERT     * daily['R_fert']
            - self.W_RESOURCE * daily['R_resource']
            - self.W_LOSSES   * daily['R_losses']
        ) * self.DAILY_SCALE

        # Seasonal reward vector (3 objectives) — only at terminal step
        reward_vec = np.zeros(3, dtype=np.float32)
        if done:
            reward_vec = self._compute_reward_vec(crop_obs_for_reward)

        merged_obs = self._build_observation(crop_obs)

        info['reward_vec']    = reward_vec          # [R_yield, R_ane, R_water_eff]
        info['R_daily']       = float(R_daily)
        info['daily_components'] = daily
        info['sos_state'] = {
            'sensor_health':  self.sensor_health.copy(),
            'energy_budget':  self.energy_budget,
            'comm_quality':   self.comm_quality,
            'total_nitrogen': self.total_nitrogen,
            'total_water':    self.total_water,
            'total_rain':     self.total_rain,
            'grnwt': float(crop_obs_for_reward.get('grnwt', 0.0) or 0.0),
        }

        return merged_obs, R_daily, done, info

    # ================================================================== #
    # Seasonal reward vector                                               #
    # ================================================================== #
    def _compute_reward_vec(self, crop_obs):
        grnwt = float(crop_obs.get('grnwt', 0.0) or 0.0)

        # R_yield ∈ [-1, +1]
        R_yield = float(np.clip(
            (grnwt - self.BASELINE_YIELD) / (self.MAX_EXPECTED_YIELD - self.BASELINE_YIELD),
            -1.0, 1.0
        ))

        # R_ane ∈ [-1, +1] with yield floor (blocks zero-N exploit)
        if grnwt >= self.MIN_VIABLE_YIELD:
            ane   = grnwt / max(self.total_nitrogen, 1.0)
            R_ane = float(np.clip(
                (ane - self.BASELINE_ANE) / (self.MAX_ANE - self.BASELINE_ANE),
                -1.0, 1.0
            ))
        else:
            R_ane = -1.0

        # R_water_eff ∈ [-1, +1]: reward for using LESS total irrigation
        R_water_eff = float(np.clip(
            (self.WATER_TARGET - self.total_water) / self.WATER_TARGET,
            -1.0, 1.0
        ))

        return np.array([R_yield, R_ane, R_water_eff], dtype=np.float32)

    # ================================================================== #
    # Daily reward components                                              #
    # ================================================================== #
    def _compute_daily_reward(self, crop_obs, anfer, amir):
        swfac   = float(crop_obs.get('swfac', 1.0) or 1.0)
        R_swfac = 2.0 * swfac - 1.0

        mr = self._moisture_ratio(crop_obs)
        if mr is not None:
            if self.MR_LOW <= mr <= self.MR_HIGH:
                R_mr = 1.0
            elif mr < self.MR_LOW:
                R_mr = float(np.clip(2.0 * mr / self.MR_LOW - 1.0, -1.0, 1.0))
            else:
                over = (mr - self.MR_HIGH) / max(1.0 - self.MR_HIGH, 1e-6)
                R_mr = float(np.clip(1.0 - 2.0 * over, -1.0, 1.0))
            R_water = 0.60 * R_swfac + 0.40 * R_mr
        else:
            R_water = R_swfac

        nstres   = float(crop_obs.get('nstres', 1.0) or 1.0)
        trnu     = float(crop_obs.get('trnu',   0.0) or 0.0)
        R_nstres = 2.0 * nstres - 1.0
        R_trnu   = float(np.clip(trnu / self.TRNU_MAX, 0.0, 1.0))
        R_fert   = 0.60 * R_nstres + 0.40 * R_trnu

        r_irrig    = float(np.clip(amir  / 50.0,  0.0, 1.0))
        r_fert_use = float(np.clip(anfer / 200.0, 0.0, 1.0))
        R_resource = 0.50 * r_irrig + 0.50 * r_fert_use

        runoff = float(crop_obs.get('runoff',  0.0) or 0.0)
        leacn  = float(crop_obs.get('tleachd', 0.0) or 0.0)
        denitr = float(crop_obs.get('tnoxd',   0.0) or 0.0)
        R_losses = (
            0.40 * float(np.clip(runoff / self.RUNOFF_MAX, 0.0, 1.0))
          + 0.35 * float(np.clip(leacn  / self.LEACH_MAX,  0.0, 1.0))
          + 0.25 * float(np.clip(denitr / self.DENITR_MAX, 0.0, 1.0))
        )

        return {
            'R_water':    float(np.clip(R_water, -1.0, 1.0)),
            'R_fert':     float(np.clip(R_fert,  -1.0, 1.0)),
            'R_resource': float(R_resource),
            'R_losses':   float(R_losses),
        }

    def _moisture_ratio(self, crop_obs):
        sw_raw = crop_obs.get('sw', None)
        sw = np.array(sw_raw if sw_raw is not None else [], dtype=np.float32)
        if len(sw) == 0 or self._ll is None or self._dul is None or self._dlayr is None:
            return None
        n     = min(len(sw), len(self._ll), len(self._dul), len(self._dlayr))
        sw    = sw[:n]; ll = self._ll[:n]; dul = self._dul[:n]; dlayr = self._dlayr[:n]
        swxd  = float(np.sum(np.maximum(sw - ll,  0.0) * dlayr))
        swtd  = float(np.sum(np.maximum(dul - ll, 0.0) * dlayr))
        return float(np.clip(swxd / swtd, 0.0, 1.0)) if swtd >= 1e-6 else None

    def _cache_soil_profile(self, info):
        ll = info.get('ll', None); dul = info.get('dul', None); dlayr = info.get('dlayr', None)
        if ll is not None and dul is not None and dlayr is not None:
            self._ll = np.array(ll, dtype=np.float32)
            self._dul = np.array(dul, dtype=np.float32)
            self._dlayr = np.array(dlayr, dtype=np.float32)

    def _inject_faults(self):
        for i in range(self.n_sensors):
            if self.sensor_health[i] == 1.0 and np.random.random() < self.fault_rate:
                self.sensor_health[i] = 0.0
            elif self.sensor_health[i] == 0.0 and np.random.random() < 0.01:
                self.sensor_health[i] = 1.0
        self.comm_quality = float(np.clip(
            self.comm_quality + np.random.normal(0.0, 0.05), 0.3, 1.0
        ))

    def _build_observation(self, crop_obs):
        merged = OrderedDict()
        for key, value in crop_obs.items():
            merged[f'crop_{key}'] = value
        for i in range(self.n_sensors):
            merged[f'sensor_{i}'] = float(self.sensor_health[i])
        merged['energy_budget'] = self.energy_budget / 100.0
        merged['comm_quality']  = self.comm_quality
        return merged

    def close(self):
        self.env.close()
