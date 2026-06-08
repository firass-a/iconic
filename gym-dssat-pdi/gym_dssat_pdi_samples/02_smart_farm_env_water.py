"""
SmartFarmSoSEnv_Water — reward design tuned for WATER MINIMISATION.

Objective: reduce total irrigation while maintaining acceptable yield.

Key changes vs base reward:
  Daily   — W_WATER    0.40→0.20 (less reward for chasing soil moisture)
           — W_RESOURCE 0.15→0.35 (stronger per-step irrigation cost)
  Seasonal— IRRIG_EXCESS 500→150 mm (penalty maxes out at 150 mm)
           — R_penalty split 85% water / 15% N (was 50/50)
           — W_PENALTY 0.20→0.35 (stronger seasonal water penalty)
           — W_YIELD   0.40→0.25 (yield sacrifice accepted)
           — W_ANE     0.30 (restored to base — prevents N side effect)
"""
import gym
import numpy as np
from collections import OrderedDict


class SmartFarmSoSEnv:

    BASELINE_YIELD     = 7620.0
    MAX_EXPECTED_YIELD = 11430.0

    BASELINE_HIAD = 0.45
    MAX_HIAD      = 0.60

    BASELINE_ANE = 40.0
    MAX_ANE      = 80.0

    MR_LOW  = 0.40
    MR_HIGH = 0.80

    TRNU_MAX = 3.0

    RUNOFF_MAX = 20.0
    LEACH_MAX  =  5.0
    DENITR_MAX =  2.0

    IRRIG_EXCESS = 150.0   # penalty maxes out at 150 mm total irrigation
    N_EXCESS     = 200.0   # unchanged — not targeting N

    DAILY_SCALE = 0.01

    # Daily weights — water minimisation focus
    W_WATER    = 0.20   # ↓ agent tolerates slight dryness
    W_FERT     = 0.30   # unchanged
    W_RESOURCE = 0.35   # ↑ stronger per-step irrigation cost
    W_LOSSES   = 0.15

    # Seasonal weights
    W_YIELD   = 0.25   # ↓ accept yield sacrifice
    W_HIAD    = 0.10
    W_ANE     = 0.30   # restored to base — prevents N side effect
    W_PENALTY = 0.35   # ↑ stronger seasonal penalty

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

        daily    = self._compute_daily_reward(crop_obs_for_reward, anfer, amir)
        seasonal = self._compute_seasonal_reward(crop_obs_for_reward) if done else {}

        R_daily = (
              self.W_WATER    * daily['R_water']
            + self.W_FERT     * daily['R_fert']
            - self.W_RESOURCE * daily['R_resource']
            - self.W_LOSSES   * daily['R_losses']
        ) * self.DAILY_SCALE

        R_seasonal = 0.0
        if done:
            R_seasonal = (
                  self.W_YIELD   * seasonal['R_yield']
                + self.W_HIAD    * seasonal['R_hiad']
                + self.W_ANE     * seasonal['R_ane']
                - self.W_PENALTY * seasonal['R_penalty']
            )

        scalar_reward = float(R_daily + R_seasonal)
        merged_obs    = self._build_observation(crop_obs)

        components = {
            **{k: float(v) for k, v in daily.items()},
            'R_daily':    float(R_daily),
            'R_seasonal': float(R_seasonal),
            'reward':     scalar_reward,
        }
        if seasonal:
            components.update({k: float(v) for k, v in seasonal.items()})

        info['reward_components'] = components
        info['sos_state'] = {
            'sensor_health':  self.sensor_health.copy(),
            'energy_budget':  self.energy_budget,
            'comm_quality':   self.comm_quality,
            'total_nitrogen': self.total_nitrogen,
            'total_water':    self.total_water,
            'total_rain':     self.total_rain,
            'grnwt': float(crop_obs_for_reward.get('grnwt', 0.0) or 0.0),
        }
        return merged_obs, scalar_reward, done, info

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

    def _compute_seasonal_reward(self, crop_obs):
        grnwt = float(crop_obs.get('grnwt', 0.0) or 0.0)
        topwt = float(crop_obs.get('topwt', 0.0) or 0.0)

        R_yield = float(np.clip(
            (grnwt - self.BASELINE_YIELD) / (self.MAX_EXPECTED_YIELD - self.BASELINE_YIELD),
            -1.0, 1.0
        ))

        hiad = float(crop_obs.get('hiad', 0.0) or 0.0)
        if hiad == 0.0 and topwt > 0.0:
            hiad = grnwt / topwt
        R_hiad = float(np.clip(
            (hiad - self.BASELINE_HIAD) / (self.MAX_HIAD - self.BASELINE_HIAD),
            -1.0, 1.0
        ))

        ane   = grnwt / max(self.total_nitrogen, 1.0)
        R_ane = float(np.clip(
            (ane - self.BASELINE_ANE) / (self.MAX_ANE - self.BASELINE_ANE),
            -1.0, 1.0
        ))

        r_irrig_pen = float(np.clip(self.total_water    / self.IRRIG_EXCESS, 0.0, 1.0))
        r_n_pen     = float(np.clip(self.total_nitrogen / self.N_EXCESS,     0.0, 1.0))
        # Water-min: 70% weight on irrigation penalty, 30% on N penalty
        R_penalty   = 0.85 * r_irrig_pen + 0.15 * r_n_pen

        return {'R_yield': R_yield, 'R_hiad': R_hiad,
                'R_ane': R_ane, 'R_penalty': R_penalty}

    def _moisture_ratio(self, crop_obs):
        sw_raw = crop_obs.get('sw', None)
        sw = np.array(sw_raw if sw_raw is not None else [], dtype=np.float32)
        if len(sw) == 0 or self._ll is None or self._dul is None or self._dlayr is None:
            return None
        n     = min(len(sw), len(self._ll), len(self._dul), len(self._dlayr))
        sw    = sw[:n];  ll = self._ll[:n]; dul = self._dul[:n]; dlayr = self._dlayr[:n]
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
