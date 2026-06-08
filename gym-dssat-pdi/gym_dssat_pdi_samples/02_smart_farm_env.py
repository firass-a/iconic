"""
SmartFarmSoSEnv — hierarchical daily + seasonal scalar reward for PPO.

Reward operates on two time scales:

  Daily shaping (every step):
      R_daily = (0.40 * R_water
               + 0.30 * R_fert
               - 0.15 * R_resource
               - 0.15 * R_losses) * DAILY_SCALE

  Seasonal evaluation (terminal step only):
      R_seasonal = (0.60 * R_yield
                  + 0.15 * R_hiad
                  + 0.15 * R_ane
                  - 0.10 * R_penalty)

  Step reward:
      R_t = R_daily_t
      R_T = R_daily_T + R_seasonal      (harvest day only)

DAILY_SCALE = 0.01 keeps cumulative daily (~0.5 max) below seasonal (~1.0),
so seasonal objectives dominate as required by the design spec.

All sub-rewards are clipped to [-1, +1] or [0, 1] before weighting.
Full component breakdown is logged in info['reward_components'] every step.

Run inside Docker: python3 /workspace/gym-dssat-pdi/gym_dssat_pdi_samples/02_smart_farm_env.py
"""
import gym
import numpy as np
from collections import OrderedDict


class SmartFarmSoSEnv:
    """
    Wraps gym-DSSAT with a hierarchical scalar reward for simple PPO.

    The observation merges DSSAT crop state with SoS state (sensor health,
    energy budget, comm quality). Fault injection optionally perturbs the
    SoS observation variables without affecting the reward.
    """

    # ------------------------------------------------------------------ #
    # Yield calibration  (anchored to stage-based baseline: ~7620 kg/ha)  #
    # ------------------------------------------------------------------ #
    BASELINE_YIELD     = 7620.0
    MAX_EXPECTED_YIELD = 11430.0   # 1.5 × baseline

    # ------------------------------------------------------------------ #
    # Harvest index (maize: typical 0.45-0.55, excellent ~0.60)           #
    # ------------------------------------------------------------------ #
    BASELINE_HIAD = 0.45
    MAX_HIAD      = 0.60

    # ------------------------------------------------------------------ #
    # Agronomic N efficiency proxy: grnwt / total_N  (kg grain / kg N)   #
    # ------------------------------------------------------------------ #
    BASELINE_ANE = 40.0
    MAX_ANE      = 80.0

    # ------------------------------------------------------------------ #
    # Moisture ratio thresholds (SWXD/SWTD, unitless [0,1])               #
    # ------------------------------------------------------------------ #
    MR_LOW  = 0.40   # below → drought risk
    MR_HIGH = 0.80   # above → waterlogging risk

    # ------------------------------------------------------------------ #
    # N uptake normalization                                               #
    # ------------------------------------------------------------------ #
    TRNU_MAX = 3.0   # typical daily max (kg N/ha)

    # ------------------------------------------------------------------ #
    # Loss normalization (per-step typical maxima)                         #
    # ------------------------------------------------------------------ #
    RUNOFF_MAX = 20.0   # mm          (DSSAT: 'runoff')
    LEACH_MAX  =  5.0   # kg N/ha    (DSSAT: 'tleachd')
    DENITR_MAX =  2.0   # kg N/ha    (DSSAT: 'tnoxd')

    # ------------------------------------------------------------------ #
    # Seasonal penalty normalization                                       #
    # ------------------------------------------------------------------ #
    IRRIG_EXCESS = 500.0   # mm   — above this is clearly excessive
    N_EXCESS     = 200.0   # kg/ha

    # ------------------------------------------------------------------ #
    # Daily / seasonal balance                                             #
    # ------------------------------------------------------------------ #
    DAILY_SCALE = 0.01   # cumulative daily ≈ 0.5 << seasonal ≈ 1.0

    # ------------------------------------------------------------------ #
    # Daily component weights  (sum = 1.0)                                #
    # ------------------------------------------------------------------ #
    W_WATER    = 0.40   # water stress most critical for maize
    W_FERT     = 0.30   # N stress acts on slower timescale
    W_RESOURCE = 0.15   # moderate: must still be willing to act
    W_LOSSES   = 0.15   # equal: discourage waste symmetrically

    # ------------------------------------------------------------------ #
    # Seasonal component weights                                           #
    # ------------------------------------------------------------------ #
    W_YIELD   = 0.40   # primary objective — farmer income
    W_HIAD    = 0.10   # biomass-to-grain conversion efficiency
    W_ANE     = 0.30   # N efficiency — penalises over-fertilisation
    W_PENALTY = 0.20   # strong: discourages extreme over-application

    # ================================================================== #
    # Construction                                                         #
    # ================================================================== #
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

        # NASA POWER CLI injection disabled — pending format verification
        # cli_path = '/workspace/weather/UFGA.CLI'
        # if os.path.exists(cli_path):
        #     env_args['auxiliary_file_paths'] = [cli_path]

        self.env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)

        self.n_sensors      = n_sensors
        self.initial_energy = initial_energy
        self.enable_faults  = enable_faults
        self.fault_rate     = fault_rate

        # Episode accumulators — initialised by reset()
        self.sensor_health  = None
        self.energy_budget  = None
        self.comm_quality   = None
        self.total_water    = None
        self.total_rain     = None
        self.total_nitrogen = None
        self.day            = None
        self.done           = False
        self._last_crop_obs = {}

        # Soil profile for moisture ratio (set from context on first step)
        self._ll    = None   # lower limit per layer   (cm³/cm³)
        self._dul   = None   # drained upper limit     (cm³/cm³)
        self._dlayr = None   # layer thickness         (cm)

    # ================================================================== #
    # Reset                                                                #
    # ================================================================== #
    def reset(self):
        obs = self.env.reset()
        if obs is None:
            obs = {}
        self._last_crop_obs = obs

        # Soil profile — try to read from obs; finalised on first step info
        self._ll    = None
        self._dul   = None
        self._dlayr = None

        # SoS state
        self.sensor_health  = np.ones(self.n_sensors, dtype=np.float32)
        self.energy_budget  = float(self.initial_energy)
        self.comm_quality   = 1.0
        self.total_water    = 0.0
        self.total_rain     = 0.0
        self.total_nitrogen = 0.0
        self.day            = 0
        self.done           = False

        return self._build_observation(obs)

    # ================================================================== #
    # Step                                                                 #
    # ================================================================== #
    def step(self, action_dict):
        """
        One day step.

        Returns
        -------
        merged_obs   : OrderedDict  — crop state + SoS state
        scalar_reward: float        — combined daily (+ seasonal if done)
        done         : bool
        info         : dict         — includes 'reward_components', 'sos_state'
        """
        crop_obs, _, done, info = self.env.step(action_dict)
        if info is None:
            info = {}

        # gym-DSSAT returns None obs on the terminal step
        if crop_obs is None or len(crop_obs) == 0:
            crop_obs = {}
            crop_obs_for_reward = self._last_crop_obs
        else:
            self._last_crop_obs = crop_obs
            crop_obs_for_reward = crop_obs

        self.done  = done
        self.day  += 1

        # Cache soil profile from context (info dict) on first step
        if self._ll is None and info:
            self._cache_soil_profile(info)

        # ---- Resource tracking ----
        anfer      = float(action_dict.get('anfer', 0.0))
        amir       = float(action_dict.get('amir',  0.0))
        rain_today = float(crop_obs.get('rain', 0.0) or 0.0)
        self.total_nitrogen += anfer
        self.total_water    += amir
        self.total_rain     += max(rain_today, 0.0)

        # ---- Energy bookkeeping (SoS observation, NOT in reward) ----
        sensor_drain = float(np.sum(self.sensor_health)) * 0.1
        daily_cost   = sensor_drain + amir * 0.3 + anfer * 0.05
        self.energy_budget = float(np.clip(
            self.energy_budget - daily_cost + 2.0, 0.0, 100.0
        ))

        # ---- Fault injection (SoS obs only) ----
        if self.enable_faults:
            self._inject_faults()

        # ---- Reward ----
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

        # ---- Observation ----
        merged_obs = self._build_observation(crop_obs)

        # ---- Info ----
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

    # ================================================================== #
    # Daily reward  (4 components, each ∈ [-1,+1] or [0,1])              #
    # ================================================================== #
    def _compute_daily_reward(self, crop_obs, anfer, amir):
        """
        R_water   : encourages healthy soil moisture and low water stress.
        R_fert    : encourages adequate N availability and uptake.
        R_resource: penalty proportional to resources applied today.
        R_losses  : penalty for runoff, drainage, leaching, denitrification.
        """

        # ---- R_water ----
        swfac   = float(crop_obs.get('swfac', 1.0) or 1.0)
        R_swfac = 2.0 * swfac - 1.0          # [0,1] → [-1, +1]

        mr = self._moisture_ratio(crop_obs)
        if mr is not None:
            if self.MR_LOW <= mr <= self.MR_HIGH:
                R_mr = 1.0                    # optimal band
            elif mr < self.MR_LOW:
                # linearly from -1 (bone dry) to +1 (lower threshold)
                R_mr = float(np.clip(
                    2.0 * mr / self.MR_LOW - 1.0, -1.0, 1.0
                ))
            else:
                # linearly from +1 (upper threshold) to -1 (fully saturated)
                over = (mr - self.MR_HIGH) / max(1.0 - self.MR_HIGH, 1e-6)
                R_mr = float(np.clip(1.0 - 2.0 * over, -1.0, 1.0))
            R_water = 0.60 * R_swfac + 0.40 * R_mr
        else:
            R_water = R_swfac               # fallback when profile unavailable

        # ---- R_fert ----
        nstres   = float(crop_obs.get('nstres', 1.0) or 1.0)
        trnu     = float(crop_obs.get('trnu',   0.0) or 0.0)
        R_nstres = 2.0 * nstres - 1.0        # [0,1] → [-1, +1]
        R_trnu   = float(np.clip(trnu / self.TRNU_MAX, 0.0, 1.0))
        R_fert   = 0.60 * R_nstres + 0.40 * R_trnu

        # ---- R_resource  (penalty ∈ [0,1]) ----
        r_irrig    = float(np.clip(amir  / 50.0,  0.0, 1.0))
        r_fert_use = float(np.clip(anfer / 200.0, 0.0, 1.0))
        R_resource = 0.50 * r_irrig + 0.50 * r_fert_use

        # ---- R_losses  (penalty ∈ [0,1]) ----
        # 'drain' is not exposed by gym-DSSAT — omitted.
        # DSSAT names: tleachd (daily leaching), tnoxd (daily denitrification)
        runoff = float(crop_obs.get('runoff', 0.0) or 0.0)
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

    # ================================================================== #
    # Seasonal reward  (4 components, terminal step only)                 #
    # ================================================================== #
    def _compute_seasonal_reward(self, crop_obs):
        """
        R_yield  : normalised grain weight relative to baseline.
        R_hiad   : harvest index (grain / total biomass).
        R_ane    : agronomic N efficiency proxy (grnwt / total_N).
        R_penalty: weak penalty for excessive seasonal resource use.
        """
        grnwt = float(crop_obs.get('grnwt', 0.0) or 0.0)
        topwt = float(crop_obs.get('topwt', 0.0) or 0.0)

        # ---- R_yield ----
        yield_norm = (grnwt - self.BASELINE_YIELD) / (
            self.MAX_EXPECTED_YIELD - self.BASELINE_YIELD
        )
        R_yield = float(np.clip(yield_norm, -1.0, 1.0))

        # ---- R_hiad ----
        hiad = float(crop_obs.get('hiad', 0.0) or 0.0)
        if hiad == 0.0 and topwt > 0.0:
            hiad = grnwt / topwt           # compute if not directly reported
        hiad_norm = (hiad - self.BASELINE_HIAD) / (
            self.MAX_HIAD - self.BASELINE_HIAD
        )
        R_hiad = float(np.clip(hiad_norm, -1.0, 1.0))

        # ---- R_ane  (proxy: grnwt / total_N_applied) ----
        ane      = grnwt / max(self.total_nitrogen, 1.0)
        ane_norm = (ane - self.BASELINE_ANE) / (self.MAX_ANE - self.BASELINE_ANE)
        R_ane    = float(np.clip(ane_norm, -1.0, 1.0))

        # ---- R_penalty  (∈ [0,1]) ----
        r_irrig_pen = float(np.clip(self.total_water    / self.IRRIG_EXCESS, 0.0, 1.0))
        r_n_pen     = float(np.clip(self.total_nitrogen / self.N_EXCESS,     0.0, 1.0))
        R_penalty   = 0.50 * r_irrig_pen + 0.50 * r_n_pen

        return {
            'R_yield':   R_yield,
            'R_hiad':    R_hiad,
            'R_ane':     R_ane,
            'R_penalty': R_penalty,
        }

    # ================================================================== #
    # Moisture ratio  MR = SWXD / SWTD                                   #
    # ================================================================== #
    def _moisture_ratio(self, crop_obs):
        """
        Compute MR = sum[(sw - ll) * dlayr] / sum[(dul - ll) * dlayr].

        Returns None when soil profile data has not yet been cached.
        SWXD (extractable water) / SWTD (total drainable) gives a
        normalised [0,1] index of how full the soil profile is.
        """
        sw_raw = crop_obs.get('sw', None)
        sw = np.array(sw_raw if sw_raw is not None else [], dtype=np.float32)
        if (
            len(sw) == 0
            or self._ll    is None
            or self._dul   is None
            or self._dlayr is None
        ):
            return None

        n     = min(len(sw), len(self._ll), len(self._dul), len(self._dlayr))
        sw    = sw[:n]
        ll    = self._ll[:n]
        dul   = self._dul[:n]
        dlayr = self._dlayr[:n]

        swxd = float(np.sum(np.maximum(sw - ll,  0.0) * dlayr))
        swtd = float(np.sum(np.maximum(dul - ll, 0.0) * dlayr))

        if swtd < 1e-6:
            return None
        return float(np.clip(swxd / swtd, 0.0, 1.0))

    def _cache_soil_profile(self, info):
        """Read ll, dul, dlayr from gym-DSSAT context dict (one-time per episode)."""
        ll    = info.get('ll',    None)
        dul   = info.get('dul',   None)
        dlayr = info.get('dlayr', None)
        if ll is not None and dul is not None and dlayr is not None:
            self._ll    = np.array(ll,    dtype=np.float32)
            self._dul   = np.array(dul,   dtype=np.float32)
            self._dlayr = np.array(dlayr, dtype=np.float32)

    # ================================================================== #
    # Fault injection  (SoS observation only — not in reward)             #
    # ================================================================== #
    def _inject_faults(self):
        for i in range(self.n_sensors):
            if self.sensor_health[i] == 1.0 and np.random.random() < self.fault_rate:
                self.sensor_health[i] = 0.0
            elif self.sensor_health[i] == 0.0 and np.random.random() < 0.01:
                self.sensor_health[i] = 1.0
        comm_noise = np.random.normal(0.0, 0.05)
        self.comm_quality = float(np.clip(self.comm_quality + comm_noise, 0.3, 1.0))
        if np.random.random() < 0.005:
            self.comm_quality = 0.3

    # ================================================================== #
    # Observation                                                          #
    # ================================================================== #
    def _build_observation(self, crop_obs):
        """Merge crop state and SoS state into a single OrderedDict."""
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


# ====================================================================== #
# Demo                                                                     #
# ====================================================================== #
if __name__ == '__main__':
    print("=" * 70)
    print("SmartFarmSoSEnv DEMO — hierarchical daily + seasonal reward")
    print("=" * 70)

    env = SmartFarmSoSEnv(mode='all', seed=42,
                          run_dssat_location='run_dssat',
                          enable_faults=False)
    obs = env.reset()
    print(f"\nInitial obs keys : {list(obs.keys())[:6]}... ({len(obs)} total)")

    cum_reward = 0.0
    for day in range(1, 162):
        anfer = 40.0 if day in (35, 65, 95) else 0.0
        amir  = 10.0 if day % 3 == 0 else 0.0
        obs, R, done, info = env.step({'anfer': anfer, 'amir': amir})
        cum_reward += R
        if done:
            c   = info['reward_components']
            sos = info['sos_state']
            print(f"\nEpisode end — day {day}")
            print(f"  cumulative reward : {cum_reward:+.4f}")
            print(f"  R_daily (last)    : {c['R_daily']:+.6f}")
            print(f"  R_seasonal        : {c['R_seasonal']:+.4f}")
            print(f"    R_yield  = {c['R_yield']:+.4f}  (grnwt = {sos['grnwt']:.0f} kg/ha)")
            print(f"    R_hiad   = {c['R_hiad']:+.4f}")
            print(f"    R_ane    = {c['R_ane']:+.4f}  (proxy = {sos['grnwt'] / max(sos['total_nitrogen'],1):.1f})")
            print(f"    R_penalty= {c['R_penalty']:+.4f}")
            print(f"  total irrigation  : {sos['total_water']:.0f} mm")
            print(f"  total rainfall    : {sos['total_rain']:.0f} mm")
            print(f"  total N applied   : {sos['total_nitrogen']:.0f} kg/ha")
            break

    env.close()
