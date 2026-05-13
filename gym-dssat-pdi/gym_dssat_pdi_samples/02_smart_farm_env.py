"""
SmartFarmSoSEnv — 3-objective MORL wrapper for gym-DSSAT.

This wrapper exposes a vector-valued reward suitable for multi-objective
reinforcement learning over agricultural decision-making:

    R[0] = R_yield   — calibrated final grain weight + small per-step shaping
    R[1] = R_wue     — seasonal grain-per-mm-of-total-water (rain + irrig)
    R[2] = R_energy  — 1 − season_energy / baseline_energy

NOTE on the dropped 4th objective. An earlier version included R_resilience
based on sensor health, comm quality, and energy budget. In our wrapper those
SoS state variables are exogenous stochastic processes — the agent has no
actions that affect their dynamics — so R_resilience contributed only
uncontrollable noise and did not produce meaningful preference-conditioned
behavior. We restrict the present formulation to the three CONTROLLABLE
objectives and document SoS-controllable actions as future work.

The wrapper still injects sensor faults and exposes sensor health in the
observation when `enable_faults=True`, so that experiments on the effect of
SoS degradation on yield/water/energy trade-offs remain possible — but
those SoS quantities no longer contribute to the reward.

Run inside Docker: python3 /workspace/02_smart_farm_env.py
"""
import gym
import numpy as np
from collections import OrderedDict


class SmartFarmSoSEnv:
    """
    Wraps gym-DSSAT with System-of-Systems modeling.

    Reward is a VECTOR of shape (3,): [R_yield, R_wue, R_energy]. The
    observation merges DSSAT crop state with SoS state (sensor health,
    energy budget, comm quality). Fault injection optionally perturbs the
    SoS state without modifying the reward formula.
    """

    # ============================================================
    # Reward calibration constants
    # ============================================================
    # Anchored to the Stage-based rule-based baseline reported earlier:
    #   yield ≈ 7620 kg/ha, irrigation ≈ 398 mm, N ≈ 150 kg/ha
    #   plus mean rainfall ≈ 250 mm/season for gym-DSSAT's default scenario
    BASELINE_YIELD              = 7620.0    # kg/ha at the neutral point
    MAX_EXPECTED_YIELD          = 1.5 * BASELINE_YIELD            # 11430
    BASELINE_WUE_KG_PER_MM      = 10.0      # ≈ 7620 / (398 + 250)
    MAX_EXPECTED_WUE_KG_PER_MM  = 25.0      # excellent agronomic WUE
    SYSTEM_BASELINE_KWH_DAY     = 0.5       # background load (sensors etc.)
    SEASON_DAYS                 = 161
    # baseline_energy ≈ 398*0.3 + 150*0.05 + 161*0.5 ≈ 207 kWh
    BASELINE_SEASON_ENERGY      = 207.0
    PER_STEP_YIELD_COEF         = 0.1       # small shaping for credit assignment

    # ============================================================
    # Construction / reset
    # ============================================================
    def __init__(self, mode='fertilization', seed=None,
                 enable_faults=False, fault_rate=0.02,
                 n_sensors=5, initial_energy=100.0):
        env_args = {'mode': mode}
        if seed is not None:
            env_args['seed'] = seed
        self.env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)

        self.n_sensors = n_sensors
        self.initial_energy = initial_energy
        self.enable_faults = enable_faults
        self.fault_rate = fault_rate

        # State (filled by reset)
        self.sensor_health = None
        self.energy_budget = None
        self.comm_quality = None
        self.prev_biomass = None
        self.total_water = None
        self.total_rain = None
        self.total_nitrogen = None
        self.day = None
        self.done = False
        self._last_crop_obs = {}

    def reset(self):
        """Reset env and SoS state."""
        obs = self.env.reset()
        if obs is None:
            obs = {}
        self._last_crop_obs = obs

        self.sensor_health = np.ones(self.n_sensors)
        self.energy_budget = self.initial_energy
        self.comm_quality = 1.0
        self.prev_biomass = obs.get('topwt', 0.0)
        self.total_water = 0.0       # cumulative irrigation (mm)
        self.total_rain = 0.0        # cumulative rainfall (mm), for WUE denominator
        self.total_nitrogen = 0.0
        self.day = 0
        self.done = False

        return self._build_observation(obs)

    # ============================================================
    # Step
    # ============================================================
    def step(self, action_dict):
        """One day step. action_dict has 'anfer' and/or 'amir'.

        Returns (merged_obs, reward_vector(3,), done, info).
        """
        # ---- 1. Underlying simulator ----
        crop_obs, default_reward, done, info = self.env.step(action_dict)
        if info is None:
            info = {}
        # gym-DSSAT returns None on the terminal step; fall back to the
        # most recent valid observation so the terminal yield can be read.
        if crop_obs is None or len(crop_obs) == 0:
            crop_obs = {}
            crop_obs_for_reward = self._last_crop_obs
        else:
            self._last_crop_obs = crop_obs
            crop_obs_for_reward = crop_obs
        self.done = done
        self.day += 1

        # ---- 2. Track resource usage ----
        nitrogen_applied = float(action_dict.get('anfer', 0.0))
        water_applied    = float(action_dict.get('amir',  0.0))
        rain_today       = float(crop_obs.get('rain', 0.0) or 0.0)
        self.total_nitrogen += nitrogen_applied
        self.total_water    += water_applied
        self.total_rain     += max(rain_today, 0.0)

        # ---- 3. Energy book-keeping (still used by SoS observation) ----
        sensor_drain   = float(np.sum(self.sensor_health)) * 0.1
        pump_cost      = water_applied * 0.3
        fert_cost      = nitrogen_applied * 0.05
        daily_cost     = sensor_drain + pump_cost + fert_cost
        solar_recharge = 2.0
        self.energy_budget = max(0.0, min(100.0,
            self.energy_budget - daily_cost + solar_recharge))

        # ---- 4. Fault injection (perturbs SoS state, not the reward) ----
        if self.enable_faults:
            self._inject_faults(crop_obs)

        # ---- 5. Reward vector ----
        reward_vector = self._compute_rewards(
            crop_obs=crop_obs_for_reward,
            done=done,
        )

        # ---- 6. Merged observation ----
        merged_obs = self._build_observation(crop_obs)

        # ---- 7. Info dict ----
        info['reward_components'] = {
            'R_yield':  float(reward_vector[0]),
            'R_wue':    float(reward_vector[1]),
            'R_energy': float(reward_vector[2]),
        }
        info['sos_state'] = {
            'sensor_health':  self.sensor_health.copy(),
            'energy_budget':  self.energy_budget,
            'comm_quality':   self.comm_quality,
            'total_nitrogen': self.total_nitrogen,
            'total_water':    self.total_water,
            'total_rain':     self.total_rain,
            'grnwt':          float(crop_obs_for_reward.get('grnwt', 0.0) or 0.0),
        }
        info['default_scalar_reward'] = default_reward

        return merged_obs, reward_vector, done, info

    # ============================================================
    # Reward function
    # ============================================================
    def _compute_rewards(self, crop_obs, done):
        """Three-component reward vector, all in approximately [-1, +1].

        R_yield  : small per-step biomass-gain shaping (coef 0.1) + a
                   terminal bonus driven by final grain weight, calibrated
                   such that grnwt = BASELINE_YIELD gives 0 and 1.5x gives +1.
                   Small shaping is potential-based and does not bias the
                   optimal policy (Ng et al. 1999); it provides daily credit
                   signal so PPO can learn before the terminal reward.

        R_wue    : zero per step, then at episode end:
                       wue = grnwt / (irrigation + rainfall)
                       R_wue = clip((wue - 10) / (25 - 10), -1, 1)
                   Rainfall in denominator avoids the "infinite WUE by being
                   rain-fed" pathology of per-step proxies.

        R_energy : zero per step, then at episode end:
                       energy = 0.3*irrig + 0.05*N + 161*0.5
                       R_energy = clip(1 - energy / BASELINE_SEASON_ENERGY, -1, 1)
                   At zero use -> +1; at baseline -> 0; at 2x baseline -> -1.
        """
        # ---- R_yield ----
        current_biomass = float(crop_obs.get('topwt', 0.0) or 0.0)
        biomass_gain = current_biomass - self.prev_biomass
        self.prev_biomass = current_biomass
        R_yield = self.PER_STEP_YIELD_COEF * float(
            np.clip(biomass_gain / 150.0, -1.0, 1.0)
        )
        if done:
            grnwt = float(crop_obs.get('grnwt', 0.0) or 0.0)
            yield_norm = (grnwt - self.BASELINE_YIELD) / (
                self.MAX_EXPECTED_YIELD - self.BASELINE_YIELD
            )
            R_yield += float(np.clip(yield_norm, -1.0, 1.0))

        # ---- R_wue ----
        if done:
            grnwt = float(crop_obs.get('grnwt', 0.0) or 0.0)
            total_water_used = self.total_water + self.total_rain
            if total_water_used > 1.0:
                wue_kg_per_mm = grnwt / total_water_used
                wue_norm = (
                    (wue_kg_per_mm - self.BASELINE_WUE_KG_PER_MM)
                    / (self.MAX_EXPECTED_WUE_KG_PER_MM - self.BASELINE_WUE_KG_PER_MM)
                )
                R_wue = float(np.clip(wue_norm, -1.0, 1.0))
            else:
                R_wue = 0.0
        else:
            R_wue = 0.0

        # ---- R_energy ----
        if done:
            season_energy = (
                self.total_water    * 0.3
                + self.total_nitrogen * 0.05
                + self.SEASON_DAYS  * self.SYSTEM_BASELINE_KWH_DAY
            )
            energy_ratio = season_energy / self.BASELINE_SEASON_ENERGY
            R_energy = float(np.clip(1.0 - energy_ratio, -1.0, 1.0))
        else:
            R_energy = 0.0

        return np.array([R_yield, R_wue, R_energy], dtype=np.float32)

    # ============================================================
    # Fault injection (SoS state perturbations)
    # ============================================================
    def _inject_faults(self, crop_obs):
        """Stochastic sensor dropouts and comm-quality fluctuations.

        These no longer contribute to the reward (R_resilience was removed)
        but still perturb the observation so that fault-aware analyses are
        possible without retraining.
        """
        for i in range(self.n_sensors):
            if self.sensor_health[i] == 1.0 and np.random.random() < self.fault_rate:
                self.sensor_health[i] = 0.0
        for i in range(self.n_sensors):
            if self.sensor_health[i] == 0.0 and np.random.random() < 0.01:
                self.sensor_health[i] = 1.0
        comm_noise = np.random.normal(0, 0.05)
        self.comm_quality = float(np.clip(self.comm_quality + comm_noise, 0.3, 1.0))
        if np.random.random() < 0.005:
            self.comm_quality = 0.3

    # ============================================================
    # Observation
    # ============================================================
    def _build_observation(self, crop_obs):
        """Merge crop state and SoS state into a single OrderedDict."""
        merged = OrderedDict()
        for key, value in crop_obs.items():
            merged[f'crop_{key}'] = value
        for i in range(self.n_sensors):
            merged[f'sensor_{i}'] = self.sensor_health[i]
        merged['energy_budget'] = self.energy_budget / 100.0
        merged['comm_quality']  = self.comm_quality
        return merged

    # ============================================================
    # Scalarization helpers (3-D preference simplex)
    # ============================================================
    def scalarize_reward(self, reward_vector, weights=None):
        """Linear scalarization. Default weights yield-dominant trade-off."""
        if weights is None:
            weights = np.array([0.55, 0.30, 0.15])
        return float(np.dot(weights, reward_vector))

    def chebyshev_scalarize(self, reward_vector, weights=None, ideal=None):
        """Chebyshev scalarization. Can recover non-convex Pareto solutions."""
        if weights is None:
            weights = np.array([0.55, 0.30, 0.15])
        if ideal is None:
            ideal = np.array([1.0, 1.0, 1.0])
        weighted_dist = weights * np.abs(reward_vector - ideal)
        return -float(np.max(weighted_dist))

    def close(self):
        self.env.close()


# ============================================================
# DEMO
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SmartFarmSoSEnv DEMO — 3-objective MORL rewards")
    print("=" * 70)

    env = SmartFarmSoSEnv(mode='all', seed=42, enable_faults=False)
    obs = env.reset()
    print(f"\nInitial obs keys: {list(obs.keys())[:6]}... ({len(obs)} total)")

    total = np.zeros(3)
    for day in range(1, 162):
        # crude rule for the demo: apply 40 kg N on days 35/65/95, 10 mm water every 3 days
        anfer = 40.0 if day in (35, 65, 95) else 0.0
        amir  = 10.0 if day % 3 == 0 else 0.0
        obs, R, done, info = env.step({'anfer': anfer, 'amir': amir})
        total += R
        if done:
            sos = info['sos_state']
            print(f"\nEpisode end at day {day}:")
            print(f"  cumulative R     = {total.round(3)}")
            print(f"  scalar (linear)  = {env.scalarize_reward(R):+.3f}  (terminal step)")
            print(f"  scalar (cheby)   = {env.chebyshev_scalarize(R):+.3f}")
            print(f"  yield            = {sos['grnwt']:.0f} kg/ha")
            print(f"  total irrigation = {sos['total_water']:.0f} mm")
            print(f"  total rainfall   = {sos['total_rain']:.0f} mm")
            print(f"  total N          = {sos['total_nitrogen']:.0f} kg/ha")
            break

    env.close()
    print("\nNext: re-run baselines, then re-train PC-PPO with pc_env updated to 3 objectives.")
