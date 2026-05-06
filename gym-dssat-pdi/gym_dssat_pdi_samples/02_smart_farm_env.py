"""
Step 2: SmartFarmSoSEnv - Multi-Objective SoS Wrapper for gym-DSSAT
This is the CORE of your thesis. It wraps gym-DSSAT and adds:
  - Multi-reward vector (4 objectives instead of 1 scalar)
  - SoS state variables (sensor health, energy, comm quality)
  - Fault injection (sensor dropout, comm delay)
  - Energy model

Run inside Docker: python3 /workspace/02_smart_farm_env.py
"""
import gym
import numpy as np
from collections import OrderedDict


class SmartFarmSoSEnv:
    """
    Wraps gym-DSSAT with System-of-Systems modeling.

    Key changes from default gym-DSSAT:
    1. Reward is a VECTOR [R_yield, R_water, R_energy, R_resilience]
    2. Observation includes SoS state (sensor health, energy, etc.)
    3. Faults can be injected (sensor dropout, comm delay)
    4. Energy model tracks power consumption
    """

    def __init__(self, mode='fertilization', seed=None,
                 enable_faults=False, fault_rate=0.02,
                 n_sensors=5, initial_energy=100.0):

        # Create the underlying gym-DSSAT environment
        env_args = {'mode': mode}
        if seed is not None:
            env_args['seed'] = seed
        self.env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)

        # SoS parameters
        self.n_sensors = n_sensors
        self.initial_energy = initial_energy
        self.enable_faults = enable_faults
        self.fault_rate = fault_rate

        # SoS state (initialized in reset)
        self.sensor_health = None
        self.energy_budget = None
        self.comm_quality = None
        self.prev_biomass = None
        self.total_water = None
        self.total_nitrogen = None
        self.day = None
        self.done = False

    def reset(self):
        """Reset environment and SoS state."""
        obs = self.env.reset()

        # Reset SoS state
        self.sensor_health = np.ones(self.n_sensors)
        self.energy_budget = self.initial_energy
        self.comm_quality = 1.0
        self.prev_biomass = obs.get('topwt', 0.0)
        self.total_water = 0.0
        self.total_nitrogen = 0.0
        self.day = 0
        self.done = False

        return self._build_observation(obs)

    def step(self, action_dict):
        """
        Execute one day step.

        Args:
            action_dict: dict with keys depending on mode
                fertilization: {'anfer': float}
                irrigation: {'amir': float}
                all: {'anfer': float, 'amir': float}

        Returns:
            obs: merged observation (crop state + SoS state)
            reward_vector: np.array of shape (4,)
                [R_yield, R_nitrogen_eff, R_energy, R_resilience]
            done: bool
            info: dict with extra information
        """
        # ---- Step 1: Run DSSAT ----
        crop_obs, default_reward, done, info = self.env.step(action_dict)
        self.done = done
        self.day += 1

        # ---- Step 2: Track resource usage ----
        nitrogen_applied = action_dict.get('anfer', 0.0)
        water_applied = action_dict.get('amir', 0.0)
        self.total_nitrogen += nitrogen_applied
        self.total_water += water_applied

        # ---- Step 3: Update energy model ----
        sensor_drain = np.sum(self.sensor_health) * 0.1   # active sensors cost energy
        pump_cost = water_applied * 0.3                    # irrigation pumping
        fert_cost = nitrogen_applied * 0.05                # fertilizer application
        total_cost = sensor_drain + pump_cost + fert_cost

        solar_recharge = 2.0  # daily solar recharge
        self.energy_budget = max(0, min(100,
            self.energy_budget - total_cost + solar_recharge))

        # ---- Step 4: Inject faults (if enabled) ----
        if self.enable_faults:
            self._inject_faults(crop_obs)

        # ---- Step 5: Compute reward vector ----
        reward_vector = self._compute_rewards(crop_obs, nitrogen_applied,
                                                water_applied, total_cost)

        # ---- Step 6: Build merged observation ----
        merged_obs = self._build_observation(crop_obs)

        # ---- Step 7: Build info dict ----
        info['reward_components'] = {
            'R_yield': reward_vector[0],
            'R_nitrogen_eff': reward_vector[1],
            'R_energy': reward_vector[2],
            'R_resilience': reward_vector[3],
        }
        info['sos_state'] = {
            'sensor_health': self.sensor_health.copy(),
            'energy_budget': self.energy_budget,
            'comm_quality': self.comm_quality,
            'total_nitrogen': self.total_nitrogen,
            'total_water': self.total_water,
        }
        info['default_scalar_reward'] = default_reward

        return merged_obs, reward_vector, done, info

    def _compute_rewards(self, crop_obs, nitrogen_applied, water_applied, energy_cost):
        """
        Compute the 4-component reward vector.
        Each component is normalized to approximately [-1, 1].
        """
        # R1: Yield proxy (biomass gain today)
        current_biomass = crop_obs.get('topwt', 0.0)
        biomass_gain = current_biomass - self.prev_biomass
        self.prev_biomass = current_biomass
        R_yield = np.clip(biomass_gain / 150.0, -1.0, 1.0)

        # R2: Nitrogen efficiency (penalize excessive application)
        # 0 nitrogen = 0 penalty, 200 kg = -1.0 penalty
        R_nitrogen = -nitrogen_applied / 200.0

        # R3: Energy efficiency (penalize high consumption)
        R_energy = -np.clip(energy_cost / 15.0, 0.0, 1.0)

        # R4: System resilience (fraction of SoS still operational)
        sensors_up = np.mean(self.sensor_health)
        energy_ok = 1.0 if self.energy_budget > 10 else 0.0
        R_resilience = (sensors_up + energy_ok + self.comm_quality) / 3.0

        return np.array([R_yield, R_nitrogen, R_energy, R_resilience],
                        dtype=np.float32)

    def _inject_faults(self, crop_obs):
        """Randomly inject sensor failures and communication delays."""
        # Sensor dropout: each sensor has fault_rate chance of failing per day
        for i in range(self.n_sensors):
            if self.sensor_health[i] == 1.0 and np.random.random() < self.fault_rate:
                self.sensor_health[i] = 0.0

        # Sensor recovery: small chance of coming back online
        for i in range(self.n_sensors):
            if self.sensor_health[i] == 0.0 and np.random.random() < 0.01:
                self.sensor_health[i] = 1.0

        # Communication quality: random fluctuation
        comm_noise = np.random.normal(0, 0.05)
        self.comm_quality = np.clip(self.comm_quality + comm_noise, 0.3, 1.0)

        # Extreme event: rare communication failure
        if np.random.random() < 0.005:
            self.comm_quality = 0.3

    def _build_observation(self, crop_obs):
        """Merge crop observation with SoS state into a single dict."""
        merged = OrderedDict()

        # Crop state from DSSAT
        for key, value in crop_obs.items():
            merged[f'crop_{key}'] = value

        # SoS state
        for i in range(self.n_sensors):
            merged[f'sensor_{i}'] = self.sensor_health[i]
        merged['energy_budget'] = self.energy_budget / 100.0  # normalize
        merged['comm_quality'] = self.comm_quality

        return merged

    def scalarize_reward(self, reward_vector, weights=None):
        """
        Convert reward vector to scalar using linear scalarization.

        Args:
            reward_vector: np.array of shape (4,)
            weights: np.array of shape (4,), default=[0.4, 0.3, 0.1, 0.2]

        Returns:
            scalar reward (float)
        """
        if weights is None:
            weights = np.array([0.4, 0.3, 0.1, 0.2])
        return float(np.dot(weights, reward_vector))

    def chebyshev_scalarize(self, reward_vector, weights=None, ideal=None):
        """
        Chebyshev scalarization (can find non-convex Pareto solutions).

        R = -max_i(w_i * |r_i - ideal_i|)
        """
        if weights is None:
            weights = np.array([0.4, 0.3, 0.1, 0.2])
        if ideal is None:
            ideal = np.array([1.0, 0.0, 0.0, 1.0])  # best possible per objective

        weighted_dist = weights * np.abs(reward_vector - ideal)
        return -float(np.max(weighted_dist))

    def close(self):
        """Close the underlying gym-DSSAT environment."""
        self.env.close()


# ============================================================
# DEMO: Run the wrapper and see multi-objective rewards
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SmartFarmSoSEnv DEMO - Multi-Objective Rewards")
    print("=" * 70)

    # --- Run WITHOUT faults ---
    print("\n--- Scenario 1: Normal operation (no faults) ---")
    env = SmartFarmSoSEnv(mode='fertilization', seed=42, enable_faults=False)
    obs = env.reset()

    total_rewards = np.zeros(4)
    for day in range(10):
        # Fertilize on days 3 and 7
        if day in [3, 7]:
            action = {'anfer': 40}
        else:
            action = {'anfer': 0}

        obs, reward_vec, done, info = env.step(action)
        total_rewards += reward_vec

        # Show the reward breakdown
        rc = info['reward_components']
        print(f"  Day {day+1:3d}: "
              f"R_yield={rc['R_yield']:+.3f}  "
              f"R_nitro={rc['R_nitrogen_eff']:+.3f}  "
              f"R_energy={rc['R_energy']:+.3f}  "
              f"R_resil={rc['R_resilience']:+.3f}  "
              f"| scalar(linear)={env.scalarize_reward(reward_vec):+.3f}  "
              f"| scalar(cheby)={env.chebyshev_scalarize(reward_vec):+.3f}")

    env.close()
    print(f"\n  Cumulative rewards (10 days): {total_rewards}")

    # --- Run WITH faults ---
    print("\n--- Scenario 2: With sensor faults ---")
    env = SmartFarmSoSEnv(mode='fertilization', seed=42,
                           enable_faults=True, fault_rate=0.10)  # 10% failure rate for demo
    obs = env.reset()

    total_rewards_fault = np.zeros(4)
    for day in range(10):
        if day in [3, 7]:
            action = {'anfer': 40}
        else:
            action = {'anfer': 0}

        obs, reward_vec, done, info = env.step(action)
        total_rewards_fault += reward_vec

        sos = info['sos_state']
        sensors_up = int(np.sum(sos['sensor_health']))
        rc = info['reward_components']
        print(f"  Day {day+1:3d}: "
              f"R_yield={rc['R_yield']:+.3f}  "
              f"R_resil={rc['R_resilience']:+.3f}  "
              f"| sensors={sensors_up}/5  "
              f"energy={sos['energy_budget']:.1f}%  "
              f"comm={sos['comm_quality']:.2f}")

    env.close()

    # --- Compare ---
    print("\n" + "=" * 70)
    print("COMPARISON: Normal vs Faults")
    print("=" * 70)
    labels = ['R_yield', 'R_nitrogen', 'R_energy', 'R_resilience']
    for i, label in enumerate(labels):
        diff = total_rewards_fault[i] - total_rewards[i]
        print(f"  {label:15s}: normal={total_rewards[i]:+.3f}  "
              f"faults={total_rewards_fault[i]:+.3f}  "
              f"diff={diff:+.3f}")

    print("\nNotice how resilience drops when sensors fail.")
    print("Your RL agent will learn to handle these failures.")
    print("\nNext step: run 03_train_morl.py to train multi-objective agents.")
