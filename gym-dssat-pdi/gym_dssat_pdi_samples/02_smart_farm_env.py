"""
SmartFarmSoSEnv — Multi-objective wrapper for gym-DSSAT.

Implements the thesis reward design:
  - Daily shaping (water, N, resources, losses)
  - Seasonal harvest bonus (yield, HI, ANE)
  - 3-objective vector for PC-PPO: [yield, water, fertilizer]

Run inside Docker:
    python3 02_smart_farm_env.py
"""
import gym
import numpy as np
from collections import OrderedDict

from smart_farm_rewards import compute_reward_vector, moisture_ratio

# Farmer-realistic observation keys (no hidden biochemical state)
FARMER_OBS_KEYS = (
    'dap', 'vstage', 'xlai', 'istage',
    'swfac', 'nstres',
    'rain', 'tmax', 'srad',
    'cumsumfert', 'totir',
    'grnwt', 'topwt',
)


class SmartFarmSoSEnv:
    """
    Wraps gym-DSSAT for multi-objective smart-farm control.

    Returns:
        obs: dict with crop_* keys + moisture_ratio
        reward_vector: np.array shape (3,) — [R_yield, R_water, R_fert]
        done: bool
        info: dict
    """

    N_OBJECTIVES = 3
    OBJECTIVE_NAMES = ('yield', 'water', 'fertilizer')

    def __init__(self, mode='all', seed=None, run_dssat_location='run_dssat',
                 random_weather=False, enable_faults=False, fault_rate=0.02,
                 n_sensors=5, initial_energy=100.0):
        env_args = {
            'mode': mode,
            'run_dssat_location': run_dssat_location,
            'random_weather': random_weather,
            'log_saving_path': None,
        }
        if seed is not None:
            env_args['seed'] = seed
        self.env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)

        self.mode = mode
        self.enable_faults = enable_faults
        self.fault_rate = fault_rate
        self.n_sensors = n_sensors
        self.initial_energy = initial_energy

        self.sensor_health = None
        self.energy_budget = None
        self.comm_quality = None
        self.total_water = 0.0
        self.total_nitrogen = 0.0
        self.prev_cnox = 0.0
        self._last_crop_obs = {}
        self._last_context = {}
        self._last_moisture = 0.5
        self.done = False

    def reset(self):
        obs = self.env.reset()
        if obs is None:
            obs = {}
        self._last_crop_obs = obs
        self._last_context = {}

        self.sensor_health = np.ones(self.n_sensors)
        self.energy_budget = self.initial_energy
        self.comm_quality = 1.0
        self.total_water = 0.0
        self.total_nitrogen = 0.0
        self.prev_cnox = float(obs.get('cnox', 0.0) or 0.0)
        self.done = False

        return self._build_observation(obs, self._last_context)

    def step(self, action_dict):
        crop_obs, _default_reward, done, context = self.env.step(action_dict)
        if context is None:
            context = {}

        if crop_obs is None or len(crop_obs) == 0:
            crop_obs = {}
            state_for_reward = dict(self._last_crop_obs)
        else:
            self._last_crop_obs = crop_obs
            state_for_reward = dict(crop_obs)

        # Full state has runoff, trnu, tleachd, etc.
        full_state = getattr(self.env, '_state', None)
        if full_state:
            state_for_reward.update(full_state)

        self.done = done
        nitrogen_applied = float(action_dict.get('anfer', 0.0) or 0.0)
        water_applied = float(action_dict.get('amir', 0.0) or 0.0)
        self.total_nitrogen += nitrogen_applied
        self.total_water += water_applied

        if self.enable_faults:
            self._inject_faults()

        totals = {'water': self.total_water, 'nitrogen': self.total_nitrogen}
        reward_vector, moisture = compute_reward_vector(
            state=state_for_reward,
            context=context,
            action=action_dict,
            totals=totals,
            prev_cnox=self.prev_cnox,
            done=done,
        )
        self.prev_cnox = float(state_for_reward.get('cnox', self.prev_cnox) or self.prev_cnox)
        self._last_context = context
        self._last_moisture = moisture

        merged_obs = self._build_observation(crop_obs, context)
        info = {
            'reward_components': {
                'R_yield': float(reward_vector[0]),
                'R_water': float(reward_vector[1]),
                'R_fert': float(reward_vector[2]),
            },
            'moisture_ratio': moisture,
            'totals': totals.copy(),
            'full_state': state_for_reward,
        }
        return merged_obs, reward_vector, done, info

    def _build_observation(self, crop_obs, context):
        """Farmer-realistic observations only."""
        merged = OrderedDict()
        for key in FARMER_OBS_KEYS:
            if key in crop_obs:
                merged[f'crop_{key}'] = crop_obs[key]

        sw = crop_obs.get('sw')
        dul = context.get('dul') if context else None
        ll = context.get('ll') if context else None
        if sw is not None and dul is not None:
            merged['moisture_ratio'] = moisture_ratio(sw, dul, ll)
        else:
            merged['moisture_ratio'] = self._last_moisture

        if self.enable_faults:
            for i in range(self.n_sensors):
                merged[f'sensor_{i}'] = self.sensor_health[i]
            merged['comm_quality'] = self.comm_quality

        return merged

    def _inject_faults(self):
        for i in range(self.n_sensors):
            if self.sensor_health[i] == 1.0 and np.random.random() < self.fault_rate:
                self.sensor_health[i] = 0.0
            if self.sensor_health[i] == 0.0 and np.random.random() < 0.01:
                self.sensor_health[i] = 1.0
        self.comm_quality = np.clip(
            self.comm_quality + np.random.normal(0, 0.05), 0.3, 1.0
        )

    def scalarize_reward(self, reward_vector, weights=None):
        if weights is None:
            weights = np.array([0.5, 0.25, 0.25], dtype=np.float32)
        return float(np.dot(weights, reward_vector))

    def close(self):
        self.env.close()


if __name__ == '__main__':
    print('=' * 70)
    print('SmartFarmSoSEnv — 3-objective reward demo')
    print('=' * 70)

    env = SmartFarmSoSEnv(mode='all', seed=42, random_weather=False)
    obs = env.reset()
    print(f'Initial obs keys: {list(obs.keys())}')
    print(f'moisture_ratio = {obs["moisture_ratio"]:.3f}')

    cum = np.zeros(3)
    step = 0
    while not env.done and step < 15:
        action = {'anfer': 10.0 if step in (5, 10) else 0.0, 'amir': 8.0 if step == 7 else 0.0}
        obs, r_vec, done, info = env.step(action)
        cum += r_vec
        rc = info['reward_components']
        print(
            f'  day {step+1:2d}: moisture={info["moisture_ratio"]:.2f} '
            f'R_y={rc["R_yield"]:+.3f} R_w={rc["R_water"]:+.3f} R_f={rc["R_fert"]:+.3f}'
        )
        step += 1

    env.close()
    print(f'\nCumulative R⃗ (partial season): {cum.round(3)}')
    print('Next: python3 pc_env.py  →  then 03_sb3_sanity_check.py  →  04_pc_ppo_quick_train.py')
