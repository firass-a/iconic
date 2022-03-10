import gym
import numpy as np

# helpers for action normalization
def normalize_action(action_space_limits, action):
    """Normalize the action from [low, high] to [-1, 1]"""
    low, high = action_space_limits
    return 2.0 * ((action - low) / (high - low)) - 1.0

def denormalize_action(action_space_limits, action):
    """Denormalize the action from [-1, 1] to [low, high]"""
    low, high = action_space_limits
    return low + (0.5 * (action + 1.0) * (high - low))


class SB3GymDssatWrapper(gym.Wrapper):
    """Wrapper for easy and uniform interfacing with SB3.
    
    Currently assumes utilization in fertilization mode only.
    Can be extended for both modes. See: https://gitlab.inria.fr/demukper/gymdssat_playground/-/blob/master/wrappers.py
    """
    def __init__(self, env):
        super(GymDssatWrapper, self).__init__(env)

        self.action_low, self.action_high = self._get_action_space_bounds()

        # using a normalized action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype="float32")

        # using a vector representation of observations to allow
        # easily using SB3 MlpPolicy
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=np.inf,
                                                shape=env.observation_dict_to_array(
                                                    env.observation).shape,
                                                dtype="float32"
                                                )

        # to avoid annoying problem with Monitor when episodes end and things are None
        self.last_info = {}
        self.last_obs = None

    def _get_action_space_bounds(self):
        box = self.env.action_space['anfer']
        return box.low, box.high

    def _format_action(self, action):
        return { 'anfer': action[0] }

    def _format_observation(self, observation):
        return self.env.observation_dict_to_array(observation)

    def reset(self):
        return self._format_observation(self.env.reset())


    def step(self, action):
        # Rescale action from [-1, 1] to original action space interval
        denormalized_action = denormalize_action((self.action_low, self.action_high), action)
        formatted_action = self._format_action(denormalized_action)
        obs, reward, done, info = self.env.step(formatted_action)

        # handle `None`s in obs, reward, and info on done step
        if done:
            obs, reward, info = self.last_obs, 0, self.last_info
        else:
            self.last_obs = obs
            self.last_info = info

        formatted_observation = self._format_observation(obs)
        return formatted_observation, reward, done, info

    def close(self):
        return self.env.close()

    def eval(self):
        self.env.set_evaluation()

    def __del__(self):
        self.close()
