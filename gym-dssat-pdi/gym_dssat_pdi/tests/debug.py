import gym
import logging
import multiprocessing
import faulthandler

faulthandler.enable()
from gym_dssat_pdi.envs.utils import utils

import numpy as np
from copy import deepcopy
from pprint import pprint

import os

dirname = os.path.dirname(__file__)


def default_policy(dap):
    fertilization_dic = {
        40: 27,
        45: 35,
        80: 54,
    }
    irrigation_dic = {
        6: 13,
        20: 10,
        37: 10,
        50: 13,
        54: 18,
        65: 25,
        69: 25,
        72: 13,
        75: 15,
        77: 19,
        80: 20,
        84: 20,
        91: 15,
        101: 19,
        104: 4,
        105: 25,
    }
    if dap in fertilization_dic:
        anfer = fertilization_dic[dap]
    else:
        anfer = 0
    if dap in irrigation_dic:
        amir = irrigation_dic[dap]
    else:
        amir = 0
    return {'anfer': anfer, 'amir': amir}


def interact_with_env(env, verbose=True):
    interactions = []
    i = 0
    while not env.done:
        observation = env.observation
        observation_list = env.observation_dict_to_array(observation)
        dap = observation['dap']
        action = default_policy(dap)
        res = env.step(action)
        new_state, reward, done, info = res
        if verbose:
            pprint(f'dap : {dap} -> fertilizing {action["anfer"]} kg N/ha ; reward {reward}')
        if new_state is not None:
            interactions.append(new_state)
        i += 1
    return interactions


if __name__ == '__main__':
    dir = './logs/'
    utils.make_folder(dir)
    try:
        for file in os.scandir(dir):
            os.remove(file.path)
    except:
        pass
    cwd = os.path.dirname(os.path.realpath(__file__))
    for i, mode in enumerate([
                              'fertilization',
                              ]):
        env_args = {
            'run_dssat_location': 'run_dssat',
            'log_saving_path': './logs/dssat_pdi.log',
            'mode': mode,
            'seed': 123456,
            'random_weather': True,
            'cultivar' : 'cotton'
        }
        try:
            env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
            env.seed(123)
            yields = []
            env.reset()
            interactions = interact_with_env(env, verbose=False)
            yields.append(interactions[-1]['grnwt'])
            print(f'mean of yields: {np.mean(yields)} kg/ha')
            print(f'variance of yields: {np.var(yields)} kg/ha')
            env.reset_hard()
        except Exception as e:
            logging.exception(e)
        finally:
            env.close()