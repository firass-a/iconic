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

from contextlib import redirect_stdout
import filecmp


def test_default_env():
    dir = './logs/'
    utils.make_folder(dir)
    try:
        for file in os.scandir(dir):
            os.remove(file.path)
    except:
        pass
    utils.make_folder('./render')
    cwd = os.path.dirname(os.path.realpath(__file__))
    
    with open('test_output.log', 'w') as f:
        with redirect_stdout(f):
            for i, mode in enumerate([
                                    'fertilization',
                                    'irrigation',
                                    'all'
                                    ]):
                for j, cultivar in enumerate(['maize','cotton']):
                    print(f'MODE: {mode}, CULTIVAR: {cultivar}')
                    ## Random weather for cotton is currently using maize WGEN,
                    ## AZMC.CLI is a copy of UFGA.CLI inside dssat-cm-data/Weather/Climate
                    env_args = {
                        'run_dssat_location': 'run_dssat',
                        'log_saving_path': './logs/dssat_pdi.log',
                        'mode': mode,
                        'seed': 123456,
                        'random_weather': True,
                        'cultivar' : cultivar          
                    }
                    try_interact = True
                    verbose = not True
                    if try_interact:
                        try:
                            env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
                            if i == 0:
                                env.get_env_info(user_input=False)
                            env.seed(123)
                            n_rep = 8
                            yields = []
                            for j in range(n_rep):
                                env.reset()
                                interactions = interact_with_env(env, verbose=verbose)
                                yields.append(interactions[-1]['grnwt'])
                                if (j + 1) % 10 == 0:
                                    print(f'{j + 1}/{n_rep}')
                            print(f'mean of yields: {np.mean(yields)} kg/ha')
                            print(f'variance of yields: {np.var(yields)} kg/ha')
                            env.reset_hard()
                        except Exception as e:
                            logging.exception(e)
                        finally:
                            env.close()
    
    #compare = filecmp.cmp(cwd+'/test_env_expected.log','test_output.log')
    #assert compare == True
    #os.remove("test_output.log")


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
    # if dap in fertilization_dic:
    #     anfer = fertilization_dic[dap]
    # else:
    #     anfer = 0
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


test_default_env()