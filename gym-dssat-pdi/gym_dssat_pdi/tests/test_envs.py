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


from contextlib import redirect_stdout
import filecmp


def cultivar_test(cultivar): 
    dirname = os.path.dirname(__file__)
    log_dir = os.path.join(dirname,'logs')
    utils.make_folder(log_dir)
    try:
        for file in os.scandir(log_dir):
            os.remove(file.path)
    except:
        pass

    env_args = {
        'run_dssat_location': 'run_dssat',
        'log_saving_path': os.path.join(log_dir,"dssat_pdi.log"),
        'mode': "fertilization",
        'seed': 123456,
        'random_weather': False,
        'cultivar' : cultivar          
    }
    try_interact = True
    verbose = not True
    if try_interact:
        try:
            env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
            env.activate_automatic_fertilization()
            env.deactivate_automatic_planting()
            env.seed(123)
            yields = []
            env.reset()
            interactions = interact_with_env(env, verbose=verbose)
            yields.append(interactions[-1]['grnwt'])
            print(f'mean of yields: {np.mean(yields)} kg/ha')
            print(f'variance of yields: {np.var(yields)} kg/ha')
            env.reset_hard()
        except Exception as e:
            logging.exception(e)
        finally:
            env.close()
 
    logfile = os.path.join(log_dir,"dssat_pdi.log")
    logfile_format = os.path.join(log_dir,"dssat_pdi_format.log")
    format_file(logfile,logfile_format)
    cultivar_expected_output = os.path.join(dirname,"dssat_output",f"{cultivar}.out")
    test = filecmp.cmp(logfile_format, cultivar_expected_output)
    os.remove(logfile)
    os.remove(logfile_format)
    os.rmdir(log_dir)
    return test

def interact_with_env(env, verbose=True):
    interactions = []
    i = 0
    while not env.done:
        observation = env.observation
        observation_list = env.observation_dict_to_array(observation)
        action = {'anfer': 0, 'amir': 0}
        res = env.step(action)
        new_state, reward, done, info = res
        if verbose:
            pprint(f'dap : {dap} -> fertilizing {action["anfer"]} kg N/ha ; reward {reward}')
        if new_state is not None:
            interactions.append(new_state)
        i += 1
    return interactions

def format_file(input_file, output_file):
    with open(input_file, 'r') as f1:
        with open(output_file, 'w') as f2:
            for i, line in enumerate(f1):
                if i in (47, 48, 49):
                    f2.write(line)

def test_cultivars():
    # Test maize
    assert cultivar_test("maize")

    # Test cotton
    assert cultivar_test("cotton")

    # Test rice
    assert cultivar_test("rice")

