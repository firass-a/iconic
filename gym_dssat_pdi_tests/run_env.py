import gym
import logging
import multiprocessing
import faulthandler
import pathlib

faulthandler.enable()
from gym_dssat_pdi.envs.utils import utils
import os
import gc

import pdb
import numpy as np
from copy import deepcopy
import time
from pympler.tracker import SummaryTracker

from pprint import pprint


def default_policy(YRDOY):
    fertilization_dic = {
        1982097: 27,
        1982102: 35,
        1982137: 54,
    }
    irrigation_dic = {
        1982063: 13,
        1982077: 10,
        1982094: 10,
        1982107: 13,
        1982111: 18,
        1982122: 25,
        1982126: 25,
        1982129: 13,
        1982132: 15,
        1982134: 19,
        1982137: 20,
        1982141: 20,
        1982148: 15,
        1982158: 19,
        1982161: 4,
        1982162: 25,
    }
    if YRDOY in fertilization_dic:
        anfer = fertilization_dic[YRDOY]
    else:
        anfer = 0
    if YRDOY in irrigation_dic:
        amir = irrigation_dic[YRDOY]
    else:
        amir = 0
    return {'anfer': anfer, 'amir': amir}


def interact_with_env(env, verbose=True):
    interactions = []
    i = 0
    while not env.done:
        observation = env.observation
        observation_list = env.observation_dict_to_array(observation)
        YRDOY = observation['yrdoy']
        action = default_policy(YRDOY)
        if verbose:
            pprint(f'observation: {observation}')
            print(f'yrdoy : {YRDOY} -> fertilizing {action["anfer"]} kgN/ha')
            # print(f'sw: {env._state["sw"]}')
        res = env.step(action)
        new_state, reward, done, info = res
        interactions.append(new_state)
        i += 1
    return interactions


def multiprocess_trial(env_args, cwd, rep, save_log=False):
    arguments = []
    n_cores = multiprocessing.cpu_count()
    rep_by_core = rep // (100 * n_cores)
    for i in range(100 * n_cores):
        env_args['log_saving_path'] = f'{cwd}/logs/dssat-pdi-{i}.log'
        env_args['seed'] = np.random.randint(1, 999999)
        arguments.append((deepcopy(env_args), rep_by_core, save_log))
    with multiprocessing.Pool() as pool:
        raw_result = list(pool.imap_unordered(_multiprocess_trial_func, arguments))
    return raw_result

def _multiprocess_trial_func(args):
    try:
        all_interactions = []
        env_args, rep, save_log = args
        if not save_log:
            env_args['log_saving_path'] = None
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        for i in range(rep):
            interactions = interact_with_env(env, verbose=False)
            all_interactions.append(interactions)
            env.reset()
        return all_interactions
    except Exception as e:
        logging.exception(e)
    finally:
        env.close()


def multiprocess_trial_hard_reset(env, cwd, rep, save_log=False):
    env.close()
    arguments = []
    n_cores = multiprocessing.cpu_count()
    rep_by_core = rep // (100 * n_cores)
    for i in range(100 * n_cores):
        arguments.append((env, rep_by_core, f'{cwd}/logs/dssat-pdi-{i}.log', save_log))
    with multiprocessing.Pool() as pool:
        raw_result = list(pool.imap_unordered(_multiprocess_trial_func_hard_reset, arguments))
    return raw_result


def _multiprocess_trial_func_hard_reset(args):
    try:
        env, rep, log_saving_path, save_log = args
        all_interactions = []
        if save_log:
            env.log_saving_path = log_saving_path
        else:
            env.log_saving_path = None
        env.reset_hard()
        for i in range(rep):
            interactions = interact_with_env(env, verbose=False)
            all_interactions.append(interactions)
            env.reset()
        return all_interactions
    except Exception as e:
        logging.exception(e)
    finally:
        env.close()


if __name__ == '__main__':
    dir = './logs'
    utils.make_folder(dir)
    try:
        for file in os.scandir(dir):
            os.remove(file.path)
    except:
        pass
    cwd = os.path.dirname(os.path.realpath(__file__))
    env_args = {
        'run_dssat_location': f'{pathlib.Path.home()}/dssat_pdi/run_dssat',
        'log_saving_path': './logs/dssat-pdi.log',
        # 'mode': 'irrigation',
        # 'mode': 'fertilization',
        'mode': 'all',
        'experiment_number': 3,
        'seed': 123456,
        'random_weather': not True,
    }
    try_interact = True
    try_multiproc = not True
    verbose = not True
    if try_interact:
        try:
            env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
            interact_with_env(env, verbose=verbose)
            env.render(type='ts',
                       feature_name_1='nstres',
                       feature_name_2='grnwt')
            env.render(type='reward',
                       cumsum=True)
            env.render(type='reward',
                       cumsum=False)
        except Exception as e:
            logging.exception(e)
        finally:
            env.close()
    if try_multiproc:
        try:
            tracker = SummaryTracker()
            raw_results1 = multiprocess_trial(env_args, cwd, rep=100, save_log=True)
            print(len(raw_results1))
            env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
            raw_results2 = multiprocess_trial_hard_reset(env, cwd, rep=100)
            print(len(raw_results2))
        except Exception as e:
            logging.exception(e)
            raise e
        finally:
            tracker.print_diff()
