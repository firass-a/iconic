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


def default_policy(dap):
    fertilization_dic = {
        1: 67,
        39: 22,
        66: 22,
    }
    irrigation_dic = {
        1:10,
        7:43,
        11:43,
        14:34,
        18:47,
        22:43,
        24:30,
        32:73,
        38:30,
        49:52,
        56:43,
        71:60,
        73:13,
        75:47,
        78:26,
        81:26,
        85:34,
        89:60,
        92:56,
        93:13,
        94:34,
        101:26,
        103:17,
        105:34,
        107:17,
        109:30,
        111:22,
        113:17,
        115:9,
        117:22,
        119:13,
        121:22,
        123:17,
        125:30,
        127:26
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


def multiprocess_trial(env_args, cwd, rep, save_log=False):
    arguments = []
    n_cores = multiprocessing.cpu_count()
    rep_by_core = rep // n_cores
    for i in range(n_cores):
        env_args['log_saving_path'] = f'./logs/dssat-pdi-{i}.log'
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
            env.reset()
            interactions = interact_with_env(env, verbose=False)
            all_interactions.append(interactions)
        return all_interactions
    except Exception as e:
        logging.exception(e)
    finally:
        env.close()


def multiprocess_trial_hard_reset(env, cwd, rep, save_log=False):
    arguments = []
    n_cores = multiprocessing.cpu_count()
    rep_by_core = rep // n_cores
    for i in range(n_cores):
        arguments.append((env, rep_by_core, f'./logs/dssat-pdi-{i}.log', save_log))
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
            env.reset()
            interactions = interact_with_env(env, verbose=False)
            all_interactions.append(interactions)
        return all_interactions
    except Exception as e:
        logging.exception(e)
    finally:
        env.close()


if __name__ == '__main__':
    dir = './logs/'
    utils.make_folder(dir)
    try:
        for file in os.scandir(dir):
            os.remove(file.path)
    except:
        pass
    utils.make_folder('./render')
    cwd = os.path.dirname(os.path.realpath(__file__))
    for i, mode in enumerate([
                              'fertilization',
                              'irrigation',
                              'all'
                              ]):
        print(f'MODE: {mode}')
        env_args = {
            'run_dssat_location': 'run_dssat',
            'log_saving_path': './logs/dssat_pdi.log',
            'mode': mode,
            'seed': 123456,
            'random_weather': True,
            'cultivar': "cotton",
        }
        try_interact = True
        try_multiproc = True
        verbose = not True
        if try_interact:
            try:
                env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
                #if i == 0:
                    #env.get_env_info(user_input=False)
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
                if mode == 'mode':
                    env.render(type='ts',
                               feature_name_1='cleach',
                               feature_name_2='totaml')
                    env.render(type='reward',
                               cumsum=True)
                    env.render(type='reward',
                               cumsum=False)
                env.reset_hard()
            except Exception as e:
                logging.exception(e)
            finally:
                env.close()
        if try_multiproc:
            try:
                rep = 80
                n_cores = multiprocessing.cpu_count()
                # rep_by_core = rep // n_cores
                raw_results1 = multiprocess_trial(env_args, cwd, rep=rep, save_log=True)
                print(f'{len(raw_results1)}/{n_cores} multiprocess_trial')
                env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
                env.close()
                raw_results2 = multiprocess_trial_hard_reset(env, cwd, rep=rep)
                print(f'{len(raw_results2)}/{n_cores} multiprocess_trial_hard_reset')
            except Exception as e:
                logging.exception(e)
