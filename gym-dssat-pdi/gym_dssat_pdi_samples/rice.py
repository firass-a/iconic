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
        1: 120,
    }
    irrigation_dic = {
        7:27,
        10:26,
        14:26,
        20:25,
        27:28,
        31:27,
        38:31,
        42:14,
        45:11,
        49:28,
        52:23,
        57:8,
        60:17,
        63:18,
        65:11,
        67:10,
        69:8,
        71:15,
        73:13,
        76:27,
        78:16,
        81:19,
        90:23,
        94:22,
        99:24,
        101:15,
        103:14,
        105:11,
        108:19,
        111:5,
        112:14,
        115:20,
        121:15,
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
        #print(observation)
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
                              #'all'
                              ]):
        print(f'MODE: {mode}')
        env_args = {
            'run_dssat_location': 'run_dssat',
            'log_saving_path': './logs/dssat_pdi.log',
            'mode': mode,
            'seed': 123456,
            'random_weather': False,
            'cultivar': "rice",
        }
        try_interact = True
        try_multiproc = False
        verbose = False
        if try_interact:
            try:
                env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
                #if i == 0:
                    #env.get_env_info(user_input=False)
                n_rep = 1
                env.seed(123)
                yields = []
                env.reset_hard()

                # FOR DEBUG
                # initial_observation = env.observation  # Save the initial observation
                # print("Initial observation:", initial_observation)

                for j in range(n_rep):
                    env.reset()
                    interactions = interact_with_env(env, verbose=verbose)
                    #print(interactions[-1])
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
