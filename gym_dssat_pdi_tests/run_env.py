import gym
import logging
import multiprocessing
import faulthandler

faulthandler.enable()
from gym_dssat_pdi.envs.utils.utils import DssatPdiHandler
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
    while not env.done:
        state = env.state
        YRDOY = state['yrdoy']
        action = default_policy(YRDOY)
        if verbose:
            pprint(state)
            print(f'yrdoy : {YRDOY} -> fertilizing {action["anfer"]} kgN/ha')
        res = env.step(action)
        new_state, reward, done, info = res
        interactions.append(res)
    return interactions


def multiprocess_trial(env_args, cwd, rep):
    arguments = []
    for i in range(rep):
        env_args['log_saving_path'] = f'{cwd}/logs/dssat-pdi-{i}.log'
        arguments.append((deepcopy(env_args)))
    with multiprocessing.Pool() as pool:
        raw_result = list(pool.imap_unordered(_multiprocess_trial_func, arguments))
    return raw_result


def _multiprocess_trial_func(args):
    try:
        all_interactions = []
        env_args = args
        env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
        env.save_log = False
        for _ in range(10):
            interactions = interact_with_env(env, verbose=False)
            all_interactions.append(interactions)
            env.reset()
        if env.save_log in env_args:
            time.sleep(1)
        print(interactions)
        return interactions
    except Exception as e:
        logging.exception(e)
    finally:
        env.close()
        gc.collect()


if __name__ == '__main__':
    dir = './logs'
    try:
        for file in os.scandir(dir):
            os.remove(file.path)
    except:
        pass
    cwd = os.path.dirname(os.path.realpath(__file__))
    env_args = {
        'run_dssat_location': '/home/rgautron/dssat_pdi/run_dssat',
        'log_saving_path': './logs/dssat-pdi.log',
        'mode': 'fertilization',
        'experiment_number': 3,
    }
    done = False
    try_interact = True
    try_multiproc = not True
    if try_interact:
        with DssatPdiHandler():
            try:
                interactions = []
                env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
                # state_variables = list(env.state.keys())
                # with open('./state_variables.txt', 'w') as f_:
                #     for state_variables in state_variables:
                #         f_.write(f'{state_variables}\n')
                env.save_log = True
                interaction = interact_with_env(env, verbose=False)
                interactions.append(interaction)
                if env.save_log in env_args:
                    time.sleep(1)
                env.render(type='ts',
                           feature_name_1='nstres',
                           feature_name_2='grnwt')
                env.render(type='reward',
                           cumsum=True)
                env.render(type='reward',
                           cumsum=False)
                env.reset()
                # env.get_env_info()
                interaction = interact_with_env(env, verbose=False)
                if env.save_log in env_args:
                    time.sleep(1)
                interactions.append(interaction)
                # print(interactions)
            except Exception as e:
                logging.exception(e)
            finally:
                env.close()
    if try_multiproc:
        with DssatPdiHandler():  # avoid zombies in code crashes
            try:
                tracker = SummaryTracker()
                raw_results = multiprocess_trial(env_args, cwd, rep=100)
                # print(raw_results)
            except Exception as e:
                logging.exception(e)
                raise e
            finally:
                tracker.print_diff()
