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

def fertilization_policy(YRDOY):
    # return {'anfer': 1}
    fertilization_dic = {
        1982097: 27,
        1982102: 35,
        1982137: 54,
    }
    if YRDOY not in fertilization_dic:
        anfer = 0
    else:
        anfer = fertilization_dic[YRDOY]
    return {'anfer': anfer}


def interact_with_env(env, verbose=True):
    interactions = []
    while not env.done:
        state = env.state
        YRDOY = state['yrdoy']
        action = fertilization_policy(YRDOY)
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
            print(interactions)
            all_interactions.append(interactions)
            env.reset()
        if env.save_log in env_args:
            time.sleep(1)
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
    }
    done = False
    try_interact = True
    try_multiproc = not True
    if try_interact:
        try:
            env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
            # env.reset()
            # state_variables = list(env.state.keys())
            # with open('./state_variables.txt', 'w') as f_:
            #     for state_variables in state_variables:
            #         f_.write(f'{state_variables}\n')
            env.save_log = True
            interaction = interact_with_env(env)
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
        with DssatPdiHandler():  # avoid zombies in code crashes
            try:
                tracker = SummaryTracker()
                raw_results = multiprocess_trial(env_args, cwd, rep=10)
                # print(raw_results)
            except Exception as e:
                logging.exception(e)
                raise e
            finally:
                tracker.print_diff()