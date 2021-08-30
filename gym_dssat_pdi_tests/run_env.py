import gym
import logging
import multiprocessing
import faulthandler

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
    while not env.done:
        observation = env.observation
        observation_list = env.observation_dict_to_array(observation)
        YRDOY = observation['yrdoy']
        action = default_policy(YRDOY)
        if verbose:
            pprint(observation)
            print(f'yrdoy : {YRDOY} -> fertilizing {action["anfer"]} kgN/ha')
        res = env.step(action)
        new_state, reward, done, info = res
        interactions.append(new_state)
    if env.log_saving_path:
        # time.sleep(1)
        pass
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
        for i in range(10):
            # if env.done:
            env.reset()
            interactions = interact_with_env(env, verbose=False)
            all_interactions.append(interactions)
            if env.log_saving_path:
                # time.sleep(.5)
                pass
            print(env.done)
        return interactions
    except Exception as e:
        logging.exception(e)
    finally:
        env.close()


def multiprocess_trial_hard_reset(env, cwd, rep):
    env.close()
    arguments = []
    for i in range(rep):
        arguments.append((env, f'{cwd}/logs/dssat-pdi-{i}.log'))
    with multiprocessing.Pool() as pool:
        raw_result = list(pool.imap_unordered(_multiprocess_trial_func_hard_reset, arguments))
    return raw_result


def _multiprocess_trial_func_hard_reset(args):
    try:
        env, log_saving_path = args
        all_interactions = []
        env.reset_hard()
        env.log_saving_path = log_saving_path
        for _ in range(10):
            interactions = interact_with_env(env, verbose=False)
            all_interactions.append(interactions)
            if env.log_saving_path:
                time.sleep(.5)
            env.reset()
        # print(interactions[-1]['dap'])
        return interactions
    except Exception as e:
        logging.exception(e)
    finally:
        env.close()


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
        'seed': 123456,
        'random_weather': False,
    }
    done = False
    try_interact = True
    try_multiproc = not True
    if try_interact:
        # with utils.DssatPdiHandler():
        try:
            # interactions = []
            env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
            for i in range(100):
                try:
                    print(env._tmp_folder)
                    interact_with_env(env, verbose=False)
                    env.reset()
                except Exception as e:
                    logging.exception(e)
            env.close()
            env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
            # state_variables = list(env.state.keys())
            # with open('./state_variables.txt', 'w') as f_:
            #     for state_variables in state_variables:
            #         f_.write(f'{state_variables}\n')
            # env.save_log = True
            interaction = interact_with_env(env, verbose=False)
            # interactions.append(interaction)
            # if env.save_log in env_args:
            #     time.sleep(1)
            # env.render(type='ts',
            #            feature_name_1='nstres',
            #            feature_name_2='grnwt')
            # env.render(type='reward',
            #            cumsum=True)
            # env.render(type='reward',
            #            cumsum=False)
            env.reset_hard()
            # env.get_env_info()
            env.reset()
            env.close()
            # interaction = interact_with_env(env, verbose=False)
            # pprint(env.context)
            # print(interaction)
            # if env.save_log in env_args:
            #     time.sleep(1)
            # interactions.append(interaction)
            # print(interactions)
        except Exception as e:
            logging.exception(e)
        finally:
            env.close()
    if try_multiproc:
        with utils.DssatPdiHandler():  # avoid zombies in code crashes
            try:
                tracker = SummaryTracker()
                raw_results1 = multiprocess_trial(env_args, cwd, rep=30)
                print(len(raw_results1))
                # env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
                # raw_results2 = multiprocess_trial_hard_reset(env, cwd, rep=30)
                # print(len(raw_results2))
            except Exception as e:
                logging.exception(e)
                raise e
            finally:
                tracker.print_diff()
