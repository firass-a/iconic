import gym
import gym.spaces as spaces
from subprocess import Popen
import zmq
import json
from gym_dssat_pdi.envs.utils import utils
from gym_dssat_pdi.envs.rendering import rendering
from gym_dssat_pdi.envs.rewards import rewards

import tempfile
import random
import shutil
import logging
import os
import gc
import pdb
import time


class DssatPdi(gym.Env):

    def __init__(self, run_dssat_location, mode='fertilization', experiment_number=1, file_X_prefix='UFGA8201',
                 fileX_extension='.MZX', log_saving_path=None, yml_template_path='./templates/dssat-pdi.jinja2',
                 auxiliary_files_names=None, files_prefix='./'):
        self.action_space = spaces.Dict({'anfer': spaces.Box(low=0, high=200, shape=())})
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.experiment_number = experiment_number
        self.file_X = f'{file_X_prefix}{fileX_extension}'
        if auxiliary_files_names:
            self.auxiliary_files_names = auxiliary_files_names
        else:
            self.auxiliary_files_names = []
        self.run_dssat_location = run_dssat_location
        self.log_saving_path = log_saving_path
        self.yml_template_path = yml_template_path
        self.cwd = os.getcwd()
        self.mode = mode
        self.reward_func = rewards.fertilization_reward if mode == 'fertilization' else rewards.fertilization_reward
        self.history = {'state': [], 'action': [], 'reward': []}
        self.done = False
        self.t = 0
        self.port = None
        self.context = None
        self.server = None
        self.client_process_pid = None
        self.files_prefix = files_prefix
        self.tmp_folder = None
        self._make_tmp_folder()
        self._get_sockets_()
        self.state = self._get_state()

    def _launch_client(self):
        # print(f'Starting env client: port {self.port}')
        pdi_command = f'pdirun {self.run_dssat_location} C {self.file_X}' \
                      f' {self.experiment_number}'
        pdi_command = pdi_command.split(' ')
        if self.log_saving_path is not None:
            file_path = self.log_saving_path
        else:
            file_path = os.devnull
        with open(file_path, 'a+') as f_:
            if self.log_saving_path is not None:
                f_.write('\n********************************\n')
                f_.write(utils.get_time_stamp())
                f_.write('\n********************************\n')
            process = Popen(pdi_command, stdout=f_, shell=False, universal_newlines=True,
                            cwd=self.tmp_folder)
            self.client_process_pid = process.pid

    def _launch_server(self):
        self.context = zmq.Context()
        self.server = self.context.socket(zmq.PAIR)
        self.server.setsockopt(zmq.LINGER, 0)
        self.port = self.server.bind_to_random_port('tcp://*', max_tries=10000)  # min_port=1024, max_port=65535)

    def _write_pdi_yaml(self):
        value_dic = {'port': self.port}
        utils.write_template(value_dic=value_dic,
                             template_path=self.yml_template_path,
                             saving_path=f'{self.tmp_folder}/dssat-pdi.yml')

    def _get_state(self):
        message = self.server.recv().decode('utf-8')
        message = json.loads(message)
        state = message['state']
        if state:
            state = utils._post_treat_state(state)
        self.done = message['done']
        if self.done:
            self._close_server()
        return state

    def _get_reward(self, next_state):
        previous_state = self.state
        history = self.history
        reward = self.reward_func(previous_state, next_state, history)
        return reward

    def _get_info(self):
        state = self.state
        return {}

    def _get_sockets_(self):
        self._launch_server()
        self._write_pdi_yaml()
        self._launch_client()

    def _make_tmp_folder(self):
        # if self.tmp_folder is not None:
        shutil.rmtree(self.tmp_folder, ignore_errors=True)
        tempfile._Random = random.Random
        self.tmp_folder = tempfile.mkdtemp()
        if self.auxiliary_files_names is not None:
            self._copy_auxiliary_files([self.file_X, *self.auxiliary_files_names])

    def _copy_auxiliary_files(self, names):
        for name in names:
            shutil.copyfile(f'{self.files_prefix}{name}', f'{self.tmp_folder}/{name}')

    def _close_client(self):
        try:
            utils.recursively_kill_process(self.client_process_pid)
        except Exception as e:
            logging.exception(e)
        gc.collect()

    def _close_server(self):
        try:
            self.server.close()
            self.context.term()
        except Exception as e:
            # pass
            logging.exception(e)

    def step(self, action):
        """
        :param action: actions values to be performed
        :type dict
        :return: (state, reward, done, info)
        :rtype: (dict, float, bool, dict)
        """
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg
        assert isinstance(action, dict)
        try:
            while True:
                action_js = json.dumps(action, default=utils.convert).encode('utf-8')
                self.server.send(action_js)
                state = self._get_state()
                if state:
                    self.history['state'].append(state)
                    self.history['action'].append(action)
                    reward = self._get_reward(state)
                    self.history['reward'].append(reward)
                    self.state = state
                    self.reward = reward
                    done = self.done
                    info = self._get_info()
                    self.t += 1
                    return state, reward, done, info
                else:
                    return None, None, self.done, None


        except Exception as e:
            logging.exception(e)

    def reset(self):
        self.done = False
        self.t = 0
        self._close_client()
        self._launch_client()
        self.history = {'state': [], 'action': [], 'reward': []}
        self._get_state()

    def render(self, type, *args, **kwargs):
        if 'type' == 'ts':
            rendering.render_temporal_series(history=self.history, *args, **kwargs)
        else:
            rendering.render_reward(history=self.history,  *args, **kwargs)

    def close(self):
        self._close_server()
        self._close_client()
        shutil.rmtree(self.tmp_folder, ignore_errors=True)
        gc.collect()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
