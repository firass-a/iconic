import gym
import gym.spaces as spaces
from subprocess import Popen
import zmq
import json
import yaml

from gym_dssat_pdi.envs.utils import utils
from gym_dssat_pdi.envs.rendering import rendering
from gym_dssat_pdi.envs.rewards import rewards

import tempfile
import random
import shutil
import logging
import os
import gc
import time

import pdb


class DssatPdi(gym.Env):

    def __init__(self, run_dssat_location, experiment_number=1, fileX_prefix='UFGA8201', fileX_extension='.MZX',
                 log_saving_path=None, mode='all', auxiliary_files_names=None, files_prefix='./', random_weather=True):
        self.action_space = spaces.Dict({'anfer': spaces.Box(low=0, high=200, shape=())})
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.experiment_number = experiment_number
        self.fileX_name = f'{fileX_prefix}{fileX_extension}'
        self.mode = mode
        self.config = None
        self.action_variables = None
        self.state_variables = None
        self._load_config()
        # self.file_X_bytes = pkgutil.get_data(__name__, f'configs/{self.file_X_name}')
        # self.dssat_pdi_yaml_template_string = pkgutil.get_data(__name__, f'configs/dssat_pdi.jinja2').decode('utf-8')
        # self.env_config_string = pkgutil.get_data(__name__, f'configs/env_config.yml').decode('utf-8')
        if auxiliary_files_names:
            self.auxiliary_files_names = auxiliary_files_names
        else:
            self.auxiliary_files_names = []
        self.run_dssat_location = run_dssat_location
        self.log_saving_path = log_saving_path
        self.cwd = os.getcwd()
        self.reward_func = rewards.get_reward_function(mode)
        self.history = {'state': [], 'action': [], 'reward': []}
        self._history = {'state': [], 'action': [], 'reward': []}
        self.rseed1 = 2150
        self.random_weather = random_weather
        self.wther = 'W' if random_weather else 'M'
        self.ferti = 'L' if mode in ['all', 'irrigation'] else 'R'
        self.irrig = 'L' if mode in ['all', 'fertilization'] else 'R'
        self.done = False
        self.t = 0
        self.port = None
        self.context = None
        self.server = None
        self.client_process_pid = None
        self.files_prefix = files_prefix
        self.tmp_folder = None
        self._make_tmp_folder()
        self.fileX_template = None
        self._make_fileX_template()
        self._write_fileX_template()
        self._get_sockets_()
        self.state, self._state = self._get_state()

    def _load_config(self):
        with open('./configs/env_config.yml', 'r') as f_:
            config = yaml.load(f_, Loader=yaml.FullLoader)
            self.config = config
            setting_dict = self.config['setting']
            setting = self.mode
            if setting not in setting_dict:
                raise ValueError(f'Authorized values for the "mode" parameter  to be in {[*setting_dict]}')
            self.state_variables = setting_dict[setting]['state']
            self.action_variables = setting_dict[setting]['action']

    def _make_fileX_template(self):
        fileX_template = {'wther': self.wther, 'ferti': self.ferti, 'irrig': self.irrig}
        self.fileX_template = utils.fill_template(value_dic=fileX_template,
                                                  template_path='./configs/UFGA8201.jinja2')

    def _write_fileX_template(self):
        utils.save_file(saving_path=f'{self.tmp_folder}/{self.fileX_name}', content=self.fileX_template)

    def _launch_client(self):
        # print(f'Starting env client: port {self.port}')
        pdi_command = f'pdirun {self.run_dssat_location} C {self.fileX_name} {self.experiment_number}'
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
            process = Popen(pdi_command, stdout=f_, shell=False, universal_newlines=True, cwd=self.tmp_folder)
            self.client_process_pid = process.pid

    def _launch_server(self):
        self.context = zmq.Context()
        # self.server = self.context.socket(zmq.PAIR)
        self.server = self.context.socket(zmq.REP)
        self.server.setsockopt(zmq.LINGER, 0)
        self.server.setsockopt(zmq.IMMEDIATE, 1)
        self.port = self.server.bind_to_random_port('tcp://*', max_tries=10000)  # min_port=1024, max_port=65535)

    def _write_pdi_yaml(self):
        value_dic = {'port': self.port,
                     'rseed1': self.rseed1,
                     }
        # utils.write_template1(value_dic=value_dic,
        #                       template_string=self.dssat_pdi_yaml_template_string,
        #                       saving_path=f'{self.tmp_folder}/dssat-pdi.yml')
        utils.write_template_from_file(value_dic=value_dic,
                                       template_path='./configs/dssat_pdi.jinja2',
                                       saving_path=f'{self.tmp_folder}/dssat-pdi.yml')

    def _get_state(self):
        message = self.server.recv().decode('utf-8')
        message = json.loads(message)
        self.done = message['done']
        state = message['state']
        _state = message['state']
        if state:
            _state = utils._post_treat_state(state)
            state = utils._filter_state(full_state=_state,
                                        state_variables=self.state_variables)
        return state, _state

    def _get_reward(self, _next_state):
        _previous_state = self._state
        _history = self._history
        reward = self.reward_func(_previous_state, _next_state, _history)
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
        # with open(f'{self.tmp_folder}/{self.file_X_name}', 'wb') as f_:
        #     f_.write(self.file_X_bytes)
        # shutil.copyfile(f'./configs/{self.file_X_name}', f'{self.tmp_folder}/{self.file_X_name}')
        if self.auxiliary_files_names:
            self._copy_auxiliary_files(self.auxiliary_files_names)

    def _copy_auxiliary_files(self, names):
        if names:
            for name in names:
                shutil.copyfile(f'{self.files_prefix}{name}', f'{self.tmp_folder}/{name}')

    def _close_client(self):
        if not self.done:
            utils.recursively_kill_process(self.client_process_pid)
        gc.collect()

    def _close_server(self):
        try:
            self.server.close()
            self.context.destroy()
        except Exception as e:
            logging.exception(e)

    def _set_mewth_int(self):
        if self.random_weather:
            self.mewth_int = 87  # 'W'
            self.rseed1 = random.randint(1, 99999)
        else:
            self.mewth_int = 75  # 'M'

    def step(self, action_dict):
        """
        :param action_dict: actions values to be performed
        :type dict
        :return: (state, reward, done, info)
        :rtype: (dict, float, bool, dict)
        """
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg
        assert isinstance(action_dict, dict)
        for available_action in self.action_variables:
            assert available_action in action_dict
        try:
            while True:
                action_js = json.dumps(action_dict, cls=utils.NumpyEncoder).encode('utf-8')
                self.server.send(action_js)
                state, _state = self._get_state()
                if self.done:
                    self._close_client()
                    return None, None, self.done, None
                self.history['state'].append(state)
                self.history['action'].append(action_dict)
                self._history['state'].append(_state)
                self._history['action'].append(action_dict)
                reward = self._get_reward(_state)
                self.history['reward'].append(reward)
                self._history['reward'].append(reward)
                self.state = state
                self._state = _state
                self.reward = reward
                done = self.done
                info = self._get_info()
                self.t += 1
                return state, reward, done, info
        except Exception as e:
            logging.exception(e)

    def reset(self):
        if self.random_weather:
            self.rseed1 = random.randint(1, 99999)
        if not self.done:
            self._close_client()
        self._write_pdi_yaml()
        self._launch_client()
        self.done = False
        self.t = 0
        self.history = {'state': [], 'action': [], 'reward': []}
        self._history = {'state': [], 'action': [], 'reward': []}
        self.server.send(b'')  # to respect REQ/REP send/receive/send/receive/... scheme
        self.state, self.state_ = self._get_state()

    def close(self):
        self._close_server()
        self._close_client()
        shutil.rmtree(self.tmp_folder, ignore_errors=True)
        gc.collect()

    def render(self, type, *args, **kwargs):
        authorized_types = ['ts', 'reward']
        if type not in authorized_types:
            raise ValueError(f'"type" parameter has to be in {[*authorized_types]}!')
        if type == 'ts':
            rendering.render_temporal_series(history=self.history, *args, **kwargs)
        else:
            rendering.render_reward(history=self.history, *args, **kwargs)

    def get_env_info(self):
        utils.get_env_info(config=self.config,
                           action_variables=self.action_variables,
                           state_variables=self.state_variables)

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
