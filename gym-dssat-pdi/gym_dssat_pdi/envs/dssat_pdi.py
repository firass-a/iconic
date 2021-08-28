import gym
import gym.spaces as spaces
from gym_dssat_pdi.envs.utils import utils
from gym_dssat_pdi.envs.rendering import rendering
from gym_dssat_pdi.envs.rewards import rewards
import numpy as np
from gym.utils import seeding
from subprocess import Popen
import zmq
import json
import yaml
import tempfile
import shutil
import logging
import os
import gc
import pkgutil
from pprint import pprint
import time
import pdb

__copyright__ = 'Copyright CGIAR, Inria and CIRAD'
__credits__ = [
    'Romain Gautron',
    'Emilio Padrón González',
]
__license__ = 'BSD 3-Clause'
__author__ = 'Romain Gautron <romain.gautron@cirad.fr>'


class DssatPdi(gym.Env):

    def __init__(self, run_dssat_location, experiment_number=1, fileX_prefix='UFGA8201', fileX_extension='.MZX',
                 log_saving_path=None, mode='all', auxiliary_files_names=None, files_prefix='./', random_weather=True,
                 seed=None):
        self.experiment_number = experiment_number
        self.fileX_name = f'{fileX_prefix}{fileX_extension}'
        self.mode = mode
        self.action_variables = None
        self.observation_variables = None
        self.context_variables = None
        self.fileX_template = pkgutil.get_data(__name__, f'configs/{fileX_prefix}.jinja2').decode('utf-8')
        self.fileX = None
        self.pdi_yaml_template = pkgutil.get_data(__name__, f'configs/dssat_pdi.jinja2').decode('utf-8')
        self.pdi_yaml = None
        self.env_yaml_config = pkgutil.get_data(__name__, f'configs/env_config.yml').decode('utf-8')
        self.config = None
        self._load_config()
        self.observation_space = None
        self.context_space = None
        self.action_space = None
        for key in ['observation', 'context', 'action']:
            self._make_gym_spaces(key=key)
        if auxiliary_files_names:
            self.auxiliary_files_names = auxiliary_files_names
        else:
            self.auxiliary_files_names = []
        self.run_dssat_location = run_dssat_location
        self.log_saving_path = log_saving_path
        self.cwd = os.getcwd()
        self.reward_func = rewards.get_reward_function(mode)
        self.history = {'observation': [], 'action': [], 'reward': []}
        self._history = {'state': [], 'action': [], 'reward': []}
        self.random_generator = None
        self.seed = None
        self.set_seed(seed=seed)
        self.rseed1 = self.random_generator.randint(1, 99999)
        self.random_weather = random_weather
        self.wther = 'W' if random_weather else 'M'
        self.ferti = 'L' if mode in ['all', 'irrigation'] else 'R'
        self.irrig = 'L' if mode in ['all', 'fertilization'] else 'R'
        self.done = False
        self.t = 0
        self._port = None
        self._zmq_context = None
        self._server = None
        self._client_process_pid = None
        self.files_prefix = files_prefix
        self._tmp_folder = None
        self._make_tmp_folder()
        self._make_fileX_template()
        self._write_fileX_template()
        self._get_sockets_()
        self.observation, self._state, self.done, self.context = self._get_state()

    def _load_config(self):
        config = yaml.load(self.env_yaml_config, Loader=yaml.FullLoader)
        self.config = config
        setting_dict = self.config['setting']
        setting = self.mode
        if setting not in setting_dict:
            raise ValueError(f'Authorized values for the "mode" parameter  to be in {[*setting_dict]}')
        self.observation_variables = sorted(setting_dict[setting]['state'])
        self.action_variables = setting_dict[setting]['action']
        self.context_variables = setting_dict[setting]['context']

    def _make_gym_spaces(self, key):
        key_spaces = {}
        if key == 'observation':
            key_variables = self.observation_variables
        elif key == 'context':
            key_variables = self.context_variables
        elif key == 'action':
            key_variables = self.action_variables
        else:
            raise ValueError('"key" parameter must be in ["observation", "action", "context"]')
        if not key_variables:
            return {}
        if key in ['observation', 'context']:
            key_config = 'state'
        else:
            key_config = 'action'
        key_data = self.config[key_config]
        for key_variable in key_variables:
            key_variable_dic = key_data[key_variable]
            if 'type' not in [*key_variable_dic]:
                raise ValueError(f'"type" must be specified for {key} variable "{key_variable}"')
            type_ = key_variable_dic['type']
            if type_ == 'float' or type_ == 'int':
                if ('high' not in [*key_variable_dic]) or ('low' not in [*key_variable_dic]):
                    raise ValueError(f'"high" and "low" must be specified for {key} variable "{key_variable}"')
                low = key_variable_dic['low']
                high = key_variable_dic['high']
            if type_ == 'float':
                space = spaces.Box(low=low, high=high, shape=())
            elif type_ == 'discrete':
                if 'size' not in [*key_variable_dic]:
                    raise ValueError(f'"size" must be specified for {key} variable "{key_variable}"')
                size = key_variable_dic['size']
                space = spaces.Discrete(size)
            elif type_ == 'int':
                size = high - low + 1
                space = spaces.Discrete(size)
            elif type_ == 'array':
                if 'subtype' not in [*key_variable_dic]:
                    raise ValueError(f'"subtype" must be specified for {key} variable "{key_variable}"')
                subtype = key_variable_dic['subtype']
                if 'size' not in [*key_variable_dic]:
                    raise ValueError(f'"size" must be specified for {key} variable "{key_variable}"')
                size = key_variable_dic['size']
                if subtype == 'float':
                    if ('high' not in [*key_variable_dic]) or ('low' not in [*key_variable_dic]):
                        raise ValueError(
                            f'"high" and "low" must be specified for {key} variable "{key_variable}"')
                    low = key_variable_dic['low']
                    high = key_variable_dic['high']
                    space = spaces.Box(low=low, high=high, shape=(size, ))
                elif subtype == 'discrete':
                    atomic_spaces = []
                    if 'subsize' not in [*key_variable_dic]:
                        raise ValueError(f'"subsize" must be specified for {key} variable "{key_variable}"')
                    sub_size = key_variable_dic['subsize']
                    for element in range(size):
                        atomic_spaces.append(spaces.Discrete(sub_size))
                    space = spaces.Tuple(atomic_spaces)
                elif subtype == 'int':
                    if ('high' not in [*key_variable_dic]) or ('low' not in [*key_variable_dic]):
                        raise ValueError(
                            f'"high" and "low" must be specified for {key} variable "{key_variable}"')
                    low = key_variable_dic['low']
                    high = key_variable_dic['high']
                    atomic_spaces = []
                    sub_size = high - low + 1
                    for element in range(size):
                        atomic_spaces.append(spaces.Discrete(sub_size))
                    space = spaces.Tuple(atomic_spaces)
                else:
                    raise ValueError(f'{key} variable {key_variable} subtype {subtype} not in'
                                     f' {["float", "int", "discrete"]}')
            else:
                raise ValueError(f'{key} variable "{key_variable}" not in {[*key_data]}')
            key_spaces[key_variable] = space
        if key == 'observation':
            self.observation_space = spaces.Dict(key_spaces)
        elif key == 'context':
            self.context_space = spaces.Dict(key_spaces)
        else:
            self.action_space = spaces.Dict(key_spaces)

    def _make_fileX_template(self):
        fileX_template_values = {'wther': self.wther, 'ferti': self.ferti, 'irrig': self.irrig}
        self.fileX = utils._fill_template_from_string(value_dic=fileX_template_values,
                                                      template_string=self.fileX_template)

    def _write_fileX_template(self):
        utils.save_file(saving_path=f'{self._tmp_folder}/{self.fileX_name}', content=self.fileX)

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
            process = Popen(pdi_command, stdout=f_, shell=False, universal_newlines=True, cwd=self._tmp_folder)
            self._client_process_pid = process.pid

    def _launch_server(self):
        self._zmq_context = zmq.Context()
        self._server = self._zmq_context.socket(zmq.REP)
        self._server.setsockopt(zmq.LINGER, 0)
        self._server.setsockopt(zmq.IMMEDIATE, 1)
        self._port = self._server.bind_to_random_port('tcp://*', max_tries=10000)  # min_port=1024, max_port=65535)

    def _write_pdi_yaml(self):
        value_dic = {'port': self._port,
                     'rseed1': self.rseed1,
                     }
        self.pdi_yaml = utils._fill_template_from_string(value_dic=value_dic,
                                                         template_string=self.pdi_yaml_template)
        utils.save_file(saving_path=f'{self._tmp_folder}/dssat-pdi.yml', content=self.pdi_yaml)

    def _get_state(self):
        message = self._server.recv().decode('utf-8')
        message = json.loads(message)
        done = message['done']
        _state = message['state']
        if _state:
            _state = utils._post_treat_state(_state)
            observation = utils._filter_state(full_state=_state,
                                              observation_variables=self.observation_variables)
            context = utils._filter_state(full_state=_state,
                                              observation_variables=self.observation_variables)
        else:
            observation = {}
            context = {}
        return observation, _state, done, context

    def _get_reward(self, _next_state):
        _previous_state = self._state
        _history = self._history
        reward = self.reward_func(_previous_state, _next_state, _history)
        return reward

    def _get_sockets_(self):
        self._launch_server()
        self._write_pdi_yaml()
        self._launch_client()

    def _make_tmp_folder(self):
        shutil.rmtree(self._tmp_folder, ignore_errors=True)
        self._tmp_folder = tempfile.mkdtemp()
        if self.auxiliary_files_names:
            self._copy_auxiliary_files(self.auxiliary_files_names)

    def _copy_auxiliary_files(self, names):
        if names:
            for name in names:
                shutil.copyfile(f'{self.files_prefix}{name}', f'{self._tmp_folder}/{name}')

    def _close_client(self):
        if not self.done:
            utils.recursively_kill_process(self._client_process_pid)
        gc.collect()

    def _close_server(self):
        try:
            self._server.close()
            self._zmq_context.destroy()
        except Exception as e:
            logging.exception(e)

    def step(self, action_dict):
        assert isinstance(action_dict, dict)
        for available_action in self.action_variables:
            assert available_action in action_dict
        try:
            while True:
                action_js = json.dumps(action_dict, cls=utils.NumpyEncoder).encode('utf-8')
                self._server.send(action_js)
                observation, _state, done, context = self._get_state()  # context == ensemble of static features
                self.done = done
                if done:
                    self._close_client()
                    return None, None, self.done, None
                self.history['observation'].append(observation)
                self.history['action'].append(action_dict)
                self._history['state'].append(_state)
                self._history['action'].append(action_dict)
                reward = self._get_reward(_state)
                self.history['reward'].append(reward)
                self._history['reward'].append(reward)
                self.observation = observation
                self._state = _state
                self.reward = reward
                self.t += 1
                return observation, reward, done, context
        except Exception as e:
            logging.exception(e)

    def _reset_attributes(self):
        self.done = False
        self.t = 0
        self.history = {'observation': [], 'action': [], 'reward': []}
        self._history = {'state': [], 'action': [], 'reward': []}

    def reset(self, seed=None):
        self.set_seed(seed)
        if self.random_weather:
            self.rseed1 = self.random_generator.randint(1, 99999)
        if not self.done:
            self._close_client()
        self._write_pdi_yaml()
        self._reset_attributes()
        self._launch_client()
        self._server.send(b'')  # to respect REQ/REP send/receive/send/receive/... scheme
        self.observation, self.state_, self.done, self.context = self._get_state()
        return self.observation

    def reset_hard(self, seed=None):
        self.set_seed(seed)
        self.close()
        self._make_tmp_folder()
        self._write_fileX_template()
        if self.random_weather:
            self.rseed1 = self.random_generator.randint(1, 99999)
        self._get_sockets_()
        self._reset_attributes()
        self.observation, self.state_, self.done, self.context = self._get_state()
        return self.observation

    def close(self):
        self._close_server()
        self._close_client()
        shutil.rmtree(self._tmp_folder, ignore_errors=True)
        gc.collect()

    def set_seed(self, seed=None):
        self.random_generator, self.seed = seeding.np_random(seed)
        return self.seed

    def get_env_info(self):
        config_actions = self.config['action']
        config_states = self.config['state']
        print('\n******************')
        print('Available actions:')
        print('******************\n')
        for action in self.action_variables:
            pprint({action: config_actions[action]})
            input('press "return" to continue')
        print('\n*********************')
        print('Observation variables:')
        print('*********************\n')
        for state in self.observation_variables:
            pprint({state: config_states[state]})
            input('press "return" to continue')
        print('\n******************')
        print('Context variables:')
        print('******************\n')
        if self.context_variables:
            for state in self.context_variables:
                pprint({state: config_states[state]})
                input('press "return" to continue')
        else:
            print('no context information to display')
        print('\nno more information to display -> leaving\n')

    def render(self, type, *args, **kwargs):
        authorized_types = ['ts', 'reward']
        if type not in authorized_types:
            raise ValueError(f'"type" parameter has to be in {[*authorized_types]}!')
        if type == 'ts':
            rendering.render_temporal_series(history=self.history, *args, **kwargs)
        else:
            rendering.render_reward(history=self.history, *args, **kwargs)


    def observation_dict_to_array(self, dict):
        if dict:
            values = [dict[ordered_key] for ordered_key in self.observation_variables]
            return np.concatenate(values, axis=None)
        else:
            return []