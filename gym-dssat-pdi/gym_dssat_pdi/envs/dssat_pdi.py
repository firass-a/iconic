import gym
import gym.spaces as spaces
from gym_dssat_pdi.envs.utils import utils, rendering
from gym_dssat_pdi.envs.configs import rewards
import numpy as np
from gym.utils import seeding
import subprocess
import zmq
import json
import yaml
import tempfile
import shutil
import logging
import os
import pkgutil
from pprint import pprint
import psutil
import warnings
import sys
import atexit

__copyright__ = 'Copyright CGIAR, Inria and CIRAD'
__credits__ = [
    'Romain Gautron',
    'Emilio Padrón González',
]
__license__ = 'BSD 3-Clause'
__author__ = 'Romain Gautron <romain.gautron@cirad.fr>'


class DssatPdi(gym.Env):

    def __init__(self, run_dssat_location, log_saving_path=None, mode='all', auxiliary_file_paths=None,
                 files_prefix='./', random_weather=True, seed=None, fileX_template_path=None, experiment_number=None):
        self.experiment_number = experiment_number
        self.mode = mode
        self.action_variables = None
        self.observation_variables = None
        self.context_variables = None
        if fileX_template_path is None:
            self._fileX_template = pkgutil.get_data(__name__, f'configs/UFGA8201.jinja2').decode('utf-8')
        else:
            self._fileX_template = utils._load_fileX_template(fileX_template_path)
        self._fileX = None
        self._pdi_yaml_template = pkgutil.get_data(__name__, f'configs/dssat_pdi.jinja2').decode('utf-8')
        self._pdi_yaml = None
        self._env_yaml_config = pkgutil.get_data(__name__, f'configs/env_config.yml').decode('utf-8')
        self._config = None
        self._load_config()
        self.observation_space = None
        self.context_space = None
        self.action_space = None
        for key in ['observation', 'context', 'action']:
            self._make_gym_spaces(key=key)
        if auxiliary_file_paths is not None:
            self.auxiliary_file_paths = auxiliary_file_paths
        else:
            self.auxiliary_file_paths = []
        self._run_dssat_location = run_dssat_location
        self.log_saving_path = log_saving_path
        self._cwd = os.getcwd()
        self._reward_func = rewards.get_reward_function(mode)
        self.history = {'observation': [], 'action': [], 'reward': []}
        self._history = {'state': [], 'action': [], 'reward': []}
        self._random_generator = None
        self.seed = None
        self.set_seed(seed=seed)
        self._rseed1 = self._random_generator.randint(1, 99999)
        self.random_weather = random_weather
        self.wther = 'W' if random_weather else 'M'
        self.ferti = 'L' if mode in ['all', 'fertilization'] else 'R'
        self.irrig = 'L' if mode in ['all', 'irrigation'] else 'R'
        self._is_early_stopping = False
        self._early_stopped = False
        self.done = False
        self.closed = False
        self._f_out = None
        self.t = 0
        self.reset_counter = 0
        self._port = None
        self._zmq_context = None
        self._server = None
        self._poller = None
        self._last_is_send = True
        self._client_process_pid = None
        self._files_prefix = files_prefix
        self._tmp_folder = None
        self._make_tmp_folder()
        self._make_fileX_template()
        self._write_fileX_template()
        self._get_sockets_()
        self.observation, self._state, self.done, self.context = self._get_state()

    def _load_config(self):
        config = yaml.load(self._env_yaml_config, Loader=yaml.FullLoader)
        self._config = config
        setting_dict = self._config['setting']
        setting = self.mode
        if setting not in setting_dict:
            raise ValueError(f'Authorized values for the "mode" parameter  to be in {[*setting_dict]}')
        self.observation_variables = sorted(setting_dict[setting]['state'])
        self.action_variables = setting_dict[setting]['action']
        self.context_variables = setting_dict[setting]['context']
        if self.experiment_number is None:
            self.experiment_number = setting_dict[setting]['experiment_number']

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
        key_data = self._config[key_config]
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
                    space = spaces.Box(low=low, high=high, shape=(size,))
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
        self._fileX = utils._fill_template_from_string(value_dic=fileX_template_values,
                                                       template_string=self._fileX_template)

    def _write_fileX_template(self):
        utils.save_file(saving_path=f'{self._tmp_folder}/fileX.MZX', content=self._fileX)

    def _launch_client(self):
        pdi_command = f'sh {self._run_dssat_location} C fileX.MZX {self.experiment_number}'
        pdi_command = pdi_command.split(' ')
        if self.log_saving_path is not None:
            file_path = self.log_saving_path
        else:
            file_path = os.devnull
        self._f_out = open(file_path, 'a+')
        if self.log_saving_path is not None:
            self._f_out.write('\n********************************\n')
            self._f_out.write(utils.get_time_stamp())
            self._f_out.write('\n********************************\n')
        client_process = subprocess.Popen(pdi_command,
                                          stdout=self._f_out,
                                          stderr=sys.stderr,
                                          shell=False,
                                          universal_newlines=True,
                                          cwd=self._tmp_folder,
                                          bufsize=0,
                                          )
        self._client_process_pid = client_process.pid
        atexit.register(self._cleanup_process)

    def _cleanup_process(self):
        if self._client_process_pid is not None and psutil.pid_exists(self._client_process_pid):
            utils.recursively_kill_process(self._client_process_pid)
            # os.killpg(self._client_process_pid, signal.SIGKILL)

    def _launch_server(self):
        self._zmq_context = zmq.Context()
        self._server = self._zmq_context.socket(zmq.PAIR)
        # self._server.setsockopt(zmq.LINGER, 0)
        self._port = self._server.bind_to_random_port('tcp://*', max_tries=10000)  # min_port=1024, max_port=65535)

    def _launch_poller(self):
        self._poller = zmq.Poller()
        self._poller.register(self._server, zmq.POLLIN)

    def _write_pdi_yaml(self):
        value_dic = {'port': self._port,
                     'rseed1': self._rseed1,
                     }
        self._pdi_yaml = utils._fill_template_from_string(value_dic=value_dic,
                                                          template_string=self._pdi_yaml_template)
        utils.save_file(saving_path=f'{self._tmp_folder}/dssat-pdi.yml', content=self._pdi_yaml)

    def _get_state(self):
        if not self._last_is_send:
            self.close()
            raise ValueError("you cannot call env._get_state() two times in a row!")
        # if self._poller.poll(timeout=1000):
        message = self._server.recv().decode('utf-8')
        self._last_is_send = False
        # else:
        #     pdb.set_trace()
        #     self.close()
        #     raise ValueError("client doesn't send message")
        message = json.loads(message, object_hook=utils.NumpyDecoder)
        done = message['done']
        _state = message['state']
        if _state:
            _state = utils._post_treat_state(_state)
            observation = utils._filter_state(full_state=_state,
                                              observation_variables=self.observation_variables)
            context = utils._filter_state(full_state=_state,
                                          observation_variables=self.context_variables)
        else:
            observation = {}
            context = {}
        return observation, _state, done, context

    def _get_reward(self, _next_state):
        _previous_state = self._state
        _history = self._history
        reward = self._reward_func(_previous_state, _next_state, _history)
        return reward

    def _get_sockets_(self):
        self._launch_server()
        self._launch_poller()
        self._write_pdi_yaml()
        self._launch_client()
        # if not self._poller.poll(timeout=10000):
        #     self.close()
        #     self._get_sockets_()

    def _make_tmp_folder(self):
        if self._tmp_folder:
            shutil.rmtree(self._tmp_folder, ignore_errors=True)
        self._tmp_folder = tempfile.mkdtemp()
        if self.auxiliary_file_paths:
            self._copy_auxiliary_files()

    def _copy_auxiliary_files(self):
        for path in self.auxiliary_file_paths:
            file_name = path.split('/')[-1]
            shutil.copyfile(path, f'{self._tmp_folder}/{file_name}')

    def _reset_attributes(self):
        self.closed = False
        self._is_early_stopping = False
        self._early_stopped = False
        self.done = False
        self.t = 0
        self.history = {'observation': [], 'action': [], 'reward': []}
        self._history = {'state': [], 'action': [], 'reward': []}

    def _close_client(self):
        self._early_stopping()
        if self._client_process_pid and psutil.pid_exists(self._client_process_pid):
            psutil.wait_procs([psutil.Process(self._client_process_pid)], timeout=1)
            self._client_process_pid = None
        if not self._last_is_send:
            self._server.send(b'')
            self._last_is_send = True

    def _close_server(self):
        if self._server:
            self._server.close()
            self._server = None
        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None

    def _close_poller(self):
        if self._poller is not None:
            self._poller.unregister(self._server)
            self._poller = None

    def _close_tmp_folder(self):
        if self._tmp_folder:
            shutil.rmtree(self._tmp_folder)
            self._tmp_folder = None

    def _early_stopping(self):
        if self.done:
            self._server.send(f'{self._rseed1}'.encode('utf-8'))
            self._server.recv()
            self._last_is_send = False
        self._is_early_stopping = True
        message = {'early_stopping': True,
                   'action': {action: 0 for action in self.action_variables}}
        message_js = json.dumps(message, cls=utils.NumpyEncoder).encode('utf-8')
        self._server.send(message_js)
        self._last_is_send = True
        self._early_stopped = True

    def _get_env_done(self):
        default_action_dict = {'amir': 0, 'anfer': 0}
        while not self.done:
            message = {'early_stopping': self._is_early_stopping, 'action': default_action_dict}
            message_js = json.dumps(message, cls=utils.NumpyEncoder).encode('utf-8')
            self._server.send(message_js)
            self._last_is_send = True
            observation, _state, done, context = self._get_state()
            self.done = done

    def close(self, _close_tmp=True):
        if not self.closed:
            self._close_client()
            self._close_poller()
            self._close_server()
            if _close_tmp:
                self._close_tmp_folder()
            self.closed = True
        if self._f_out is not None and not self._f_out.closed:
            self._f_out.close()
            self._f_out = None


    def step(self, action_dict):
        if self.closed:
            raise ValueError('Environment has been previously closed, please call env.reset() or env.reset_hard()'
                             ' prior to calling env.step(...)')
        if self.done:
            warnings.warn("Warning: the environment is done ; you may call env.reset() or env.reset_hard()")
        assert isinstance(action_dict, dict)
        for available_action in self.action_variables:
            assert available_action in action_dict
        try:
            if not self.done:
                message = {'early_stopping': self._is_early_stopping, 'action': action_dict}
                message_js = json.dumps(message, cls=utils.NumpyEncoder).encode('utf-8')
                self._server.send(message_js)
                self._last_is_send = True
                observation, _state, done, context = self._get_state()  # context == ensemble of static features
                self.done = done
                if done:
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
            else:
                return None, None, self.done, None
        except Exception as e:
            logging.exception(e)

    def get_state(self):
        return self.observation

    def reset(self, seed=None):
        if self.closed: # or self.reset_counter >= 10:  # dirty fix for memory leak
            self.reset_hard()
            self.reset_counter = 0
        else:
            if not self.done:
                self._get_env_done()
            self.set_seed(seed)
            if self.random_weather:
                self._rseed1 = self._random_generator.randint(1, 99999)
            self._server.send(f'{self._rseed1}'.encode('utf-8'))
            self._last_is_send = True
            self.reset_counter += 1
            self._reset_attributes()
            self.observation, self.state_, self.done, self.context = self._get_state()
            return self.observation

    def reset_hard(self, seed=None, _new_tmp_folder=True):
        self.set_seed(seed)
        if not self.closed:
            self.close(_close_tmp=_new_tmp_folder)
        if _new_tmp_folder:
            self._make_tmp_folder()
        if self.random_weather:
            self._rseed1 = self._random_generator.randint(1, 99999)
        self._write_fileX_template()
        self._reset_attributes()
        self._get_sockets_()
        self.observation, self.state_, self.done, self.context = self._get_state()
        return self.observation

    def set_seed(self, seed=None):
        if seed is not None:
            seed = int(seed)
            self.seed = seed
        self._random_generator, self.seed = seeding.np_random(seed)
        return self.seed

    def get_env_info(self):
        config_actions = self._config['action']
        config_states = self._config['state']
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
            rendering.render_temporal_series(_history=self._history, *args, **kwargs)
        else:
            rendering.render_reward(_history=self._history, *args, **kwargs)

    def observation_dict_to_array(self, dict):
        if dict:
            values = [dict[ordered_key] for ordered_key in self.observation_variables]
            return np.concatenate(values, axis=None)
        else:
            return []
