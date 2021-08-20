import gym
import gym.spaces as spaces
from subprocess import Popen
import zmq
import json
from gym_dssat_pdi.envs.utils import serialize, write_template, get_time_stamp
import tempfile
import random
import shutil
import logging
import os
import datetime
import signal
import gc
import psutil

class DssatPdi(gym.Env):

    def __init__(self, run_dssat_location, experiment_number=1, file_X_prefix='UFGA8201', fileX_extension='.MZX',
                 log_saving_path=None, yml_template_path='./templates/dssat-pdi.jinja2',
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
        self.state = None
        self.done = False
        self.port = None
        self.context = None
        self.server = None
        self.client_process_pid = None
        self.files_prefix = files_prefix
        self.tmp_folder = None
        self._make_tmp_folder()
        self._get_sockets_()
        self._get_state()

    def _launch_client(self):
        print(f'Starting env client: port {self.port}')
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
                f_.write(get_time_stamp.get_time_stamp())
                f_.write('\n********************************\n')
            process = Popen(pdi_command, stdout=f_, shell=False, universal_newlines=True,
                                        cwd=self.tmp_folder)
            self.client_process_pid = process.pid

    def _launch_server(self):
        self.context = zmq.Context()
        self.server = self.context.socket(zmq.PAIR)
        # self.server.setsockopt(zmq.LINGER, 0)
        # self.server.bind('tcp://*:5555')
        self.port = self.server.bind_to_random_port('tcp://*', max_tries=10000) #min_port=1024, max_port=65535)
        if self.port is None:
            print('Server failed to find a free port')

    def _write_pdi_yaml(self):
        value_dic = {'port': self.port}
        write_template.write_template(value_dic=value_dic,
                                      template_path=self.yml_template_path,
                                      saving_path=f'{self.tmp_folder}/dssat-pdi.yml')

    def _get_state(self):
        message = self.server.recv().decode('utf-8')
        message = json.loads(message)
        state = message['state']
        self.state = state
        self.done = message['done']
        return state

    def _get_reward(self):
        state = self.state
        return 10

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
                if self.done:
                    self.close()
                    return None, None, self.done, None
                action = json.dumps(action, default=serialize.convert).encode('utf-8')
                self.server.send(action)
                state = self._get_state()
                reward = self._get_reward()
                done = self.done
                info = self._get_info()
                return state, reward, done, info
        except Exception as e:
            logging.exception(e)

    def reset(self):
        self.done = False
        self._close_client()
        self._launch_client()
        self._get_state()

    def render(self):
        pass

    def close(self):
        self._close_server()
        self._close_client()
        shutil.rmtree(self.tmp_folder, ignore_errors=True)
        gc.collect()

    def _close_client(self):
        try:
            self.recursively_kill_process(self.client_process_pid)
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

    @staticmethod
    def recursively_kill_process(parent_pid):
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
