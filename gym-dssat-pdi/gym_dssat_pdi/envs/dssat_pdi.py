import gym
from subprocess import Popen
import zmq
import json
from gym_dssat_pdi.envs.utils import serialize
import logging


class DssatPdi(gym.Env):
    def __init__(self):
        pass

    def _init_(self, run_dssat_location, experiment_number=1, file_X_prefix='UFGA8201', fileX_extension='.MZX'):
        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.experiment_number = experiment_number
        self.file_X = f'{file_X_prefix}{fileX_extension}'
        self.run_dssat_location = run_dssat_location
        self.state = None
        self.context = None
        self.server = None
        self.client_process = None
        self.poller = None
        self.launch_server()
        self.launch_client()
        self._get_state()

    def launch_client(self):
        print('Starting env client')
        pdi_command = f'pdirun {self.run_dssat_location} C {self.file_X} {self.experiment_number}'
        with open('./dssat_pdi.log', 'w') as f_:
            Popen(pdi_command, stdout=f_, shell=True)  # puts all shell outputs to trash

    def launch_server(self):
        print('Starting env server')
        self.context = zmq.Context()
        self.server = self.context.socket(zmq.REP)
        self.server.setsockopt(zmq.LINGER, 0)
        self.server.bind('tcp://*:5555')

    def _get_state(self):
        state = self.server.recv().decode('utf-8')
        state = json.loads(state)
        self.state = state
        return state

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
                done = self.state['done']
                if done:
                    self.close()
                    return None, None, done, None
                action = json.dumps(action, default=serialize.convert).encode('utf-8')
                self.server.send(action)
                state = self._get_state()
                reward = self.get_reward()
                return state, reward, done, {}
        except Exception as e:
            logging.exception(e)

    def get_reward(self):
        state = self.state
        return 10

    def get_info(self):
        state = self.state
        return {}

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        self.server.close()
        self.context.term()
        # self.client_process.kill()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]