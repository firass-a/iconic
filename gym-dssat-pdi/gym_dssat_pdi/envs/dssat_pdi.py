import gym
import gym.spaces as spaces
from subprocess import Popen
import zmq
import json
from gym_dssat_pdi.envs.utils import serialize, write_template
import logging
import pdb


class DssatPdi(gym.Env):
    # def __init__(self):
    #     pass

    def __init__(self, run_dssat_location, experiment_number=1, file_X_prefix='UFGA8201', fileX_extension='.MZX'):
        self.action_space = spaces.Dict({'anfer': spaces.Box(low=0, high=200, shape=())})
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.experiment_number = experiment_number
        self.file_X = f'{file_X_prefix}{fileX_extension}'
        self.run_dssat_location = run_dssat_location
        self.state = None
        self.done = False
        self.port = None
        self.context = None
        self.server = None
        self.client_process = None
        self.poller = None
        self.launch_server()
        self.write_pdi_yaml()
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
        self.server = self.context.socket(zmq.PAIR)
        self.server.setsockopt(zmq.LINGER, 0)
        # self.server.bind('tcp://*:5555')
        self.port = self.server.bind_to_random_port('tcp://*', min_port=1024, max_port=65535, max_tries=100)

    def write_pdi_yaml(self):
        value_dic = {'port': self.port}
        print(f'server: {self.port}')
        write_template.write_template(value_dic=value_dic,
                                      template_path='./templates/dssat-pdi.jinja2',
                                      saving_path='./dssat-pdi.yml')

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
