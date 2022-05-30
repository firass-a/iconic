import pdb
import numpy as np

__copyright__ = 'Copyright CGIAR, Inria and CIRAD'
__credits__ = [
    'Romain Gautron',
    'Emilio J. Padron',
]
__license__ = 'BSD 3-Clause'
__author__ = 'Romain Gautron <romain.gautron@cirad.fr>'


def fertilization_reward(_previous_state, _next_state, _history):
    if _next_state:
        last_action = _history['action'][-1]['anfer']
        penality = .5
        trnu = _next_state['trnu']
        return trnu - penality * last_action
    return None


def irrigation_reward(_previous_state, _next_state, _history):
    if _next_state:
        last_action = _history['action'][-1]['amir']
        previous_topwt = _previous_state['topwt']
        next_topwt = _next_state['topwt']
        penality = 15
        return next_topwt - previous_topwt - penality * last_action
    return None



def all_reward(_previous_state, _next_state, _history):
    ferti_reward_value = fertilization_reward(_previous_state, _next_state, _history)
    irrig_reward_value = irrigation_reward(_previous_state, _next_state, _history)
    return [ferti_reward_value, irrig_reward_value]


def get_reward_function(mode):
    reward_func_dic = {'all': all_reward,
                       'fertilization': fertilization_reward,
                       'irrigation': irrigation_reward}
    if mode not in reward_func_dic:
        raise ValueError(f'"mode" parameter has to be in {[*reward_func_dic]}!')
    return reward_func_dic[mode]


if __name__ == '__main__':
    pass
