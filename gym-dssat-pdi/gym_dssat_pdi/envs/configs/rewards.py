import pdb
import numpy as np

__copyright__ = 'Copyright CGIAR, Inria and CIRAD'
__credits__ = [
    'Romain Gautron',
    'Emilio Padrón González',
]
__license__ = 'BSD 3-Clause'
__author__ = 'Romain Gautron <romain.gautron@cirad.fr>'


def fertilization_reward(_previous_state, _next_state, _history):
    reward = None
    last_action = _history['action'][-1]['anfer']
    penality = .4
    if _next_state:
        tleachd = _next_state['tleachd']
        tnoxd = _next_state['tnoxd']
        trnu = _next_state['trnu']
        reward = (trnu - (tleachd + tnoxd)) - penality * last_action
    return reward


def irrigation_reward(_previous_state, _next_state, _history):
    reward = None
    last_action = _history['action'][-1]['amir']
    penality1 = 2
    penality2 = 4
    istage = _next_state["istage"]
    print(f'istage {istage}')
    if istage not in [9, 1, 2, 3, 4]:
        return -penality2
    if _next_state:
        rtdep = _next_state['rtdep']
        if rtdep < 1:
            return
        ll = _next_state['ll']
        dul = _next_state['dul']
        dlayr = _next_state['dlayr']
        dlayr_cumsum = dlayr.cumsum()
        sw = _next_state['sw']
        swfac = _next_state['swfac']
        layer_with_roots = dlayr_cumsum < rtdep
        first_greater_layer_index = np.argmax(np.logical_not(layer_with_roots))
        last_layer_with_root_top = dlayr_cumsum[first_greater_layer_index - 1]
        last_layer_with_root_thickness_proportion = (rtdep - last_layer_with_root_top) / dlayr[
            first_greater_layer_index]
        pawr = ((dul - ll) * dlayr * layer_with_roots).sum()
        pawr += ((dul - ll) * dlayr)[first_greater_layer_index] * last_layer_with_root_thickness_proportion
        swr = ((sw - ll) * dlayr * layer_with_roots).sum()
        swr += ((sw - ll) * dlayr)[first_greater_layer_index] * last_layer_with_root_thickness_proportion
        ratio = swr / pawr
        reward = 1 - swfac
        # print(swfac, last_action, ratio)
        if swfac == 0 and last_action > 0 and ratio > .7:
            reward -= penality1
        if swfac > .7:
            reward -= penality2
    return reward


def all_reward(_previous_state, _next_state, _history):
    ferti_reward_value = fertilization_reward(_previous_state, _next_state, _history)
    irrig_reward_value = irrigation_reward(_previous_state, _next_state, _history)
    all_reward_value = ferti_reward_value + 2 * irrig_reward_value
    return all_reward_value


def get_reward_function(mode):
    reward_func_dic = {'all': all_reward,
                       'fertilization': fertilization_reward,
                       'irrigation': irrigation_reward}
    if mode not in reward_func_dic:
        raise ValueError(f'"mode" parameter has to be in {[*reward_func_dic]}!')
    return reward_func_dic[mode]


if __name__ == '__main__':
    pass
