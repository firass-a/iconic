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
    penality = 5
    if _next_state:
        tleachd = _next_state['tleachd']
        tnoxd = _next_state['tnoxd']
        trnu = _next_state['trnu']
        reward = (trnu - (tleachd + tnoxd)) - penality * bool(last_action)
    return reward


def irrigation_reward(_previous_state, _next_state, _history):
    reward = None
    last_action = _history['action'][-1]['anfer']
    penality = 0
    if _next_state:
        rtdep = _next_state['rtdep']
        if rtdep < 1:
            return 0
        ll = _next_state['ll']
        dul = _next_state['dul']
        dlayr = _next_state['dlayr']
        dlayr_cumsum = dlayr.cumsum()
        sw = _next_state['sw']
        swfac = _next_state['swfac']
        layer_with_roots = dlayr_cumsum < rtdep
        first_greater_layer_bottom = np.argmax(np.logical_not(layer_with_roots))
        last_layer_depth_with_root_proportion = dlayr_cumsum[first_greater_layer_bottom]
        last_layer_depth_porportion_with_root = (last_layer_depth_with_root_proportion - rtdep) / dlayr[
            first_greater_layer_bottom]
        pawr = ((dul - ll) * dlayr * layer_with_roots).sum()
        pawr += ((dul - ll) * dlayr)[first_greater_layer_bottom] * last_layer_depth_porportion_with_root
        swr = ((sw - ll) * dlayr * layer_with_roots).sum()
        swr += ((sw - ll) * dlayr)[first_greater_layer_bottom] * last_layer_depth_porportion_with_root
        ratio = swr / pawr
        # print(ratio)
        if ratio <= .2:
            ratio_penality = - 10
        elif ratio > .9:
            ratio_penality = - 1
        else:
            ratio_penality = 0
        try:
            # reward = np.exp(-1 / (1 - nstres) ** 2) - 4 * ratio ** 2 + 4 * ratio - penality * bool(last_action)
            # reward = - 4 * ratio ** 2 + 4 * ratio - penality * bool(last_action)
            # print(penality * bool(last_action))
            reward = ratio + 1 - swfac + ratio_penality
        except Exception as e:
            print(e)
        # print(reward)
    return reward

def all_reward(_previous_state, _next_state, _history):
    ferti_reward = fertilization_reward(_previous_state, _next_state, _history)
    irrig_reward = irrigation_reward(_previous_state, _next_state, _history)
    if ferti_reward is None or irrig_reward is None:
        return
    return ferti_reward + irrig_reward


def get_reward_function(mode):
    reward_func_dic = {'all': all_reward,
                       'fertilization': fertilization_reward,
                       'irrigation': irrigation_reward}
    if mode not in reward_func_dic:
        raise ValueError(f'"mode" parameter has to be in {[*reward_func_dic]}!')
    return reward_func_dic[mode]


if __name__ == '__main__':
    pass
