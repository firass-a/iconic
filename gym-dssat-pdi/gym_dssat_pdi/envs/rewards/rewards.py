import pdb

def fertilization_reward(previous_state, next_state, history):
    reward = None
    last_action = history['action'][-1]['anfer']
    penality = 5
    if next_state:
        tleachd = next_state['tleachd']
        tnoxd = next_state['tnoxd']
        trnu = next_state['trnu']
        reward = (trnu - (tleachd + tnoxd)) - penality * bool(last_action)
    return reward

def get_reward_function(mode):
    reward_func_dic = {'all': fertilization_reward, 'fertilization': fertilization_reward,
                       'irrigation': fertilization_reward}
    if mode not in reward_func_dic:
        raise ValueError('"mode" parameter should be in [*reward_func_dic]!')
    return reward_func_dic[mode]

if __name__ == '__main__':
    pass
