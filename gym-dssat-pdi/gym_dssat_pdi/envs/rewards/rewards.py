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


if __name__ == '__main__':
    pass
