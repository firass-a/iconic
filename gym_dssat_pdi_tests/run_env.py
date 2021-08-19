import gym


def fertilization_policy(YRDOY):
    fertilization_dic = {
        1982097: 27,
        1982102: 35,
        1982137: 54,
    }
    if YRDOY not in fertilization_dic:
        anfer = 0
    else:
        anfer = fertilization_dic[YRDOY]
    return {'anfer': anfer}


if __name__ == '__main__':
    env = gym.make('gym_dssat_pdi:GymDssatPdi-v0')
    run_dssat_location = '/home/rgautron/dssat_pdi/run_dssat'
    env._init_(run_dssat_location=run_dssat_location)
    done = False
    try:
        while not done:
            state = env.state
            YRDOY = state['YRDOY']
            action = fertilization_policy(YRDOY)
            print(f'YRDOY : {YRDOY} -> fertilizing {action["anfer"]} kgN/ha')
            res = env.step(action)
            _, reward, done, info = res
            # print(res)
    finally:
        env.close()
