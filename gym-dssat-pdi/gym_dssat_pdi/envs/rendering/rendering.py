import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gym_dssat_pdi.envs.utils.utils import transpose_dicts
import numpy as np
import pathlib
import pdb

def make_render_folder(folder_path):
    path = pathlib.Path(folder_path)
    path.mkdir(exist_ok=True)

def render_temporal_series(history, feature_name_1, feature_name_2=None, layout_dict=None, saving_path=None,
                           folder_path='./render', *args, **kwargs):
    make_render_folder(folder_path)
    trajectory = history['state']
    trajectory = transpose_dicts(trajectory)
    y1 = trajectory[feature_name_1]
    x = [int(str(DOY)[-3:]) for DOY in trajectory['yrdoy']]
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, 'g-')
    ax1.set_xlabel('DOY')
    ax1.set_ylabel(feature_name_1, color='g')
    feature_name_2_label = ''
    if feature_name_2 is not None:
        ax2 = ax1.twinx()
        y2 = trajectory[feature_name_2]
        ax2.plot(x, y2, 'b-')
        ax2.set_ylabel(feature_name_2, color='b')
        feature_name_2_label = f'_{feature_name_2}'
    if saving_path is None:
        saving_path = f'{folder_path}/{feature_name_1}{feature_name_2_label}_DOY.pdf'
    plt.savefig(saving_path, bbox_inches='tight')

def render_reward(history, saving_path=None, folder_path='./render', cumsum=True, *args, **kwargs):
    make_render_folder(folder_path)
    trajectory = history['state']
    trajectory = transpose_dicts(trajectory)
    reward = history['reward']
    x = [int(str(DOY)[-3:]) for DOY in trajectory['yrdoy']]
    y = reward
    if cumsum:
        y = np.cumsum(y)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'g-')
    ax.set_xlabel('DOY')
    ax.set_ylabel('reward', color='g')
    cumsum_label = ''
    if cumsum:
        cumsum_label = '_cumsum'
    if saving_path is None:
        saving_path = f'{folder_path}/reward_DOY{cumsum_label}.pdf'
    plt.savefig(saving_path, bbox_inches='tight')

if __name__ == '__main__':
    pass