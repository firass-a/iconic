import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gym_dssat_pdi.envs.utils.utils import transpose_dicts, make_folder
import numpy as np
import pathlib
import pdb

__copyright__ = 'Copyright CGIAR, Inria and CIRAD'
__credits__ = [
    'Romain Gautron',
    'Emilio J. Padron',
]
__license__ = 'BSD 3-Clause'
__author__ = 'Romain Gautron <romain.gautron@cirad.fr>'

def render_temporal_series(_history, mode, feature_name_1, feature_name_2=None, layout_dict=None, saving_path=None,
                           folder_path='./render', *args, **kwargs):
    if folder_path != './':
        make_folder(folder_path)
    trajectory = _history['state']
    if not trajectory:
        raise ValueError(f'environment history is empty')
    authorized_keys = [*trajectory[0]]
    if feature_name_1 not in authorized_keys:
        raise ValueError(f'feature_name_1 "{feature_name_1}" not in {authorized_keys}')
    if feature_name_2 is not None and feature_name_2 not in authorized_keys:
        raise ValueError(f'feature_name_2 "{feature_name_2}" not in {authorized_keys}')
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
    plt.title(f'State feature evolution for mode {mode}')
    if saving_path is None:
        saving_path = f'{folder_path}/{feature_name_1}{feature_name_2_label}_DOY_mode_{mode}.pdf'
    plt.savefig(saving_path, bbox_inches='tight')

def render_reward(_history, mode, saving_path=None, folder_path='./render', cumsum=True, *args, **kwargs):
    make_folder(folder_path)
    trajectory = _history['state']
    trajectory = transpose_dicts(trajectory)
    reward = _history['reward']
    x = [int(str(DOY)[-3:]) for DOY in trajectory['yrdoy']]
    y = np.asarray(reward)
    fig, ax1 = plt.subplots()
    if mode == 'all':
        ax2 = ax1.twinx()
        y = y.T
        y1 = y[0]
        y2 = y[1]
        y1_label = 'fertilization reward'
        if cumsum:
            y2 = np.cumsum(y2)
        ax2.plot(x, y2, 'b-')
        ax2.set_ylabel('irrigation reward', color='b')
    else:
        y1_label = f'{mode} reward'
        y1 = y
    if cumsum:
        y1 = np.cumsum(y1)
    ax1.plot(x, y1, 'g-')
    ax1.set_xlabel('DOY')
    ax1.set_ylabel(y1_label, color='g')
    plt.title(f'Rewards for mode {mode}')
    cumsum_label = ''
    if cumsum:
        cumsum_label = '_cumsum'
    if saving_path is None:
        saving_path = f'{folder_path}/reward_DOY{cumsum_label}_mode_{mode}.pdf'
    plt.savefig(saving_path, bbox_inches='tight')

if __name__ == '__main__':
    pass
