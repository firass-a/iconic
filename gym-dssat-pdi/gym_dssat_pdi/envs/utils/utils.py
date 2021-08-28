import os
import signal
import datetime
import psutil
from numpy import array
import jinja2
# from jinja2 import Environment, BaseLoader
import yaml
import pdb
import json
import numpy as np
from pprint import pprint

__copyright__ = 'Copyright CGIAR, Inria and CIRAD'
__credits__ = [
    'Romain Gautron',
    'Emilio Padrón González',
]
__license__ = 'BSD 3-Clause'
__author__ = 'Romain Gautron <romain.gautron@cirad.fr>'

class DssatPdiHandler:
    """
    inspired from https://stackoverflow.com/questions/320232/ensuring-subprocesses-are-dead-on-exiting-python-program
    """

    def __enter__(self):
        os.setpgrp()

    def __exit__(self, type, value, traceback):
        try:
            os.killpg(0, signal.SIGTERM)
        except KeyboardInterrupt:
            pass
        except:
            os.killpg(0, signal.SIGKILL)

def _fill_template_from_file(value_dic, template_path):
    with open(template_path, mode='r') as f_:
        template = jinja2.Template(f_.read(), trim_blocks=True, lstrip_blocks=True)
    filled_template = template.render(**value_dic)
    return filled_template

def _fill_template_from_string(value_dic, template_string):
    template = jinja2.Environment(loader=jinja2.BaseLoader).from_string(template_string)
    filled_template = template.render(**value_dic)
    return filled_template

def _write_template_from_string(value_dic, template_string, saving_path):
    filled_template = _fill_template_from_string(value_dic, template_string)
    save_file(saving_path, filled_template)

def _write_template_from_file(value_dic, template_path, saving_path):
    filled_template = _fill_template_from_file(value_dic, template_path)
    save_file(saving_path, filled_template)

def save_file(saving_path, content):
    with open(saving_path, mode='w') as f_:
        f_.write(content)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_time_stamp():
    now = datetime.datetime.now()
    return now.strftime('%Y/%m/%d %H:%M:%S.%f')


def recursively_kill_process(parent_pid):
    parent = psutil.Process(parent_pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    parent.kill()


def transpose_dicts(dict_list):
    keys = dict_list[0].keys() if len(dict_list) > 0 else []
    vals = zip(*map(lambda x: x.values(), dict_list))
    transposed_dict_list = dict(zip(keys, vals))
    return transposed_dict_list


def _post_treat_state(state):
    state['grnwt'] *= state['pltpop']
    state['nstres'] = 1 - state['nstres']
    state['swfac'] = 1 - state['swfac']
    state['pcngrn'] /= 100
    state['wtnup'] *= 10
    state['trnu'] *= 10 * state['pltpop']
    state['ds_max'] = max(state['ds'])
    return state


def _filter_state(full_state, observation_variables):
    truncated_state = {key: full_state[key] for key in observation_variables}
    return truncated_state


def _parse_config(path_to_load):
    with open(path_to_load, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    pass
