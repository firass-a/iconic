import os
import signal
import datetime
import psutil
from numpy import array
import jinja2

class DssatPdiHandler:
    """
    from https://stackoverflow.com/questions/320232/ensuring-subprocesses-are-dead-on-exiting-python-program
    """
    def __enter__(self):
        os.setpgrp()

    def __exit__(self, type, value, traceback):
        try:
            os.killpg(0, signal.SIGTERM)
        except KeyboardInterrupt:
            pass

def write_template(value_dic, template_path, saving_path):
    with open(template_path) as f_:
        template = jinja2.Template(f_.read(), trim_blocks=True, lstrip_blocks=True)
    output = template.render(**value_dic)
    with open(saving_path, mode='w') as f_:
        f_.write(output)

def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        x = x.tolist()
    return x

def deconvert(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return array(value)  # cast back to array
    return x

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
    return state

if __name__ == '__main__':
    pass