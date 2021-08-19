import yaml
import pdb

def make_action_space():
    pass

if __name__ == '__main__':
    with open("env_config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        print(config)
    actions = config['actions']
    state = config['state']