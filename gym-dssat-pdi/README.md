# gym-DSSAT: an easy to manipulate crop environment for Reinforcement Learning
gym-DSSAT is a modification of the [Decision Support System for Agrotechnology Transfer (DSSAT)](https://dssat.net/) software into an easy to manipulate [Open AI gym](https://gym.openai.com/) environment for Reinforcement Learning (RL) researchers in Python. gym-DSSAT allows daily based interactions during the growing season between an RL agent and the crop model with usual gym conventions.

gym-DSSAT is powered by the [PDI Data Interface (PDI)](https://pdi.julien-bigot.fr/master/) !

**Disclaimer: gym-DSSAT only supports Unix systems and uses Python 3.6 or above!**

## In this package
+ ```./gym-dssat-pdi```: the custom gym-DSSAT Python environment
  [Official repository](https://gitlab.inria.fr/rgautron/gym_dssat_pdi)

## Required Python modules included in this package
    - gym
    - numpy
    - matplotlib
    - jinja2
    - pyzmq

## Related packages
+ ```libpdi```: PDI Data Interface: library for decoupling high-performance simulation codes from Input/Output concerns. (REQUIRED)
+ ```dssat-pdi```: Modified version of DSSAT with PDI snippets for interacting with gym-DSSAT (REQUIRED)
+ ```dssat-csm-data```: the required experimental files used by DSSAT
+ ```gym-dssat-pdi-examples```: examples of how to run gym-DSSAT

## The environment
gym-DSSAT is designed to allow great setting flexibility. The environment comes with default settings that can easily been modified by editing the [gym environment's yaml configuration file](https://gitlab.inria.fr/rgautron/gym_dssat_pdi/-/blob/stable/gym-dssat-pdi/gym_dssat_pdi/envs/configs/env_config.yml). The ```action``` key gives the raw action space, the ```state``` key give the raw action space and each individual setting is found and can be edited in the ```setting``` key. Furthermore, the ```context``` key defines additional contextual variables.

gym-DSSAT uses by default the UFGA8201 maize experiment from the University of Florida, but is usable with any DSSAT experiment using the CERES-Maize module.

#### Action/State spaces

The environment comes with 3 modes:
+ nitrogen fertilization only (continuous quantity): ```mode=='fertilization'``` => nitrogen fertilizer quantity (kg/ha)
+ irrigation only (continuous quantity): ```mode=='irrigation'``` => water quantity (mm)
+ both nitrogen fertilization and irrigation (both continuous quantities): ```mode=='all'```

The action/state spaces depend on each mode and are detailed in [gym environment's yaml configuration file](https://gitlab.inria.fr/rgautron/gym_dssat_pdi/-/blob/stable/gym-dssat-pdi/gym_dssat_pdi/envs/configs/env_config.yml). State variables can be continuous, discrete and arrays of arbitrary shapes. Actions are continuous. By default, the observed state is given as a dictionnary as show below:

```python
{'cleach': 39.01179885864258,
'cnox': 0.07707925885915756,
'cumsumfert': 116.0,
'dap': 126,
'dtt': 20.900001525878906
...}
```

#### gym-DSSAT yaml configuration file structure
```yaml
action:
  anfer:
    type: float
    low: 0
    high: 200
    info: nitrogen to fertilize for current day (kg/ha)
...
state:
  cleach:
    type: float
    low: 0
    high: .inf
    info: cumulative nitrate leaching (kg/ha)

...
setting:
  all:
    action:
      - anfer
      - amir
    state:
      - yrdoy
...
```

#### Rewards
Reward functions are explicitely defined in a [separated file](https://gitlab.inria.fr/rgautron/gym_dssat_pdi/-/blob/stable/gym-dssat-pdi/gym_dssat_pdi/envs/utils/rewards.py) allowing easy custom reward function definitions. Default reward functions are designed to make challenging problems taking into account the costs of actions and the environmental factors.

## Usage

### Initialization
You can use gym-DSSAT as any gym environment. You need to pass gym-DSSAT configuration as following:

```python
import gym
env_args = {
    'run_dssat_location': '/opt/dssat_pdi/run_dssat',  # assuming (modified) DSSAT has been installed in /opt/dssat_pdi
    'log_saving_path': './logs/dssat-pdi.log',  # if you want to save DSSAT outputs for inspection
    # 'mode': 'irrigation',  # you can choose one of those 3 modes
    # 'mode': 'fertilization',
    'mode': 'all',
    'seed': 123456,
    'random_weather': True,  # if you want stochastic weather
}
env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)
```
That's all !
### Interacting with gym-DSSAT
Actions are provided to the environment in a dictionary:
```python
action_dict = {
    'amir': 10,  # if mode == irrigation or mode == all ; water to irrigate in L/ha
    'anfer': 5,  # if mode == fertilization or mode == all ; nitrogen to fertilize in kg/ha
}
observation, reward, done, info = env.step(action_dict=action_dict)  # info are contextual variables
```
The ```observation``` variable is a dictionnary. The current observation can be retrieved via ```env.observation``` or equivalently ```env.get_state()```:
```python
env.observation = 
{'cleach': 39.01179885864258,
'cnox': 0.07707925885915756,
'cumsumfert': 116.0,
'dap': 126,
'dtt': 20.900001525878906
...}
```
This dictionary can be concatenated to a list using ```env.observation_dict_to_array(env.observation)```. When crop is harvested, ```env.done``` is flagged ```True```.

An example of episode (for ```mode=='fertilization'```) is given by:
```python
while not env.done:
    observation = env.observation
    print(observation)
    # observation_list = env.observation_dict_to_array(observation)
    action = {'anfer': 1} # put 1 kg/ha of nitrogen
    observation, reward, done, info = env.step(action)
```
The ```info``` variable contains contextual informations. At any time, you can access the whole history for the ongoing episode with ```env.history```. Once crop is harvested, you need to reset the environment with ```env.reset()```. Else, you will not get any new state.

Once you're done, **terminate gym-DSSAT**:
```
env.close()
```
The preferred usage is:
```python
try:
  env = gym.make(...)
  ...
  env.step(action_dict=...)
  ...
finally:
  env.close()
```

### Data visualization
gym-DSSAT provides a visualization interface both for raw state variables or rewards.
#### Trajectories
You can call:
```python
env.render(type='ts',  # time series mode
           feature_name_1='nstres',  # mandatory first raw state variable, here the nitrogen stress factor (unitless)
           feature_name_2='grnwt')  # optional second raw state variable, here the grain weight (kg/ha)
```
![plot](./readme_figures/nstresGrnwt.png)

#### Reward vizualization
For quick reward inspection, you can use:

```python
env.render(type='reward')  # plot reward time series (DOY for Day Of Year)
```

![plot](./readme_figures/rewards.png)

Or

```python
env.render(type='reward',
            cumsum=True)  # if you want cumulated rewards
```           
![plot](./readme_figures/cumsumRewards.png)

#### Getting information
You can get information about your current mode using:
```python
env.get_env_info()
```
This outputs:
```shell
******************
Available actions:
******************

{'anfer': {'high': 200,
           'info': 'nitrogen to fertilize for current day (kg/ha)',
           'low': 0,
           'type': 'float'}}
press "return" to continue
{'amir': {'high': 50,
          'info': 'water depth to irrigate for current day (mm/ha)',
          'low': 0,
          'type': 'float'}}
press "return" to continue

*********************
Observation variables:
*********************

{'cleach': {'high': inf,
            'info': 'cumulative nitrate leaching (kg/ha)',
            'low': 0,
            'type': 'float'}}
press "return" to continue
...
```
### Using a custom experimental file
Remember that gym-DSSAT currently only supports DSSAT's CERES-Maize module. If you want to use any maize experiment distinct from the default one, you can use the 3 last arguments as provided below:
```python
env_args = {
    'run_dssat_location': f'{pathlib.Path.home()}/dssat_pdi/run_dssat',
    'log_saving_path': './logs/dssat-pdi.log',
    'mode': 'fertilization',
    'seed': 123456,
    'random_weather': True,
    'fileX_template_path': './my_custom_fileX.MZX',  # where the jinja2 fileX template is located
    'experiment_number': 1,  # the number of the experiment in the fileX to be run
    'auxiliary_file_paths': './my_custom_climate_file.CLI'  # required if random_weather is set True
}
```
**The custom fileX template must present the same expressions ```{{ ... }}``` as the ones found in the [original example](https://gitlab.inria.fr/rgautron/gym_dssat_pdi/-/blob/stable/gym-dssat-pdi/gym_dssat_pdi/envs/configs/UFGA8201.jinja2). Check [jinja documentation](https://jinja.palletsprojects.com/en/3.0.x/templates/). Be careful, any faulty space in the template will raise errors. If you want stochastic climate generation, you must provide to DSSAT the corresponding ".CLI" file with the extra argument ```auxiliary_files_paths```.** Note that ```my_custom_climate_file.CLI``` must be named according to the 4 first characters of the value of ```WSTA``` in the custom fileX. For instance, if in ```./my_custom_fileX.MZX``` the variable ```WSTA``` is set to ```GAGR9626```, then the corresponding climate file must be named ```GAGR.CLI```.

### More information
You can check [more examples](https://gitlab.inria.fr/rgautron/gym_dssat_pdi/-/blob/stable/gym_dssat_pdi_tests/run_env.py), including how to use gym-DSSAT in a multiprocessing context, using the ```env.reset_hard()``` feature.

## About this project
#### Authors
Romain Gautron and Emilio J. Padron

#### Acknowledgements
We acknowledge the DSSAT team, especially Gerrit Hoogenboom and Cheryl Porter. Thanks to the PDI team, especially to Julien Bigot. We acknowledge the Consultative Group for International Agricultural Research (CGIAR), the French Agricultural Research Centre for International Development (CIRAD) and the French Institute for Research in Computer Science and Automation (Inria), in particular the SCOOL team, for their support.
