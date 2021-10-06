# gym-DSSAT: an easy to manipulate crop environment for Reinforcement Learning
gym-DSSAT is a modification of the [Decision Support System for Agrotechnology Transfer (DSSAT)](https://dssat.net/) software into an easy to manipulate [Open AI gym](https://gym.openai.com/) environment for Reinforcement Learning (RL) researchers in Python. gym-DSSAT allows daily based interactions during the growing season between an RL agent and the crop model with usual gym conventions.

gym-DSSAT is powered by the [PDI Data Interface (PDI)](https://pdi.julien-bigot.fr/master/) !

**Disclaimer: gym-DSSAT only supports Unix systems!**

## In this repository
+ ```./dssat-csm-os```: the submodule of modified the DSSAT Fortran code with PDI for gym-DSSAT
+ ```./gym-dssat-pdi```: the custom gym-DSSAT Python environment
+ ```./dssat-csm-data```: the required experimental files used by DSSAT
+ ```./gym_dssat_pdi_tests```: examples of how to run gym-DSSAT
## The environment
gym-DSSAT is designed to allow great setting flexibility. The environment comes with default settings that can easily been modified by editing the [gym environment's yaml configuration file](https://gitlab.inria.fr/rgautron/gym_dssat_pdi/-/blob/stable/gym-dssat-pdi/gym_dssat_pdi/envs/configs/env_config.yml). The ```action``` key gives the raw action space, the ```state``` key give the raw action space and each individual setting is found and can be edited in the ```setting``` key. Furthermore, the ```context``` key defines additional contextual variables.

gym-DSSAT uses by default the UFGA8201 maize experiment from the University of Florida, but is usable with any DSSAT experiment using the CERES-Maize module.

#### Action space
The environment comes with 3 modes:
+ nitrogen fertilization only (continuous quantity): ```mode=='fertilization'```
+ irrigation only (continuous quantity): ```mode=='irrigation'```
+ both nitrogen fertilization and irrigation (both continuous quantities): ```mode=='all'```

Actions are provided to the environment in a dictionary:
```python
action_dict = {
    'amir': 10,  # if mode == irrigation or mode == all ; water to irrigate in L/ha
    'anfer': 5,  # if mode == fertilization or mode == all ; nitrogen to fertilize in kg/ha
}
observation, reward, done, info = env.step(action_dict=action_dict)  # info are contextual variables
```
#### State space
The action space depends on each mode and are detailed in gym environment's yaml configuration file. State variables can be continuous, discrete and arrays of arbitrary shapes. By default, the observed state is given as a dictionnary as show below:

```python
env.observation = 
{'cleach': 39.01179885864258,
'cnox': 0.07707925885915756,
'cumsumfert': 116.0,
'dap': 126,
'dtt': 20.900001525878906
...}
```
This dictionary can be concatenated to a list using ```env.observation_dict_to_array(env.observation)```

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
#### Rewards
Reward functions are explicitely defined in a [separated file](https://gitlab.inria.fr/rgautron/gym_dssat_pdi/-/blob/stable/gym-dssat-pdi/gym_dssat_pdi/envs/utils/rewards.py) allowing easy custom reward function definitions. Default reward functions are designed to make challenging problems taking into account the costs of actions and the environmental factors.

### Data visualization
gym-DSSAT provides a visualization interface both for raw state variables or rewards.
#### Trajectories
You can call:
```python
env.render(type='ts',  # time series mode
           feature_name_1='nstres',  # mandatory first raw state variable
           feature_name_2='grnwt')  # optional second raw state variable
```
![plot](./readme_figures/nstresGrnwt.png)

#### Reward vizualization
For quick reward inspection, you can use:
```python
env.render(type='reward')  # plot reward time series
```
![plot](./readme_figures/rewards.png)
Or
```python
env.render(type='reward',
            cumsum=True)  # if you want cumulated rewards
```           
![plot](./readme_figures/cumsumRewards.png)

## Installing gym-DSSAT
Here you will find how to install in the order the various components of gym-DSSAT