# gym-DSSAT: an easy to manipulate crop environment for Reinforcement Learning
gym-DSSAT is a modification of the [Decision Support System for Agrotechnology Transfer (DSSAT)](https://dssat.net/) software into an easy to manipulate [Open AI gym](https://gym.openai.com/) environment for Reinforcement Learning (RL) researchers in Python.

The modified DSSAT allows daily based interactions during the growing season between an RL agent and the crop model with usual gym conventions.

##In this repository
+ ```./dssat-csm-os-pdi```: the modified DSSAT Fortran code
+ ```./gym_dssat_pdi_project```: the custom gym environment
##The environment
### Action space
The environment comes with 3 modes:
+ nitrogen fertilization only (continuous quantity)
+ irrigation only (continuous quantity)
+ both nitrogen fertilization and irrigation (both continuous quantities)

gym-DSSAT uses by default the UFGA8201 maize experiment from the University of Florida, but is usable with any DSSAT experiment using the CERES-Maize module.

### State space
gym-DSSAT allows to potentially access numerous DSSAT's internal variables. The environment comes with default selected state variables for each mode, but this is fully and easily customizable.

### Rewards
Reward functions are explicitely defined in a separated file allowing easy custom reward function definitions.
Default reward functions are designed to make challenging problems taking into account the costs of actions and the possible induced pollution.

### Data visualization
gym-DSSAT provides a simple visualization interface.

## Under the hood
gym-DSSAT is powered by the [PDI Data Interface (PDI)](https://pdi.julien-bigot.fr/master/) !

## Installing gym-DSSAT
In process, stable version with full documentation will come soon!