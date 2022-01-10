
# gym-dssat: an easy to manipulate crop environment for Reinforcement Learning

<img src="./readme_figures/logo_margin.png" align="right" width="20%"/>

## What is gym-dssat?
```gym-dssat``` is a modification of the [Decision Support System for Agrotechnology Transfer (DSSAT)](https://dssat.net/) Fortran software into an easy to manipulate Python [Open AI gym](https://gym.openai.com/) environment for Reinforcement Learning (RL) researchers. ```gym-DSSAT``` allows daily based interactions during the growing season between an RL agent and the crop model with usual gym conventions.

**Important: At the moment gym-DSSAT only supports common Linux distributions**

## Getting started
| Link | Content |
|-|-|
| <a href="https://rgautron.gitlabpages.inria.fr/gym-dssat-docs/Installation/index.html"><img alt="Installation Instructions" src="./readme_figures/installation_button.png" style="height:30px"></a> | Installation with package (recommended), using [Docker](https://www.docker.com/) or from source. |
| <a href="https://rgautron.gitlabpages.inria.fr/gym-dssat-docs/"><img alt="User Documentation" src="./readme_figures/documentation_button.png" style="height:30px"></a> | Get a tour of gym-dssat, including advanced usage. |
| <a href="https://rgautron.gitlabpages.inria.fr/gym-dssat-docs/Tutorials/index.html"><img alt="Tutorials" src="./readme_figures/tutorials_button.png" style="height:30px"></a> | Learn to use by practice in Google Colab (to come soon) |

## How is gym-dssat tested?

Before deploying any new version, **gym-dssat** is built from scratch in a custom Docker image, and the environment tested in a container using [Gitlab CI/CD](https://docs.gitlab.com/ee/ci/), for all supported platforms. If you experience any problem, please create [an issue](https://gitlab.inria.fr/rgautron/gym_dssat_pdi/issues/new?title=Issue%20on%20page%20%2Findex.html&body=Your%20issue%20content%20here).


## Repository content
+ ```./dssat-csm-os```: the submodule of ```dssat-pdi```:  modified the DSSAT Fortran code with PDI for ```gym-DSSAT```
+ ```./gym-dssat-pdi```: the custom ```gym-DSSAT``` Python environment with a use example in ```./gym-dssat-pdi/gym_dssat_pdi_samples```
+ ```./dssat-csm-data```: submodules of the required experimental files used by DSSAT
+ ```./deb_pking```: Linux installation packages
+ ```./docker_recipes```: Dockerfiles for ```gym-dssat```.


## About this project

```gym-DSSAT``` is powered by the [PDI Data Interface (PDI)](https://pdi.julien-bigot.fr/master/) !

#### Contributing
Any [new issue](https://gitlab.inria.fr/rgautron/gym_dssat_pdi/issues/new?title=Issue%20on%20page%20%2Findex.html&body=Your%20issue%20content%20here) is welcomed! If you want to actively contribute, you can check issues the **wishlist** tag.

#### Authors
Romain Gautron and Emilio Padrón González

#### Acknowledgements
We acknowledge the DSSAT team, especially Gerrit Hoogenboom and Cheryl Porter. Thanks to the PDI team, especially to Julien Bigot. We acknowledge Bruno Raffin, leader of Inria’s DataMove team, for his support. We acknowledge the Consultative Group for International Agricultural Research’s (CGIAR) Platform for Big Data in Agriculture, special thanks to Brian King. Thanks to the French Agricultural Research Centre for International Development (CIRAD) and the French Institute for Research in Computer Science and Automation (Inria), in particular the SCOOL team, for their support.
