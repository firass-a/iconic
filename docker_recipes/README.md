# gym-DSSAT-PDI coupling docker image
This repository contain Dockerfiles to build images using pre-compiled gym-DSSAT packages under:
- [Debian11 (Bullseye)](https://www.debian.org/releases/bullseye): ```Dockerfile_Debian_Bullseye```
- [Debian Unstable (Sid)](https://www.debian.org/releases/sid): ```Dockerfile_Debian_Sid```
- [Ubuntu 22.04 (Jammy Jellyfish)](http://releases.ubuntu.com/jammy): ```Dockerfile_Ubuntu_Jammy```
- [Ubuntu 21.10 (Impish Indri)](http://releases.ubuntu.com/impish): ```Dockerfile_Ubuntu_Impish```
- [Ubuntu 20.04 LTS (Focal Fossa)](http://releases.ubuntu.com/focal): ```Dockerfile_Ubuntu_Focal```
- [Ubuntu 18.04 LTS (Bionic Beaver)](http://releases.ubuntu.com/bionic): ```Dockerfile_Ubuntu_Bionic``` (Python3.7-based version of gym-DSSAT-PDI, useful for Google Colab)

## Building an image
We take the example of the ```Dockerfile_Debian_Bullseye``` Dockerfile. To build a different Dockerfile, you just need to substitute the Dockerfile name as [referred above](#gym-dssat-pdi-coupling-docker-image).

<!-- ```bash
git clone https://gitlab.inria.fr/rgautron/gym_dssat_pdi.git
cd gym_dssat_pdi/docker_recipes
``` -->
To build an image called ```gym-dssat:debian-bullseye``` form the Dockerfile named ```Dockerfile_Debian_Bullseye```, simply run:
```bash
docker build https://gitlab.inria.fr/rgautron/gym_dssat_pdi.git\#dev:docker_recipes -t "gym-dssat:debian-bullseye" -f Dockerfile_Debian_Bullseye 
```
## Run the container
To check the ```gym-dssat:debian-bullseye``` image previously built, you can run the default example just with:

```bash
docker run gym-dssat:debian-bullseye
```

Or you can interactively run the Docker image with:

```bash
docker run -it gym-dssat:debian-bullseye bash
```

# Docker image with Spack installation
Additionaly, these Dockerfiles to build docker images using Spack are
provided. This recipes do not use pre-compiled packages, so gym-DSSAT
and all its dependencies are built from sources (see gym-DSSAT Spack
packages in https://gitlab.inria.fr/rgautron/gym_dssat_pdi-spack):
- [Debian11 (Bullseye)](https://www.debian.org/releases/bullseye): ```Dockerfile_Debian_Bullseye_Spack```
- [Ubuntu 22.04 LTS (Jammy Jellyfish)](http://releases.ubuntu.com/jammy): ```Dockerfile_Ubuntu_Jammy_Spack```
- [Ubuntu 20.04 LTS (Focal Fossa)](http://releases.ubuntu.com/focal): ```Dockerfile_Ubuntu_Focal_Spack```
- [CUDA on Ubuntu 20.04 LTS (Focal Fossa)](http://releases.ubuntu.com/focal): ```Dockerfile_CUDA_Spack```
- [CentOS (latest)](https://www.centos.org): ```Dockerfile_CentOS_Spack```
