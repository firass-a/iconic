# gym-DSSAT-PDI coupling docker image
This repository contain Dockerfiles to build images under:
- Ubuntu () :
- [Debian11 (Bullseye)](https://www.debian.org/releases/bullseye/): ```Dockerfile_Debian_Bullseye```
- [Debian Unstable (Sid)](https://www.debian.org/releases/sid/): ```Dockerfile_Debian_Sid```

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
