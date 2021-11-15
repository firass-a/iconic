# gym-DSSAT-PDI coupling debug docker image

To build & use of of these images:

OPTIONAL: Clone the repository and enter this subdirectory

```bash
git clone https://gitlab.inria.fr/rgautron/gym_dssat_pdi.git
cd gym_dssat_pdi/docker_recipes
```

To build a Debian Bullseye (Debian 11 stable) docker image (choose
the corresponding Dockerfile for the recipe you are interested in) just
type:

```bash
docker build . -t "gym_dssat_pdi_debian_11" -f Dockerfile_Debian_Bullseye
```

To check the image you can run the default example just with:

```bash
docker run -it gym_dssat_pdi_debian_11
```

To interactively run the Docker image:

```bash
docker run -it gym_dssat_pdi_debian_11 bash
```
