#####################################
############## BUILDER ##############
#####################################
FROM debian:sid as builder

USER 0

RUN apt update -y \
 && apt upgrade -y \
 && apt autoremove -y \
 && apt clean \
 && apt autoclean

RUN apt install -y wget python3-pip python3-dev

### PDI package install
RUN echo "deb [ arch=amd64 ] https://raw.githubusercontent.com/pdidev/repo/debian sid main" | tee /etc/apt/sources.list.d/pdi.list > /dev/null
RUN wget -O /etc/apt/trusted.gpg.d/pdidev-archive-keyring.gpg https://raw.githubusercontent.com/pdidev/repo/debian/pdidev-archive-keyring.gpg
RUN chmod a+r /etc/apt/trusted.gpg.d/pdidev-archive-keyring.gpg /etc/apt/sources.list.d/pdi.list

# Temporary workaround to guarantee the updated tarball is downloaded
# and we are using up-to-date debian packages
#
# Calls for a random number to break the cahing of the following wget
# (https://stackoverflow.com/questions/35134713/disable-cache-for-specific-run-commands/58801213#58801213)
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y pdidev-archive-keyring libpdi-dev pdiplugin-pycall cmake gfortran python3-numpy git

# DSSAT INSTALLATION
RUN git clone --recurse-submodules https://gitlab.inria.fr/rgautron/gym_dssat_pdi.git
RUN mkdir /gym_dssat_pdi/dssat-csm-os/build
WORKDIR /gym_dssat_pdi/dssat-csm-os/build
RUN cmake -DCMAKE_INSTALL_PREFIX='/opt/dssat_pdi' -DCMAKE_PREFIX_PATH='/usr/share/paraconf/cmake;/usr/share/pdi/cmake' ../
RUN make
RUN make install


# # COPY DSSAT FILES
WORKDIR /gym_dssat_pdi
RUN mv dssat-csm-data/* /opt/dssat_pdi/
RUN pip3 install -e ./gym-dssat-pdi

WORKDIR /

ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["cd /gym_dssat_pdi/gym-dssat-pdi/gym_dssat_pdi_samples && python3 run_env.py"]
