FROM ubuntu:22.04

RUN apt update
RUN apt install -y git build-essential m4 zlib1g-dev libx11-dev gfortran pkg-config autoconf python3-dev vim tmux nano gdb valgrind libssl-dev
RUN apt autoremove && apt clean

RUN git clone https://github.com/wortiz/goma -b bst /opt/goma

RUN PYTHONUNBUFFERED=1 /opt/goma/tpls/install-tpls.py --cc=gcc --cxx=g++ --fc=gfortran --build-shared --download-dir=/tmp/downloads -j 32 --extract-dir=/tmp/extract /opt/goma-libs

RUN rm -r /opt/goma
RUN rm -r /tmp/downloads
RUN rm -r /tmp/extract
RUN rm -r /opt/goma-libs/sources
RUN rm -r /opt/goma-libs/logs

RUN { \
      echo '#!/usr/bin/env bash' \
      && echo ". /opt/goma-libs/config.sh" \
      && echo 'exec "$@"'; \
    } > /entrypoint.sh \
&& chmod a+x /entrypoint.sh


ARG USER=goma
RUN adduser -m $USER
RUN groupadd sudo
RUN usermod -a -G sudo $USER
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# make sure everything is in place
USER $USER
ENV HOME /home/$USER
ENV USER $USER
ENV OMPI_MCA_btl_base_warn_component_unuse "0"
ENV OMPI_ALLOW_RUN_AS_ROOT 1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1
WORKDIR $HOME

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/bin/bash" ]

