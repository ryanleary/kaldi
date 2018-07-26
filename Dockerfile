# NGC
# FROM nvcr.io/nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 as build
# Dockerhub
FROM nvidia/cuda:9.2-devel-ubuntu16.04

ENV CUDA_HOME=/usr/local/cuda
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        flac \
        sox \
	      numactl \
        subversion \
        libnss-wrapper \
        cmake \
        wget \
        unzip \
        libatlas-dev \ 
        autoconf \
        libtool \
        automake \
        git \
        vim \
        ssh \
        zlib1g-dev \
        libatlas3-base \
        python3 \
        python2.7 \
        sudo \
        fuse \
    && cd /tmp \
    && wget https://github.com/git-lfs/git-lfs/releases/download/v2.4.2/git-lfs-linux-amd64-2.4.2.tar.gz \
    && tar -xzf git-lfs-linux-amd64-2.4.2.tar.gz \
    && cd git-lfs-2.4.2 && ./install.sh && cd ../ \
    && rm -rf /var/lib/apt/lists/* \
    && echo "ALL   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir /work && chmod 777 /work
COPY entrypoint.sh /usr/bin
RUN chmod 755 /usr/bin/entrypoint.sh
WORKDIR /work
ENV HOME=/work
ENTRYPOINT ["/usr/bin/entrypoint.sh"]

