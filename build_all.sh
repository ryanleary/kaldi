#!/bin/bash

source environment.docker
cd tools
make -j8
cd ../src
./configure --use-cuda --cudatk-dir=/usr/local/cuda/
make -j8

