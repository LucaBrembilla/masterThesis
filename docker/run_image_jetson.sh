#!/bin/bash
set -e

CODE_FOLDER=/home/airlab/brembilla/masterThesis
IMAGE_NAME=openpcdet
TAG=latest
CONTAINER_NAME=airlab_pcdet_$(date +%Y%m%d_%H%M%S)

WORKING_DIR=/home/airlab/exp 

# options
INTERACTIVE=1
LOG_OUTPUT=1


# if [ -z $CPU_SET ]; then
  # source ~/.bashrc
  # CPU_SET=$(get-cpu ${GPU_DEVICE})
# fi

#source ~/.bashrc

# create data directory to store features
#if [ ! -d ${HOME}/data/features/ ]; then
#	mkdir -p ${HOME}/data/features/
##fi


docker_args=(
	"-d --rm -it --runtime=nvidia --gpus all  \
			--mount type=bind,source=${CODE_FOLDER},target=${WORKING_DIR} \
			-w ${WORKING_DIR} \
			-e log=/home/log.txt \
			-e HOST_UID=$(id -u) \
			-e HOST_GID=$(id -g) \
			--name ${CONTAINER_NAME} "
)
  



docker run ${docker_args}  ${IMAGE_NAME}:${TAG}