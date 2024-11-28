#!bin/bash
set -e
cd ..

CODE_FOLDER=/home/$(whoami)/project
IMAGE_NAME=djiajun1206/pcdet
TAG=pytorch1.6
CONTAINER_NAME=$(whoami)_headless_open3d_$(date +%Y%m%d_%H%M%S)

WORKING_DIR=/home/$(whoami)/exp 
MEMORY_LIMIT=32g

# options
INTERACTIVE=1
LOG_OUTPUT=1

docker_args=(
			"-d --rm -it  \
			--mount type=bind,source=${CODE_FOLDER},target=${WORKING_DIR} \
			-m ${MEMORY_LIMIT} \
			-w ${WORKING_DIR} \
			-e log=/home/log.txt \
			-e HOST_UID=$(id -u) \
			-e HOST_GID=$(id -g) \
			--name ${CONTAINER_NAME} "
)

docker run ${docker_args}  ${IMAGE_NAME}:${TAG}

# docker run -d -it --env OPEN3D_CPU_RENDERING=true -v "$PWD" --name brembilla_open3d_headless djiajun1206/pcdet:pytorch1.6

#docker run -d -it --env OPEN3D_CPU_RENDERING=true \
# --name brembilla_open3d_headless djiajun1206/pcdet:pytorch1.6 \
 