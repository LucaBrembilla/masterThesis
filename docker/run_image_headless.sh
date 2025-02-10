#!bin/bash
set -e
cd ..

CODE_FOLDER=/home/$(whoami)/project

GPU_DEVICE=3
CPU_SET=8-11

IMAGE_NAME=djiajun1206/pcdet
TAG=pytorch1.6

CONTAINER_NAME=$(whoami)_pcdet_$(date +%Y%m%d_%H%M%S)

######################################################################
######################################################################

STORAGE_FOLDER=/multiverse/storage/$(whoami)/
SHARED_DATASET_FOLDER=/multiverse/datasets/shared/
PRIVATE_DATASET_FOLDER=/multiverse/datasets/$(whoami)/

WORKING_DIR=/home/$(whoami)/exp 
MEMORY_LIMIT=32g

# options
INTERACTIVE=1
LOG_OUTPUT=1

while [[ $# -gt 0 ]]
do 
	key="$1"

	case $key in
		-im|--image_name)
		IMAGE_NAME="$2"
		shift # past argument
		shift # past value
		;;
		-t|--tag)
		TAG="$2"
		shift # past argument
		shift # past value
		;;
		-i|--interactive)
		INTERACTIVE="$2"
		shift # past argument
		shift # past value
		;;
		-gd|--gpu_device)
		GPU_DEVICE="$2"
		shift # past argument
		shift # past value
		;;
		-m|--memory_limit)
		MEMORY_LIMIT="$2"
		shift # past argument
		shift # past value
	

	MEMORY_LIMIT=32g
		shift # past argument
		shift # past value
		;;
		-cpu|--cpu_set)
		CPU_SET="$2"
		shift # past argument
		shift # past value
		;;
		-h|--help)
		shift # past argument
		echo "Options:"
		echo "	-im, --image_name 	name of the docker image (default \"base_images/tensorflow\")"
		echo "	-t, --tag 		image tag name (default \"tf2-gpu\")"
		echo "	-gd, --gpu_device 	gpu to be used inside docker (default 1)"
		echo "	-cn, --container_name	name of container (default \"tf2_run\" )"
		echo "	-m, --memory_limit 	RAM limit (default 32g)"
		echo "	-cpu, --cpu_set 	cpu ids to be used inside docker"
		exit
		;;
		*)
		echo " Wrong option(s) is selected. Use -h, --help for more information "
		exit
		;;
	esac
done

if [ -z $CPU_SET ]; then
  source ~/.bashrc
  CPU_SET=$(get-cpu ${GPU_DEVICE})
fi

#source ~/.bashrc

echo "WORKING_DIR	= ${WORKING_DIR}"
echo "GPU_DEVICE 	= ${GPU_DEVICE}"
echo "CPU_SET		= ${CPU_SET}"
echo "CONTAINER_NAME 	= ${CONTAINER_NAME}"
echo "PORT		= ${PORT}"

echo "Running docker in interactive mode"

# create data directory to store features
#if [ ! -d ${HOME}/data/features/ ]; then
#	mkdir -p ${HOME}/data/features/
##fi


docker_args=(
			"-d --rm -it --gpus "device=${GPU_DEVICE}"  \
			--cpuset-cpus ${CPU_SET} \
			--mount type=bind,source=${CODE_FOLDER},target=${WORKING_DIR} \
			--mount type=bind,source=${SHARED_DATASET_FOLDER},target=${WORKING_DIR}/shared_datasets \
			--mount type=bind,source=${PRIVATE_DATASET_FOLDER},target=${WORKING_DIR}/private_datasets \
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
 