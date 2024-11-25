# Base image with CUDA 11.1, suitable for PyTorch 1.8.1 (required by pointpillars repo)
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# my user
ARG USERNAME=brembilla 
ARG USER_UID=1148
ARG USER_GID=$USER_UID

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Set the working directory inside the container
WORKDIR /exp

ENV DEBIAN_FRONTEND noninteractive
# add Nvidea signature keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# add repo for opython 3.X
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa 

RUN apt-get update && apt-get install -y \
	build-essential ca-certificates python3 python3-dev python3-distutils git vim wget cmake python3-pip  glibc-source 

RUN apt-get install -y libgl1


# Set my user (non-root)
RUN groupadd -g ${USER_GID} ${USERNAME} && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME}
USER ${USERNAME}

# Copy the code into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r ./requirements.txt

# Expose GPU libraries
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Expose the shell
ENV SHELL /bin/bash

# Run the shell
CMD ["bash"]

# Default command to run your test script
# CMD ["bash", "run_a_test.sh"]
