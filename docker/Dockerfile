FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

# Install basics
RUN apt-get update -y \
    && apt-get install build-essential \
    && apt-get install -y apt-utils git curl ca-certificates bzip2 tree htop wget \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ python3.10 python3.10-dev python3.10-distutils

# Install cmake v3.13.2


# Install python
RUN ln -sv /usr/bin/python3.10 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL numpy llvmlite numba 

# Install torch and torchvision
# See https://pytorch.org/ for other options if you use a different version of CUDA
# RUN pip install --user torch==1.6 torchvision==0.7.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL tensorboardX easydict pyyaml scikit-image tqdm SharedArray six

WORKDIR /root

# Install Boost geometry


# Install open3D

# A weired problem that hasn't been solved yet


# git clone https://github.com/traveller59/spconv.git --recursive
# apt-get install -y libboost-all-dev
# apt-get install -y cmake
# cd spconv
# pip install pccm
# python setup.py bdist_wheel
# pip install spconv-2.3.6-py3-none-any.whl