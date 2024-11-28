apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

apt-get update \
  && apt-get install --yes \
  libgl1 libgomp1 python3-pip \
  libdrm2 libedit2 libexpat1 libgcc-s1 libglapi-mesa libllvm10 libx11-xcb1 \
  libxcb-dri2-0 libxcb-glx0 libxcb-shm0 libxcb-xfixes0 libxfixes3 \
  libxxf86vm1 \
  && rm -rf /var/lib/apt/lists/*

apt-get install --yes libegl1