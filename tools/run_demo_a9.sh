#!bin/bash

CGF_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/pointrcnn.yaml

CKPT=/home/brembilla/exp/pretrained/pointrcnn_7870.pth

POINT_CLOUD_DATA=/home/brembilla/exp/private_datasets/providentia/_points/1607434792_560000000_s50_lidar_ouster_south_valid.npy

# POINT_CLOUD_DATA=/home/brembilla/exp/shared_datasets/PNRR/ouster/frame_2121.npy
# cd ..

# pip install spconv-cu102

# python setup.py develop

# cd tools

# pip3 install --ignore-installed PyYAML

# pip install open3d

# python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

python demo_a9.py --cfg_file ${CGF_FILE} \
    --ckpt ${CKPT} \
    --data_path ${POINT_CLOUD_DATA} \
    --ext .npy