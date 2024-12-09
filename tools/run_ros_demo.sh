#!bin/bash

# pip install bagpy

CGF_FILE=cfgs/kitti_models/pointpillar.yaml
CKPT=/home/brembilla/exp/pretrained/pointpillar_7728.pth
POINT_CLOUD_DATA=/home/brembilla/exp/private_datasets/similar_dataset/velodyne/frame_0000.npy

python ros_demo.py --cfg_file ${CGF_FILE} \
    --ckpt ${CKPT} \
    --data_path ${POINT_CLOUD_DATA}