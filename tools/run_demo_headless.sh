#!bin/bash

CGF_FILE=cfgs/kitti_models/pv_rcnn.yaml
CGF_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/pv_rcnn_plusplus_reproduced_by_community.yaml
CGF_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/pointrcnn.yaml
CGF_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/pointpillar.yaml
CGF_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/second.yaml
CGF_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/voxel_rcnn_car.yaml
CGF_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/second_iou.yaml
CGF_FILE=cfgs/kitti_models/pv_rcnn.yaml
CFG_FILE=/home/airlab/brembilla/masterThesis/tools/cfgs/kitti_models/second_iou_ros2.yaml
CFG_FILE=/home/airlab/brembilla/masterThesis/tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint_ros2.yaml



CKPT=/home/brembilla/exp/pretrained/pv_rcnn_8369.pth
CKPT=/home/brembilla/exp/pretrained/pointrcnn_7870.pth
CKPT=/home/brembilla/exp/pretrained/pointpillar_7728.pth
CKPT=/home/brembilla/exp/pretrained/second_7862.pth
CKPT=/home/brembilla/exp/pretrained/voxel_rcnn_car_84.54.pth
CKPT=/home/brembilla/exp/pretrained/second_iou7909.pth
CKPT=/home/brembilla/exp/pretrained/pv_rcnn_8369.pth
CKPT=/home/airlab/brembilla/masterThesis/pretrained/pv_rcnn_8369.pth
CKPT=/home/airlab/brembilla/masterThesis/pretrained/second_iou7909.pth
CKPT=/home/airlab/brembilla/masterThesis/pretrained/cbgs_pp_centerpoint_nds6070.pth


POINT_CLOUD_DATA=/home/brembilla/exp/shared_datasets/kitti/kitti/testing/velodyne/000001.bin
POINT_CLOUD_DATA=/home/brembilla/exp/shared_datasets/PNRR/ouster/frame_0250.npy
POINT_CLOUD_DATA=/home/brembilla/exp/data/zeroed_frame_0250.npy
POINT_CLOUD_DATA=/home/brembilla/exp/data/zeroed_frame_0000.npy
POINT_CLOUD_DATA=/home/brembilla/exp/shared_datasets/PNRR/ouster/frame_0000.npy
POINT_CLOUD_DATA=/home/brembilla/exp/shared_datasets/PNRR/ouster/example
POINT_CLOUD_DATA=/media/airlab/000F9736000EFC5E/PNRR/ouster

# POINT_CLOUD_DATA=/home/brembilla/exp/shared_datasets/PNRR/ouster/frame_2121.npy
# cd ..

# pip install spconv-cu102

# python setup.py develop

# cd tools

# pip3 install --ignore-installed PyYAML

# pip install open3d

# python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

python demo_headless.py --cfg_file ${CGF_FILE} \
    --ckpt ${CKPT} \
    --data_path ${POINT_CLOUD_DATA} \
    --ext .npy