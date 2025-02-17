#!bin/bash

CONFIG_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/pv_rcnn_a9.yaml
CONFIG_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/pointrcnn_a9.yaml
CONFIG_FILE=/home/brembilla/exp/tools/cfgs/kitti_models/second_iou_a9.yaml

CKPT=/home/brembilla/exp/pretrained/second_7862.pth
CKPT=/home/brembilla/exp/pretrained/pv_rcnn_8369.pth
CKPT=/home/brembilla/exp/pretrained/pointrcnn_7870.pth
CKPT=/home/brembilla/exp/pretrained/second_iou7909.pth

POINT_CLOUD_DATA=/home/brembilla/exp/private_datasets/providentia/_numpy

python demo_a9_less_computation.py --cfg_file ${CONFIG_FILE} \
    --ckpt ${CKPT} \
    --data_path ${POINT_CLOUD_DATA} \
    --ext .npy