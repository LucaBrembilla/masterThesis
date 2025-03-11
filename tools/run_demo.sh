#!bin/bash

CGF_FILE=cfgs/kitti_models/pv_rcnn.yaml
CKPT=/home/brembilla/exp/pretrained/pv_rcnn_8369.pth
POINT_CLOUD_DATA=/home/brembilla/exp/models/pointpillars/dataset/demo_data/val/000134.bin

python demo.py --cfg_file ${CGF_FILE} \
    --ckpt ${CKPT} \
    --data_path ${POINT_CLOUD_DATA}