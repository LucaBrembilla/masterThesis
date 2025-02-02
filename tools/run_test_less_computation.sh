#!bin/bash
CGF_FILE=cfgs/kitti_models/pv_rcnn.yaml
CKPT=/home/brembilla/exp/pretrained/pv_rcnn_8369.pth
BATCH_SIZE=1

python test_less_computation.py --cfg_file ${CGF_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
