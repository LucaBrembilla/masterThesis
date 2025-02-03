#!/bin/bash
CONFIG_FILE=/home/brembilla/exp/tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml
BATCH_SIZE=1
CKPT=/home/brembilla/exp/pretrained/cbgs_second_multihead_nds6229_updated.pth
python test_less_computation.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
