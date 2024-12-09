#!bin/bash
CGF_FILE=cfgs/kitti_models/pointpillar.yaml
CKPT=/home/brembilla/exp/pretrained/pointpillar_7728.pth
BATCH_SIZE=2

python test.py --cfg_file ${CGF_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
