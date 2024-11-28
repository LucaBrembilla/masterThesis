#!bin/bash

CGF_FILE=cfgs/kitti_models/pv_rcnn.yaml
CKPT=/home/brembilla/exp/pretrained/pv_rcnn_8369.pth
#POINT_CLOUD_DATA=/home/brembilla/exp/shared_datasets/kitti/data/sync/2011_10_03/2011_10_03_drive_0027_sync/velodyne_points/data/0000000541.bin
POINT_CLOUD_DATA=/home/brembilla/exp/models/pointpillars/dataset/demo_data/val/000134.bin
# cd ..

# pip install spconv-cu102

# python setup.py develop

# cd tools

# pip3 install --ignore-installed PyYAML

# pip install open3d

# python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

python demo.py --cfg_file ${CGF_FILE} \
    --ckpt ${CKPT} \
    --data_path ${POINT_CLOUD_DATA}