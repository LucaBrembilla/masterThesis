import argparse
from pathlib import Path
import numpy as np
import torch
from visualization_msgs.msg import Marker, MarkerArray
import rosbag

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import json
import yaml

def pc_preprocess(pointcloud):
    
    if pointcloud.shape[1] < 4:
        raise ValueError("Pointcloud data must have at least 4 dimensions (x, y, z, intensity).")

    # Set intensity to 0
    pointcloud[:, 3] = 0

    sub = 4.4
    pointcloud[:, 2] -= sub 
    pointcloud[:, 2] = -pointcloud[:, 2]

    return pointcloud

class SinglePointCloudDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, logger=None, root_path=None, pointcloud=None):
        """
        Args:
            dataset_cfg: Configuration for the dataset.
            class_names: List of class names.
            training: Boolean indicating if the dataset is for training.
            logger: Logger instance.
            pointcloud: A single point cloud as a numpy array.
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=None, logger=logger
        )
        self.pointcloud = pointcloud

    def __len__(self):
        return 1  # Single sample

    def __getitem__(self, index):
        assert self.pointcloud is not None, "Point cloud data is not provided."

        points = pc_preprocess(self.pointcloud)

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/brembilla/exp/tools/cfgs/kitti_models/pv_rcnn_ros.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/brembilla/exp/shared_datasets/PNRR/ouster/example',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='/home/brembilla/exp/pretrained/pv_rcnn_8369.pth')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def process_point_cloud(CFG_FILE, CKPT, pointcloud):
    args, cfg = parse_config()
    
    logger = common_utils.create_logger()
    
    demo_dataset = SinglePointCloudDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger,
        pointcloud=pointcloud
    )
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=CKPT, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for data_dict in demo_dataset:
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_dicts_serializable = {
                "pred_boxes": pred_dicts[0]['pred_boxes'].tolist(),
                "pred_scores": pred_dicts[0]['pred_scores'].tolist(),
                "pred_labels": pred_dicts[0]['pred_labels'].tolist()
            }
            
            return pred_dicts_serializable

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    single_pointcloud = np.random.rand(10000, 4).astype(np.float32)  # Replace with actual point cloud

    demo_dataset = SinglePointCloudDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger, pointcloud=single_pointcloud
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_dicts_serializable = {
                "pred_boxes": pred_dicts[0]['pred_boxes'].tolist(),
                "pred_scores": pred_dicts[0]['pred_scores'].tolist(),
                "pred_labels": pred_dicts[0]['pred_labels'].tolist()
            }

            # Print as JSON
            print(json.dumps(pred_dicts_serializable))

            # Format of the results:
            # pred_label x_center y_center z_center x_size y_size z_size yaw pred_score pred_cls_score pred_iou_scores
            # Labels: 1 - Car/Vehicle, 2 - Pedestrian, 3 - Cyclist
            

if __name__ == "__main__":
    main()