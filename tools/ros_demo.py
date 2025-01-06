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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import json

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
    def __init__(self, dataset_cfg, class_names, training=True, logger=None, pointcloud=None):
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

        points = pc_preprocess(points)

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def process_predictions(pred_dict):
    pred_boxes = pred_dict['pred_boxes'].cpu().numpy()  # Convert to CPU and numpy array
    pred_scores = pred_dict['pred_scores'].cpu().numpy()
    pred_labels = pred_dict['pred_labels'].cpu().numpy()
    
    return pred_boxes, pred_scores, pred_labels

def save_predictions_as_ros_markers(pred_boxes, idx):
    """
    Save predictions as ROS markers into a bag file.
    """
    marker_array = MarkerArray()

    # Iterate over the boxes pred_boxes
    for i, box in enumerate(pred_boxes):
        marker = Marker()
        marker.header.frame_id = "map"  # Frame ID for visualization in RViz
        # marker.header.stamp = rospy.Time.now()
        marker.ns = "predictions"
        marker.id = idx * 100 + i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Set box center and dimensions
        marker.pose.position.x = float(box[0])
        marker.pose.position.y = float(box[1])
        marker.pose.position.z = float(box[2])
        marker.scale.x = float(box[3])
        marker.scale.y = float(box[4])
        marker.scale.z = float(box[5])

        # Set box orientation (assuming no rotation)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set color (e.g., based on label or a default color)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        marker_array.markers.append(marker)

    # Return the MarkerArray
    return marker_array

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def process_point_cloud(CFG_FILE, CKPT, pointcloud, model):
    cfg = {}
    cfg_from_yaml_file(CFG_FILE, cfg)
    
    logger = common_utils.create_logger()
    
    demo_dataset = SinglePointCloudDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger, pointcloud=single_pointcloud
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