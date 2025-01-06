#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
import math

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import argparse
from pathlib import Path
import numpy as np
import torch

import glob

# Paths to your OpenPCDet configuration and model checkpoint
# CFG_FILE = '/home/brembilla/exp/tools/cfgs/kitti_models/pv_rcnn_ros.yaml'
CKPT = '/home/brembilla/exp/pretrained/pv_rcnn_8369.pth'

# Global model instance
model = None
demo_dataset = None

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
    def process_point_cloud(self, points):
        input_dict = {
            'points': points,
            'frame_id': 0,
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

def load_model():
    """Load the model and keep it on the GPU."""
    global model

    logger = common_utils.create_logger()

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=CKPT, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

def pc_preprocess(pointcloud):
    
    if pointcloud.shape[1] < 4:
        raise ValueError("Pointcloud data must have at least 4 dimensions (x, y, z, intensity).")

    # Set intensity to 0
    pointcloud[:, 3] = 0

    sub = 4.4
    pointcloud[:, 2] -= sub 
    pointcloud[:, 2] = -pointcloud[:, 2]

    return pointcloud

def parse_and_publish(predictions):
    """Publish bounding box predictions as ROS markers."""
    marker_array = MarkerArray()

    for i, box in enumerate(predictions['pred_boxes']):
        x_center, y_center, z_center, x_size, y_size, z_size, yaw = box
        pred_label = predictions['pred_labels'][i]
        pred_score = predictions['pred_scores'][i]

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_boxes"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Set pose and dimensions
        marker.pose.position.x = x_center
        marker.pose.position.y = y_center
        marker.pose.position.z = z_center
        marker.pose.orientation.z = math.sin(yaw / 2)
        marker.pose.orientation.w = math.cos(yaw / 2)
        marker.scale.x = x_size
        marker.scale.y = y_size
        marker.scale.z = z_size

        # Set color based on type
        if pred_label == 1:  # Car
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = pred_score  # Transparency
        elif pred_label == 2:  # Pedestrian
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = pred_score
        elif pred_label == 3:  # Cyclist
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = pred_score
        else:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = pred_score

        marker_array.markers.append(marker)

    bbox_publisher.publish(marker_array)
    rospy.loginfo(f"Published {len(marker_array.markers)} bounding boxes.")


def pointcloud_callback(msg):
    """Callback to process incoming point clouds."""
    try:
        # Convert PointCloud2 to numpy array
        pointcloud = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=["x", "y", "z", "intensity"])))
        rospy.loginfo(f"Received point cloud with {pointcloud.shape[0]} points.")

        # Preprocess point cloud
        points = pc_preprocess(pointcloud)

        # Perform inference
        global model
        global demo_dataset
        with torch.no_grad():
            data_dict = demo_dataset.process_point_cloud(points)
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

        # Prepare predictions for publishing
        pred_dicts_serializable = {
            "pred_boxes": pred_dicts[0]['pred_boxes'].tolist(),
            "pred_scores": pred_dicts[0]['pred_scores'].tolist(),
            "pred_labels": pred_dicts[0]['pred_labels'].tolist()
        }
        parse_and_publish(pred_dicts_serializable)

    except Exception as e:
        rospy.logerr(f"Error processing point cloud: {e}")


def main():
    global bbox_publisher
    global demo_dataset 
    rospy.init_node('pointcloud_subscriber', anonymous=True)
    bbox_publisher = rospy.Publisher('/ouster/bounding_boxes', MarkerArray, queue_size=10)
    rospy.Subscriber('/ouster/points', PointCloud2, pointcloud_callback)

    args, cfg = parse_config()

    logger = common_utils.create_logger()
    
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger,
    )

    # Load the model once
    load_model()

    rospy.loginfo("PointCloud2 Subscriber Node Initialized.")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
