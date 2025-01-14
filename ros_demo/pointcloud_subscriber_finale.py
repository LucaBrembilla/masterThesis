#!/usr/bin/env python
import rospy
import ros_numpy
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
CKPT = '/home/brembilla/exp/pretrained/pointpillar_7728.pth'

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
    parser.add_argument('--cfg_file', type=str, default='/home/brembilla/exp/tools/cfgs/kitti_models/pointpillar_ros.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/brembilla/exp/shared_datasets/PNRR/ouster/example',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='/home/brembilla/exp/pretrained/pointpillar_7728.pth')

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

def parse_and_publish(predictions):
    """Publish bounding box predictions as ROS markers."""
    marker_array = MarkerArray()

    for i, box in enumerate(predictions['pred_boxes']):
        x_center, y_center, z_center, x_size, y_size, z_size, yaw = box
        z_center = -z_center + 4.4
        pred_label = predictions['pred_labels'][i]
        pred_score = predictions['pred_scores'][i]

        marker = Marker()
        marker.header.frame_id = "os_sensor"
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
    """ rospy.loginfo(f"Published {len(marker_array.markers)} bounding boxes.") """


def pointcloud_callback(msg):
    """Callback to process incoming point clouds."""
    try:
        """ rospy.loginfo(f"Received point cloud. Time: {rospy.Time.now().secs}.{rospy.Time.now().nsecs}") """

        pointcloud_readonly = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        """ rospy.loginfo(f"Converted to read_only. Time: {rospy.Time.now().secs}.{rospy.Time.now().nsecs}") """

        # Flatten each field to 1D arrays and stack them into a 2D array
        x = pointcloud_readonly['x'].flatten()
        y = pointcloud_readonly['y'].flatten()
        z = -( pointcloud_readonly['z'].flatten()-4.4 )
        intensity = np.zeros_like(x)

        pointcloud = np.vstack((x, y, z, intensity)).T
        """ rospy.loginfo(f"Pointcloud reshaped to: {pointcloud.shape}") """

        # Filter out rows containing NaNs
        valid_points = ~np.isnan(pointcloud).any(axis=1)
        pointcloud = pointcloud[valid_points]
        """ rospy.loginfo(f"Filtered NaNs from point cloud. Remaining points: {pointcloud.shape[0]}") """

        
        """ rospy.loginfo(f"Converted point cloud to numpy array. Time: {rospy.Time.now().secs}.{rospy.Time.now().nsecs}") """


        # Preprocess point cloud
        # points = pc_preprocess(pointcloud)
        """ rospy.loginfo(f"Preprocessed point cloud. Time: {rospy.Time.now().secs}.{rospy.Time.now().nsecs}")
        rospy.loginfo(f"Preprocessed point cloud shape: {pointcloud.shape}")
        rospy.loginfo(f"Sample points after preprocessing: {pointcloud[:5]}") """


        # Perform inference
        global model
        global demo_dataset
        with torch.no_grad():
            data_dict = demo_dataset.process_point_cloud(pointcloud)
            data_dict = demo_dataset.collate_batch([data_dict])
            """ rospy.loginfo(f"Process for inference. Time: {rospy.Time.now().secs}.{rospy.Time.now().nsecs}") """
            load_data_to_gpu(data_dict)
            """ rospy.loginfo(f"Loaded to GPU. Time: {rospy.Time.now().secs}.{rospy.Time.now().nsecs}") """
            pred_dicts, _ = model.forward(data_dict)
            """ rospy.loginfo(f"Performed inference. Time: {rospy.Time.now().secs}.{rospy.Time.now().nsecs}")
            rospy.loginfo(f"Predictions: {pred_dicts}") """

        # Prepare predictions for publishing
        pred_dicts_serializable = {
            "pred_boxes": pred_dicts[0]['pred_boxes'].tolist(),
            "pred_scores": pred_dicts[0]['pred_scores'].tolist(),
            "pred_labels": pred_dicts[0]['pred_labels'].tolist()
        }
        """ rospy.loginfo(f"Prepared predictions for publishing. Time: {rospy.Time.now().secs}.{rospy.Time.now().nsecs}") """
        parse_and_publish(pred_dicts_serializable)
        """ rospy.loginfo(f"Published bounding boxes. Time: {rospy.Time.now().secs}.{rospy.Time.now().nsecs}") """

    except Exception as e:
        rospy.logerr(f"Error processing point cloud: {e}")


def main():
    global bbox_publisher
    global demo_dataset 
    rospy.init_node('pointcloud_subscriber', anonymous=True)
    bbox_publisher = rospy.Publisher('/ouster/bounding_boxes', MarkerArray, queue_size=10)

    args, cfg = parse_config()

    logger = common_utils.create_logger()
    
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger,
    )

    # Load the model once
    load_model()

    rospy.Subscriber('/ouster/points', PointCloud2, pointcloud_callback, queue_size=3)

    rospy.loginfo("PointCloud2 Subscriber Node Initialized.")
    rospy.spin()


if __name__ == '__main__':
    try:
        torch.cuda.empty_cache()
        main()
    except rospy.ROSInterruptException:
        pass
