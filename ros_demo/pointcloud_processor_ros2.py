import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
import torch
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import ros2_numpy

import glob

# Configuration and paths
CFG_FILE = '/home/airlab/brembilla/masterThesis/tools/cfgs/kitti_models/pointpillar_ros.yaml'
CKPT = '/home/airlab/brembilla/masterThesis/pretrained/pointpillar_7728.pth'

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

class PointCloudInference(Node):
    def __init__(self):
        super().__init__('pointcloud_inference')
        self.logger = common_utils.create_logger()
        self.publisher = self.create_publisher(MarkerArray, '/ouster/bounding_boxes', 10)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/ouster/points',
            self.pointcloud_callback,
            10
        )

        # Initialize dataset and model
        self.demo_dataset = self.init_dataset()
        self.model = self.load_model()

    def init_dataset(self):
        cfg_from_yaml_file(CFG_FILE, cfg)
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=Path('/media/airlab/000F9736000EFC5E/PNRR/ouster'),
            logger=self.logger,
        )
        return demo_dataset

    def load_model(self):
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        model.load_params_from_file(filename=CKPT, logger=self.logger, to_cpu=True)
        model.cuda()
        model.eval()
        return model

    def pointcloud_callback(self, msg):
        try:
            pointcloud = ros2_numpy.point_cloud2.point_cloud2_to_array(msg)
            print(pointcloud)
            x = pointcloud['xyz'][:, 0].flatten()
            y = pointcloud['xyz'][:, 1].flatten()
            z = -(pointcloud['xyz'][:, 2].flatten() - 4.4)
            intensity = np.zeros_like(x)

            pointcloud_np = np.vstack((x, y, z, intensity)).T
            valid_points = ~np.isnan(pointcloud_np).any(axis=1)
            pointcloud_np = pointcloud_np[valid_points]

            with torch.no_grad():
                data_dict = self.demo_dataset.process_point_cloud(pointcloud_np)
                data_dict = self.demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = self.model.forward(data_dict)

            self.publish_predictions(pred_dicts, msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

    def publish_predictions(self, pred_dicts, timestamp):
        marker_array = MarkerArray()
        for i, box in enumerate(pred_dicts[0]['pred_boxes']):
            x_center, y_center, z_center, x_size, y_size, z_size, yaw = box
            z_center = -z_center + 4.4

            marker = Marker()
            marker.header.frame_id = "os_sensor"
            marker.header.stamp = timestamp
            marker.ns = "bounding_boxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x_center
            marker.pose.position.y = y_center
            marker.pose.position.z = z_center
            marker.pose.orientation.z = math.sin(yaw / 2)
            marker.pose.orientation.w = math.cos(yaw / 2)
            marker.scale.x = x_size
            marker.scale.y = y_size
            marker.scale.z = z_size
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudInference()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
