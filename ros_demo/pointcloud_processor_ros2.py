import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
import torch
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# import ros2_numpy

import glob

# Configuration and paths
CFG_FILE = '/home/airlab/brembilla/masterThesis/tools/cfgs/kitti_models/pointrcnn_ros2.yaml'
CFG_FILE = '/home/airlab/brembilla/masterThesis/tools/cfgs/kitti_models/second_iou_ros2.yaml'
CFG_FILE = '/home/airlab/brembilla/masterThesis/tools/cfgs/kitti_models/pointpillar_ros.yaml'


CKPT = '/home/airlab/brembilla/masterThesis/pretrained/pointrcnn_7870.pth'
CKPT = '/home/airlab/brembilla/masterThesis/pretrained/second_iou7909.pth'
CKPT = '/home/airlab/brembilla/masterThesis/pretrained/pointpillar_7728.pth'


def pointcloud2_to_numpy_xyz(msg):
    """
    Converts a ROS2 PointCloud2 message into a NumPy array containing only xyz.

    :param msg: PointCloud2 message
    :return: NumPy array of shape (N, 3) with xyz coordinates, eliminating rows with NaNs.
    """
 
    pc_data = np.frombuffer(
        msg.data, dtype=np.uint8).reshape(-1, msg.point_step)

    xyz = pc_data[:, 0:12].view(dtype=np.float32).reshape(-1, 3)

    xyz_clean = xyz[~np.isnan(xyz).any(axis=1)]


    return xyz_clean



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
            1
        )

        # Initialize dataset and model
        self.demo_dataset = self.init_dataset()
        self.model = self.load_model()

        print("Ready for inference")

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
            # pointcloud =  ros2_numpy.point_cloud2.point_cloud2_to_array(msg)
            # pointcloud = point_cloud2.read_points(msg)
            """ using ros2_numpy

            pointcloud = ros2_numpy.point_cloud2.point_cloud2_to_array(msg)
            print(pointcloud)
            x = pointcloud['xyz'][:, 0].flatten()
            y = pointcloud['xyz'][:, 1].flatten()
            z = -(pointcloud['xyz'][:, 2].flatten() - 4.4)
            intensity = np.zeros_like(x)

            pointcloud_np = np.vstack((x, y, z, intensity)).T
            valid_points = ~np.isnan(pointcloud_np).any(axis=1)
            pointcloud_np = pointcloud_np[valid_points]

            """

            self.get_logger().info(f"Received point cloud.")
            # pointcloud = point_cloud2.read_points(msg, field_names=("xyz"), skip_nans=True)

            self.get_logger().info(f"Custom function.")
            pointcloud_np = pointcloud2_to_numpy_xyz(msg)
            pointcloud_np[:, 2] = 4.4 - pointcloud_np[:, 2]
            pointcloud_np = np.hstack([pointcloud_np, np.zeros((pointcloud_np.shape[0], 1))])

            self.get_logger().info(f"Converted to read_only.")

            """ Debug
            print(pointcloud)
            print(type(pointcloud))
            print(pointcloud.shape)

            print(pointcloud[0])
            print(type(pointcloud[0]))
            print(pointcloud[0].shape)
            """
            #pointcloud_list = list(pointcloud)
            self.get_logger().info(f"Converted to list.")


            # Extract all x values using list comprehension
            #x = [point[0] for point in pointcloud_list]
            #y = [point[1] for point in pointcloud_list]
            #z = [4.4 - point[2] for point in pointcloud]
            #intensity = np.zeros_like(x)

            self.get_logger().info(f"Extracted all values.")

            """ 
            # Optionally, print the first few x values
            print(x_values[:10])  # Prints the first 10 x values

            x = [point[0] for point in pointcloud]
            y = [point[1] for point in pointcloud]
            z = -([point[2] for point in pointcloud] - 4.4)
            intensity = np.zeros_like(x)
            """

            #pointcloud_np = np.vstack((x, y, z, intensity)).T
            #valid_points = ~np.isnan(pointcloud_np).any(axis=1)
            #pointcloud_np = pointcloud_np[valid_points]

            #print(pointcloud_np)
            #print(pointcloud_np.shape)


            """ Debug
            print(pointcloud_np)
            print(pointcloud_np.shape)
            print(pointcloud_np.dtype)
            print(type(pointcloud_np[0][0]))
            """
            self.get_logger().info(f"Ready for torch.")

            with torch.no_grad():
                data_dict = self.demo_dataset.process_point_cloud(pointcloud_np)
                # print(f"Data dict: {data_dict}")
                data_dict = self.demo_dataset.collate_batch([data_dict])
                # print(f"Data dict: {data_dict}")
                load_data_to_gpu(data_dict)
                # print(f"Data dict into the gpu: {data_dict}")
                self.get_logger().info(f"Loaded to gpu.")
                pred_dicts, _ = self.model.forward(data_dict)
                self.get_logger().info(f"Predictions ready.")

                # print(f"Predicted bounding boxes: {pred_dicts}")
            
            # Prepare predictions for publishing
            pred_dicts_serializable = {
                "pred_boxes": pred_dicts[0]['pred_boxes'].tolist(),
                "pred_scores": pred_dicts[0]['pred_scores'].tolist(),
                "pred_labels": pred_dicts[0]['pred_labels'].tolist()
            }
            self.get_logger().info(f"Call for publish.")

            self.publish_predictions(pred_dicts_serializable, msg.header.stamp)#
            self.get_logger().info(f"Published done.")


        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

    def publish_predictions(self, predictions, timestamp):
        marker_array = MarkerArray()
        for i, box in enumerate(predictions['pred_boxes']):

            # Skip if the score for auto is less than .4
            if predictions['pred_scores'][i] < 0.4 and predictions['pred_labels'][i] == 1:
                continue

            if predictions['pred_scores'][i] < 0.25:
                continue

            x_center, y_center, z_center, x_size, y_size, z_size, yaw = box
            z_center = -z_center + 4.4
            pred_label = predictions['pred_labels'][i]
            pred_score = predictions['pred_scores'][i]

            marker = Marker()
            marker.header.frame_id = "os_sensor"
            # The timestamp equal to the one of the read pointcloud can be useful for registering a new bag with the bounding boxes
            # marker.header.stamp = timestamp
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

        self.publisher.publish(marker_array)
        # print(f"Published {marker_array}")

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudInference()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
