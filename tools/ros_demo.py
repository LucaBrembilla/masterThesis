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


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

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
            output_bag_file = f"predictions_{idx+1:05d}.bag"

            # Save predictions to bag
            with rosbag.Bag(output_bag_file, 'w') as bag:
                for idx, pred_dict in enumerate(pred_dicts):
                    """
                    print(f"Saving predictions for frame {idx}")
                    print("Predictions:")
                    print(pred_dict)
                    """
                    pred_boxes, pred_scores, pred_labels = process_predictions(pred_dict)
                    markers = save_predictions_as_ros_markers(pred_boxes, idx)
                    bag.write('/predictions', markers)

            print(f"Predictions saved to {output_bag_file}")

    logger.info('Demo done.')


if __name__ == "__main__":
    main()