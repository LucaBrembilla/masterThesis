import argparse
import glob
from pathlib import Path

"""
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False
"""

import numpy as np
import torch

import _init_path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from tools.streaming_utilis_a9.temporal_state_nuscenes import update_temporal_state
from tools.streaming_utilis_a9.crop import crop_point_cloud

def pc_preprocess(pointcloud):
    
    if pointcloud.shape[1] < 4:
        raise ValueError("Pointcloud data must have at least 4 dimensions (x, y, z, intensity).")
    
    # print(f"Original mean intensity: {np.mean(pointcloud[:, 3])}")

    # Set intensity to 0
    pointcloud[:, 3] = 0

    # print(f"New mean intensity: {np.mean(pointcloud[:, 3])}")

    """
    print(f"Original mean z: {np.mean(pointcloud[:, 2])}")
    sub = np.max(pointcloud[:, 2])
    print(f"Original mean z: {np.mean(pointcloud[:, 2])}")
    pointcloud[:, 2] -= sub
    print(f"Substracted {sub}, new mean z: {np.mean(pointcloud[:, 2])}")
    pointcloud[:, 2] = -pointcloud[:, 2]
    print(f"Inverted mean z: {np.mean(pointcloud[:, 2])}")
    pointcloud[:, 2] = pointcloud[:, 2] - np.mean(pointcloud[:, 2])
    """

    # print(f"Original mean z: {np.mean(pointcloud[:, 2])}")
    sub =- 5.5
    pointcloud[:, 2] -= sub
    # print(f"Substracted {sub}, new mean z: {np.mean(pointcloud[:, 2])}")
    return pointcloud

def bb_postprocess(input_file, output_file, mean_value=0, add_amount=4.4):
    """
    Process the fourth number in each line of the input file by inverting it
    and adding a specified amount, then save the results to the output file.

    Parameters:
    - input_file (str): Path to the input text file.
    - output_file (str): Path to the output text file.
    - mean_value (float): Mean value to add to each number before inverting.
    - add_amount (float): Amount to add to each inverted number.
    """
    try:
        # Read the file line by line
        with open(input_file, 'r') as file:
            lines = file.readlines()
        
        processed_lines = []
        
        # Process each line
        for line in lines:
            # Split the line into numbers
            numbers = line.split()
            # Process the third number (index 2)
            numbers[3] = str((float(numbers[3]) + mean_value) + add_amount)
            # Convert back to a space-separated string
            processed_line = " ".join(numbers)
            processed_lines.append(processed_line)
        
        # Write the processed lines to the output file
        with open(output_file, 'w') as file:
            file.write("\n".join(processed_lines))
        
        print(f"Post-processing complete! Results saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


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

        if points.shape[1] == 3:
            points = np.concatenate((points, np.zeros((points.shape[0], 1), dtype=points.dtype)), axis=1)
        # print("Modifying the input pointcloud...")
        points = pc_preprocess(points)

        return points
        

        # print("Point min:", np.min(points, axis=0))
        # print("Point max:", np.max(points, axis=0))
        # print("Point mean:", np.mean(points, axis=0))

        # print("Sample list 0:", self.sample_file_list[index])
        """
        try:
            frame_id = self.sample_file_list[index].split('/')[-1].split('.')[0]
        except:
            frame_id = self.sample_file_list[index].parts[-1].split('.')[0]
        # print(f"Frame ID: {frame_id}")
        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
        """


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
    tracker = None
    prev_detections = None
    timestamp_previous = 0

    with torch.no_grad():
        for idx, points in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            # print(f'data dicts: {data_dict}')
            
            try:
                frame_id = demo_dataset.sample_file_list[idx].split('/')[-1].split('.')[0]
            except:
                frame_id = demo_dataset.sample_file_list[idx].parts[-1].split('.')[0]
            
            print(f"Frame ID: {frame_id}")
            timestamp_list = str(frame_id).split('_')[:2]
            timestamp = float(timestamp_list[0] + '.' + timestamp_list[1])
            time_step = timestamp - timestamp_previous

            if False and idx % 2 and not time_step>0.3:
                
                expand_ratio = 1.5

                pointcloud_tensor = torch.from_numpy(points).cuda()
                xyz = pointcloud_tensor[:, :3]
                boxes_np = np.stack([track['box'] for track in tracker['track_states']])

                # Convert the stacked numpy array to a torch tensor and move it to the GPU.
                boxes = torch.from_numpy(boxes_np).cuda()

                # Extract box components
                bx, by, bz, dx, dy, dz, heading = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6]

                # self.get_logger().info(f"bx {bx}")
                dx += 7
                dy += 2
                # Expand box dimensions
                dx, dy, dz = dx * expand_ratio, dy * expand_ratio, dz * expand_ratio

                # Translate points to the box center
                x_rel = xyz[:, 0].unsqueeze(1) - bx.unsqueeze(0)  # Shape: (N, M)
                y_rel = xyz[:, 1].unsqueeze(1) - by.unsqueeze(0)
                z_rel = xyz[:, 2].unsqueeze(1) - bz.unsqueeze(0)

                # Compute rotation matrices (around Z-axis)
                cos_theta = torch.cos(heading)
                sin_theta = torch.sin(heading)

                # Rotate points into the local coordinate frame of each box
                x_rot = x_rel * cos_theta.unsqueeze(0) + y_rel * sin_theta.unsqueeze(0)
                y_rot = -x_rel * sin_theta.unsqueeze(0) + y_rel * cos_theta.unsqueeze(0)

                # Check which points are inside the boxes
                in_x = (x_rot.abs() <= (dx / 2).unsqueeze(0))
                in_y = (y_rot.abs() <= (dy / 2).unsqueeze(0))
                in_z = (z_rel.abs() <= (dz / 2).unsqueeze(0))

                # Combine conditions: A point is valid if it is inside *any* box
                inside_any_box = (in_x & in_y & in_z).any(dim=1)

                # Filter points
                pointcloud_tensor = pointcloud_tensor[inside_any_box]
                
                # self.get_logger().info(f"Cropping for frame {self.counter}, new #points: {pointcloud_tensor.size()}")

                points = pointcloud_tensor.cpu().numpy()
                print(f"New points shape: {points.shape}")              

            if points.shape[-1] == 4:
                points = np.concatenate((points, np.full((points.shape[0], 1), timestamp, dtype=points.dtype)), axis=1)

            input_dict = {
                'points': points,
                'frame_id': frame_id,
            }            

            data_dict = demo_dataset.prepare_data(data_dict=input_dict)

            file_name = data_dict['frame_id']

            data_dict = demo_dataset.collate_batch([data_dict])

            load_data_to_gpu(data_dict)

            pred_dicts, _ = model.forward(data_dict)
            print(f"Prediction: {pred_dicts}")

            # Update tracking 
            if pred_dicts['pred_boxes'].shape[0] > 0:
                prev_detections, tracker = update_temporal_state(
                    pred_dicts, 
                    tracker,
                    time_step = timestamp_previous - timestamp
                )
            timestamp_previous = timestamp

            # Save the predictions to a file
            # output_file = Path("/home/brembilla/exp/output/pnrr") / (Path(args.data_path).stem + str(idx) +".txt")
            output_file = Path(f"/home/brembilla/exp/private_datasets/providentia/_predictions_less_computation/{file_name}.txt")

            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Format of the results:
            # pred_label x_center y_center z_center x_size y_size z_size yaw pred_score pred_cls_score pred_iou_scores
            # Labels: 1 - Car/Vehicle, 2 - Pedestrian, 3 - Cyclist
            with open(output_file, "w") as f:
                for i, box in enumerate(pred_dicts[0]['pred_boxes']):
                    # Extract individual values
                    x_center, y_center, z_center, x_size, y_size, z_size, yaw, velocity_x, velocity_y = box.tolist()
                    pred_label = pred_dicts[0]['pred_labels'][i].item()
                    pred_score = pred_dicts[0]['pred_scores'][i].item()

                    if pred_score < 0.3:
                        continue

                    if pred_label > 3:
                        continue
                    # pred_cls_score = pred_dicts[0]['pred_cls_scores'][i].item()
                    # pred_iou_score = pred_dicts[0]['pred_iou_scores'][i].item()

                    # Format the output line
                    # line = f"{pred_label} {x_center:.4f} {y_center:.4f} {z_center:.4f} {x_size:.4f} {y_size:.4f} {z_size:.4f} {yaw:.4f} {pred_score:.4f} {pred_cls_score:.4f} {pred_iou_score:.4f}"
                    line = f"{pred_label} {x_center:.4f} {y_center:.4f} {z_center:.4f} {x_size:.4f} {y_size:.4f} {z_size:.4f} {yaw:.4f} {pred_score:.4f}"
                    f.write(line + "\n")

            # Postprocess the bounding boxes
            bb_postprocess(output_file, output_file, mean_value=0, add_amount=- 5.5)

            """
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
            """
            logger.info(f'Inference done. Results are saved in {output_file}')
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
