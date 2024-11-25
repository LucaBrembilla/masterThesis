import argparse
import cv2
import numpy as np
import os
import torch
from utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from model import PointPillars

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    Filters points within a specified range.

    Args:
        pts (np.ndarray): Point cloud data of shape (N, 3) or (N, 4).
        point_range (list): [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        np.ndarray: Filtered points within the range.
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    return pts[keep_mask]

def calculate_iou_3d(bbox1, bbox2):
    """
    Calculate the IoU for two 3D bounding boxes.
    Each bbox is defined as [x, y, z, l, w, h, yaw].
    """
    def bbox_to_corners_3d(bbox):
        # Convert bbox to its corners in 3D space
        return bbox3d2corners_camera(bbox[None, :])[0]

    corners1 = bbox_to_corners_3d(bbox1)
    corners2 = bbox_to_corners_3d(bbox2)

    # Compute intersection volume
    intersection_volume = compute_intersection_volume(corners1, corners2)
    volume1 = compute_bbox_volume(corners1)
    volume2 = compute_bbox_volume(corners2)

    # Compute IoU
    union_volume = volume1 + volume2 - intersection_volume
    iou = intersection_volume / union_volume if union_volume > 0 else 0
    return iou

def compute_bbox_volume(corners):
    """
    Compute the volume of a 3D bounding box given its corners.
    """
    x_min, y_min, z_min = np.min(corners, axis=0)
    x_max, y_max, z_max = np.max(corners, axis=0)
    return max(0, x_max - x_min) * max(0, y_max - y_min) * max(0, z_max - z_min)

def compute_intersection_volume(corners1, corners2):
    """
    Compute the volume of the intersection of two 3D bounding boxes.
    """
    x_min = max(np.min(corners1[:, 0]), np.min(corners2[:, 0]))
    y_min = max(np.min(corners1[:, 1]), np.min(corners2[:, 1]))
    z_min = max(np.min(corners1[:, 2]), np.min(corners2[:, 2]))
    x_max = min(np.max(corners1[:, 0]), np.max(corners2[:, 0]))
    y_max = min(np.max(corners1[:, 1]), np.max(corners2[:, 1]))
    z_max = min(np.max(corners1[:, 2]), np.max(corners2[:, 2]))
    
    intersection_volume = max(0, x_max - x_min) * max(0, y_max - y_min) * max(0, z_max - z_min)
    return intersection_volume

def main(args):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
    }
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))

    if not os.path.exists(args.pc_path):
        raise FileNotFoundError
    pc = read_points(args.pc_path)
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc)
    
    if os.path.exists(args.calib_path):
        calib_info = read_calib(args.calib_path)
    else:
        calib_info = None
    
    if os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None

    if os.path.exists(args.img_path):
        img = cv2.imread(args.img_path, 1)
    else:
        img = None

    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()

        result_filter = model(batched_pts=[pc_torch], 
                              mode='test')[0]

    # Filter predictions for points within the valid LiDAR range
    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']

    # Display bounding box details with class names
    print("Detected Objects:")
    for i, bbox in enumerate(lidar_bboxes):
        class_name = LABEL2CLASSES.get(labels[i], "Unknown")
        bbox_str = " ".join(f"{coord:.2f}" for coord in bbox)
        print(f"{class_name} {bbox_str}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', help='your point cloud path')
    parser.add_argument('--calib_path', default='', help='your calib file path')
    parser.add_argument('--gt_path', default='', help='your ground truth path')
    parser.add_argument('--img_path', default='', help='your image path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)