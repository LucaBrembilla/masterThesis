import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
import math

def get_2d_corners(box):
    """
    Compute the (x,y) corners of the 2D rectangle corresponding to the base of the 3D box.
    
    Args:
        box (np.ndarray): A 7-element array [x, y, z, dx, dy, dz, heading]
        
    Returns:
        np.ndarray: A (4, 2) array of corners in (x, y) order.
    """
    x, y, z, dx, dy, dz, heading = box
    # The 2D rectangle is centered at (x, y) with dimensions dx, dy.
    # The local (unrotated) corners relative to center are:
    #   (-dx/2, -dy/2), (-dx/2, dy/2), (dx/2, dy/2), (dx/2, -dy/2)
    corners = np.array([
        [-dx/2, -dy/2],
        [-dx/2,  dy/2],
        [ dx/2,  dy/2],
        [ dx/2, -dy/2]
    ])
    
    # Rotation matrix for heading (around z-axis)
    cos_angle = math.cos(heading)
    sin_angle = math.sin(heading)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle,  cos_angle]
    ])
    
    # Rotate and translate the corners
    rotated_corners = np.dot(corners, rotation_matrix.T)
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    return rotated_corners

def compute_z_overlap(box1, box2):
    """
    Compute the overlap along the z-axis between two boxes.
    
    Args:
        box1, box2 (np.ndarray): Each is a 7-element array [x, y, z, dx, dy, dz, heading].
        
    Returns:
        float: The length of overlap along z-axis.
    """
    # For each box, we assume the center z is given and the height is dz.
    z1, dz1 = box1[2], box1[5]
    z2, dz2 = box2[2], box2[5]
    z1_min, z1_max = z1 - dz1 / 2.0, z1 + dz1 / 2.0
    z2_min, z2_max = z2 - dz2 / 2.0, z2 + dz2 / 2.0
    
    overlap_min = max(z1_min, z2_min)
    overlap_max = min(z1_max, z2_max)
    overlap = max(0.0, overlap_max - overlap_min)
    return overlap

def pairwise_3d_iou(current_boxes, predicted_boxes):
    """
    Compute the pairwise 3D IoU between two sets of oriented boxes.
    
    Args:
        current_boxes (np.ndarray): Array of shape (N, 7) for N boxes.
        predicted_boxes (np.ndarray): Array of shape (M, 7) for M boxes.
    
    Returns:
        np.ndarray: An (N, M) IoU matrix.
    """
    N = current_boxes.shape[0]
    M = predicted_boxes.shape[0]
    iou_matrix = np.zeros((N, M), dtype=np.float32)
    
    for i in range(N):
        box1 = current_boxes[i]
        # Create the 2D polygon for box1 base.
        poly1 = Polygon(get_2d_corners(box1))
        vol1 = box1[3] * box1[4] * box1[5]  # dx * dy * dz
        
        for j in range(M):
            box2 = predicted_boxes[j]
            poly2 = Polygon(get_2d_corners(box2))
            vol2 = box2[3] * box2[4] * box2[5]
            
            # Compute intersection area in bird's eye view
            if not poly1.intersects(poly2):
                inter_area = 0.0
            else:
                inter_area = poly1.intersection(poly2).area
                
            # Compute overlap in z direction
            z_overlap = compute_z_overlap(box1, box2)
            
            # Intersection volume
            inter_vol = inter_area * z_overlap
            
            # Union volume
            union_vol = vol1 + vol2 - inter_vol
            if union_vol > 0:
                iou_matrix[i, j] = inter_vol / union_vol
            else:
                iou_matrix[i, j] = 0.0
    return iou_matrix

def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem using the Hungarian algorithm.
    
    Args:
        cost_matrix (np.ndarray): A 2D cost matrix.
        
    Returns:
        list of tuple: Matched indices as (row_index, col_index).
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))

def match_detections_to_tracks(current_boxes, predicted_boxes, iou_threshold=0.3):
    """
    Match current detections to predicted tracks using 3D IoU matching.
    
    Args:
        current_boxes (np.ndarray): Array of shape (N, 7) for N detection boxes.
        predicted_boxes (np.ndarray): Array of shape (M, 7) for M track boxes.
        iou_threshold (float): Minimum IoU required to consider a match valid.
        
    Returns:
        list of tuple: List of matched pairs as (detection_index, track_index).
    """
    iou_matrix = pairwise_3d_iou(current_boxes, predicted_boxes)
    # Convert IoU to cost (maximize IoU is equivalent to minimize negative IoU)
    cost_matrix = -iou_matrix
    matched_indices = linear_assignment(cost_matrix)
    # Filter matches by IoU threshold
    valid_matches = [(d, t) for d, t in matched_indices if iou_matrix[d, t] > iou_threshold]
    return valid_matches

# Example usage:
if __name__ == '__main__':
    # Example boxes with format: [x, y, z, dx, dy, dz, heading]
    current_boxes = np.array([
        [0, 0, 0, 2, 2, 2, 0],            # Box 1, heading 0
        [3, 3, 3, 2, 2, 2, np.pi/4]        # Box 2, rotated 45 degrees
    ], dtype=np.float32)
    
    predicted_boxes = np.array([
        [0.5, 0.5, 0, 2, 2, 2, 0.1],       # Similar to Box 1
        [3.2, 3.2, 3, 2, 2, 2, np.pi/3],    # Similar to Box 2
        [10, 10, 10, 2, 2, 2, 0]            # Distant box
    ], dtype=np.float32)
    
    matches = match_detections_to_tracks(current_boxes, predicted_boxes, iou_threshold=0.1)
    #print("Matched pairs:", matches)
