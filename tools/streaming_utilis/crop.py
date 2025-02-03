import numpy as np
import math

def crop_point_cloud(pointcloud, boxes, expand_ratio = 1.2):
    """
    Find points in a point cloud that lie within any of the given 3D bounding boxes.
    
    Args:
        pointcloud: numpy.ndarray of shape (N, 5) with columns (0, x, y, z, intensity)
        boxes: numpy.ndarray of shape (M, 7) with columns (x, y, z, dx, dy, dz, heading)
        
    Returns:
        numpy.ndarray: Subset of input points that lie within any of the bounding boxes
    """
    # Extract XYZ coordinates from pointcloud (N x 3 array)
    xyz = pointcloud[:, 1:4]
    mask = np.zeros(len(pointcloud), dtype=bool)
    
    # Extract components for better memory access
    x = xyz[:, 0]
    y = xyz[:, 1]
    z_coords = xyz[:, 2]
    
    for box in boxes:
        # Unpack box parameters
        bx, by, bz, dx, dy, dz, heading = box[:7]
        cos_theta = math.cos(heading)
        sin_theta = math.sin(heading)
        
        # Translate points to box coordinates
        x_rel = x - bx
        y_rel = y - by
        z_rel = z_coords - bz
        
        # Rotate points to box's local coordinate system (around Z-axis)
        x_rot = x_rel * cos_theta + y_rel * sin_theta
        y_rot = -x_rel * sin_theta + y_rel * cos_theta
        
        # Check containment in all dimensions using vectorized operations
        in_x = np.abs(x_rot) <= (dx*expand_ratio)/2
        in_y = np.abs(y_rot) <= (dy*expand_ratio)/2
        in_z = np.abs(z_rel) <= (dz*expand_ratio)/2
        
        # Update global mask with OR operation
        mask |= in_x & in_y & in_z
    
    return pointcloud[mask]

if __name__ == "__main__":
    # Example usage
    points = np.array([[0, 1.0, 2.0, 3.0, 0.5],
                    [0, 4.0, 5.0, 6.0, 0.7],
                    [0, 7.0, 8.0, 3.0, 0.5],
                    ])  # Shape (N, 5)

    boxes = np.array([[1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.0],
                    [4.0, 5.0, 3.0, 1.0, 1.0, 1.0, np.pi/4],
                    ])  # Shape (M, 7)

    filtered_points = crop_point_cloud(points, boxes, expand_ratio=1.2)

    print(filtered_points)