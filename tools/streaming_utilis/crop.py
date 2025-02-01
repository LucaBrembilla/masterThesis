import numpy as np

def crop_point_cloud(points, prev_detections, expand_ratio=1.2):
    """
    Crop the point cloud to keep only points inside the expanded previous detections.

    Args:
        points (np.ndarray): Array of shape (N, D) where the first 3 columns are [x, y, z].
        prev_detections (np.ndarray or list): Array or list of detections, each in the format
            [x, y, z, dx, dy, dz, heading].
        expand_ratio (float): Ratio by which to expand each dimension of the detection box.

    Returns:
        np.ndarray: Cropped point cloud (subset of the input points).
    """
    # If no previous detections, return the full point cloud.
    if prev_detections is None or len(prev_detections) == 0:
        return points

    # Make sure prev_detections is a NumPy array.
    prev_detections = np.array(prev_detections)
    
    # Create a boolean mask for all points, initially False.
    mask = np.zeros(points.shape[0], dtype=bool)

    # Loop over each detection and mark points that fall inside the expanded box.
    for det in prev_detections:
        cx, cy, cz, dx, dy, dz, heading = det

        # Expand box dimensions.
        dx_exp = dx * expand_ratio
        dy_exp = dy * expand_ratio
        dz_exp = dz * expand_ratio

        # For the horizontal (xy) plane, transform points to the detection’s local coordinate system.
        # Compute the cosine and sine for -heading to align the box with the axes.
        cos_angle = np.cos(-heading)
        sin_angle = np.sin(-heading)
        R = np.array([
            [cos_angle, -sin_angle],
            [sin_angle,  cos_angle]
        ])

        # Subtract the center (cx, cy) from each point (only x and y).
        points_xy = points[:, :2] - np.array([cx, cy])
        # Rotate the points into the local coordinate frame of the box.
        points_local = np.dot(points_xy, R.T)

        # Check if points are within half the expanded dimensions in x and y.
        in_box_xy = (np.abs(points_local[:, 0]) <= dx_exp / 2) & (np.abs(points_local[:, 1]) <= dy_exp / 2)
        # Check z: points must be within half the expanded height.
        in_box_z = np.abs(points[:, 2] - cz) <= dz_exp / 2

        # Points inside this detection.
        mask_det = in_box_xy & in_box_z

        # Combine masks: a point is kept if it is inside any expanded detection box.
        mask = mask | mask_det

    # Return only the points that are within one or more expanded boxes.
    return points[mask]

# Example usage:
if __name__ == '__main__':
    # Example point cloud: 1000 random points in a 20x20x4 cube.
    np.random.seed(0)
    points = np.random.uniform(low=[-10, -10, -2], high=[10, 10, 2], size=(1000, 3))
    
    # Suppose previous detections (two boxes):
    # Box format: [x, y, z, dx, dy, dz, heading]
    prev_detections = np.array([
        [0, 0, 0, 4, 4, 4, 0],         # Centered at (0,0,0) with no rotation.
        [5, 5, 0, 4, 4, 4, np.pi / 4]   # Centered at (5,5,0) rotated 45 degrees.
    ])

    cropped_points = crop_point_cloud(points, prev_detections, expand_ratio=1.2)
    print("Original point count:", points.shape[0])
    print("Cropped point count:", cropped_points.shape[0])
