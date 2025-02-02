import numpy as np

def get_box_corners(cx, cy, dx, dy, heading, expand_ratio=1.2):
    dx_exp = dx * expand_ratio
    dy_exp = dy * expand_ratio
    # Half dimensions
    hx, hy = dx_exp / 2, dy_exp / 2
    # Define corners in the box coordinate frame (clockwise or counterclockwise)
    corners = np.array([
        [-hx, -hy],
        [-hx,  hy],
        [ hx,  hy],
        [ hx, -hy]
    ])
    # Rotation matrix for heading
    R = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading),  np.cos(heading)]
    ])
    # Rotate and translate corners
    corners_rotated = np.dot(corners, R.T) + np.array([cx, cy])

    return corners_rotated

def filter_by_aabb(points_xy, corners):
    # Compute AABB for the rotated box
    x_min, y_min = np.min(corners, axis=0)
    x_max, y_max = np.max(corners, axis=0)
    return (points_xy[:, 0] >= x_min) & (points_xy[:, 0] <= x_max) & \
           (points_xy[:, 1] >= y_min) & (points_xy[:, 1] <= y_max)

def point_in_rotated_box(points_xy, corners):
    # For each edge, compute cross product sign between edge and vector from a corner to the point.
    # For a convex polygon, all these cross products should have the same sign for points inside.
    def is_inside(pt):
        signs = []
        num_corners = corners.shape[0]
        for i in range(num_corners):
            p1 = corners[i]
            p2 = corners[(i+1) % num_corners]
            edge = p2 - p1
            to_point = pt - p1
            cross = edge[0] * to_point[1] - edge[1] * to_point[0]
            signs.append(cross)
        # Check if all cross products are non-negative or non-positive.
        return np.all(np.array(signs) >= 0) or np.all(np.array(signs) <= 0)

    # Vectorized version: this might need a loop over candidate points if not using shapely or similar libraries.
    inside = np.array([is_inside(pt) for pt in points_xy])
    return inside


def crop_point_cloud(points, prev_detections, expand_ratio=1.2):
    """
    Crop the point cloud based on detection boxes without rotating the entire point cloud.
    Points are in the format (0, x, y, z, intensity).
    """

    print("Sample points:\n", points[:15])
    print("Sample detections:\n", prev_detections)

    if prev_detections is None or len(prev_detections) == 0:
        return points

    mask_total = np.zeros(points.shape[0], dtype=bool)
    # Extract global x, y, and z from points
    x_global = points[:, 1]
    y_global = points[:, 2]
    z_global = points[:, 3]
    points_xy = np.stack([x_global, y_global], axis=1)

    for det in prev_detections:
        cx, cy, cz, dx, dy, dz, heading = det

        # Get rotated box corners in global coordinates.
        corners = get_box_corners(cx, cy, dx, dy, heading, expand_ratio)

        # Filter points with a quick AABB test.
        aabb_mask = filter_by_aabb(points_xy, corners)
        candidate_points = points_xy[aabb_mask]

        # For candidates, perform a point-in-polygon test.
        inside_polygon = point_in_rotated_box(candidate_points, corners)

        # Combine with z-filter.
        candidate_indices = np.nonzero(aabb_mask)[0]
        z_mask = np.abs(z_global[candidate_indices] - cz) <= (dz * expand_ratio / 2)
        final_mask = inside_polygon & z_mask

        # Mark these points in the global mask.
        mask_total[candidate_indices] |= final_mask

        print(f"Number of points inside boxes: {mask_total.sum()}")

    return points[mask_total]

def crop_point_cloud_3_axes(points, prev_detections, expand_ratio=1.2):
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
    print("Sample points:\n", points[:15])
    print("Sample detections:\n", prev_detections)

    # If no previous detections, return the full point cloud.
    if prev_detections is None or len(prev_detections) == 0:
        return points

    # Make sure prev_detections is a NumPy array.
    # prev_detections = np.array(prev_detections)
    
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
        cos_angle = np.cos(-heading) # TODO: it may be +heading for KITTI
        sin_angle = np.sin(-heading)
        R = np.array([
            [cos_angle, -sin_angle],
            [sin_angle,  cos_angle]
        ])

        # Subtract the center (cx, cy) from each point (only x and y).
        points_xy = points[:, :2] - np.array([cx, cy])
        # Rotate the points into the local coordinate frame of the box.
        points_local = np.dot(points_xy, R.T) # TODO: points_local = np.dot(R, points_xy.T).T Maybe this one

        # Check if points are within half the expanded dimensions in x and y.
        in_box_xy = (np.abs(points_local[:, 0]) <= dx_exp / 2) & (np.abs(points_local[:, 1]) <= dy_exp / 2)
        # Check z: points must be within half the expanded height.
        in_box_z = np.abs(points[:, 2] - cz) <= dz_exp / 2

        # Points inside this detection.
        mask_det = in_box_xy & in_box_z

        # Combine masks: a point is kept if it is inside any expanded detection box.
        mask = mask | mask_det

        print(f"Number of points inside boxes: {mask.sum()}")

    # Return only the points that are within one or more expanded boxes.
    return points[mask]

# Example usage of crop_point_cloud
if __name__ == '__main__':
    points =  np.array([[0.0000e+00, 7.0175e+01, 2.3550e+00, 2.5830e+00, 0.0000e+00],
        [0.0000e+00, 6.9983e+01, 2.5690e+00, 2.5770e+00, 0.0000e+00],
        [0.0000e+00, 6.9456e+01, 2.9870e+00, 2.5600e+00, 0.0000e+00]])
    box = [6.961555, 2.698988, 2.61757736, 3.8396633, 1.5627036, 1.4872997, 3.2178032]
    corners_rotated = get_box_corners(58.961555, 16.698988, 3.8396633, 1.5627036, 3.2178032)
    print("Corners rotated: ", corners_rotated)

    crop_point_cloud(points, box)

# Example usage of crop_point_cloud_3_axes
if __name__ == '__main__old':
    # --- Deterministic Test Case ---
    #
    # Define a small, fixed point cloud (6 points).
    # Format for points: [x, y, z, ...] (we only care about x, y, z)
    points = np.array([
        [0.0,  0.0,  0.0],  # Expected to be inside the box.
        [1.0,  1.0,  0.0],  # Expected to be inside the box.
        [3.0,  3.0,  0.0],  # Expected to be outside the box.
        [-3.0, -3.0, 0.0],  # Expected to be outside the box.
        [2.0, -1.0,  0.0],  # Expected to be inside the box.
        [5.0,  5.0,  0.0]   # Expected to be outside the box.
    ])

    # Define a single detection box.
    # Format: [x, y, z, dx, dy, dz, heading]
    # Let's define a box centered at (0,0,0) with dimensions 4x4x4 (dx, dy, dz) and no rotation (heading=0).
    # With expand_ratio=1.2, the effective dimensions become 4.8 x 4.8 x 4.8.
    # In the xy-plane, half-dimensions are 2.4. So the box will include points with x, y in [-2.4, 2.4]
    # and z in [-2.4, 2.4] (since the center is at z=0).
    prev_detections = np.array([
        [0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0]
    ])

    # Crop the point cloud using the above detection.
    cropped_points = crop_point_cloud_3_axes(points, prev_detections, expand_ratio=1.2)
    
    print("Original point cloud:")
    print(points)
    print("\nCropped point cloud:")
    print(cropped_points)
    
    # Expected cropped points (manually determined):
    # Points [0,0,0], [1,1,0], and [2,-1,0] are within the expanded box.
    expected_points = np.array([
        [0.0,  0.0,  0.0],
        [1.0,  1.0,  0.0],
        [2.0, -1.0,  0.0]
    ])
    
    # Check if the cropped points match the expected ones.
    # Here we sort both arrays by rows for consistency before comparing.
    cropped_sorted = np.array(sorted(cropped_points.tolist()))
    expected_sorted = np.array(sorted(expected_points.tolist()))
    
    print("\nExpected cropped point cloud:")
    print(expected_sorted)
    
    # Use np.allclose to compare the arrays.
    if np.allclose(cropped_sorted, expected_sorted):
        print("\nTest passed: The cropped point cloud matches the expected result.")
    else:
        print("\nTest failed: The cropped point cloud does not match the expected result.")