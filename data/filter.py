import numpy as np

def filter_point_cloud(point_cloud_path, point_cloud_range):
    """
    Filters out points outside the specified point cloud range.
    
    Parameters:
    - point_cloud_path (str): Path to the .npy file containing the point cloud.
    - point_cloud_range (list): The range [x_min, y_min, z_min, x_max, y_max, z_max] to crop the point cloud.
    
    Returns:
    - filtered_points (numpy.ndarray): The filtered point cloud.
    """
    # Load the point cloud
    if point_cloud_path.endswith(".bin"):
        # Load the point cloud from the .bin file
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
    elif point_cloud_path.endswith(".npy"):
        # Load the point cloud from the .npy file
        point_cloud = np.load(point_cloud_path)
    
    #Print pointcloud max and min values
    print(f"Pointcloud min values. x: {np.min(point_cloud[:,0])}, y: {np.min(point_cloud[:,1])}, z: {np.min(point_cloud[:,2])}")
    print(f"Pointcloud max values. x: {np.max(point_cloud[:,0])}, y: {np.max(point_cloud[:,1])}, z: {np.max(point_cloud[:,2])}")

    # Extract the point cloud range
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    
    # Filter the points based on the specified range
    mask = (
        (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &   # x-axis range
        (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) &   # y-axis range
        (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)     # z-axis range
    )
    
    # Apply the mask to filter the points
    filtered_points = point_cloud[mask]

    #Print filtered pointcloud max and min values
    print(f"Filtered pointcloud min values. x: {np.min(filtered_points[:,0])}, y: {np.min(filtered_points[:,1])}, z: {np.min(filtered_points[:,2])}")
    print(f"Filtered pointcloud max values. x: {np.max(filtered_points[:,0])}, y: {np.max(filtered_points[:,1])}, z: {np.max(filtered_points[:,2])}")

    return filtered_points

def save_filtered_point_cloud(filtered_points, output_path):
    """
    Saves the filtered point cloud to a .npy file.
    
    Parameters:
    - filtered_points (numpy.ndarray): The filtered point cloud to save.
    - output_path (str): Path to save the filtered point cloud.
    """
    np.save(output_path, filtered_points)
    print(f"Filtered point cloud saved to: {output_path}")

def main():
    # Example input
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    point_cloud_path = "/home/brembilla/exp/shared_datasets/kitti/kitti/training/velodyne/000000.bin"
    # point_cloud_path = "/home/brembilla/exp/shared_datasets/PNRR/ouster/frame_0000.npy"
    output_path = "frame_filtered.npy"
    
    # Filter the point cloud
    filtered_points = filter_point_cloud(point_cloud_path, point_cloud_range)
    
    # Save the filtered point cloud
    save_filtered_point_cloud(filtered_points, output_path)

if __name__ == "__main__":
    main()
