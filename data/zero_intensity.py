import numpy as np
import os

def load_pointcloud(file_path):
    """Load point cloud data from .bin or .npy file."""
    if file_path.endswith('.bin'):
        pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        # Assuming each point has x, y, z, intensity
    elif file_path.endswith('.npy'):
        pointcloud = np.load(file_path)
        # The shape should be (N, 3) or (N, 4)
    else:
        raise ValueError("Unsupported file format. Use .bin or .npy files.")
    
    return pointcloud

def set_zero_intensity(pointcloud):
    
    if pointcloud.shape[1] < 4:
        raise ValueError("Pointcloud data must have at least 4 dimensions (x, y, z, intensity).")
    
    print(f"Original mean intensity: {np.mean(pointcloud[:, 3])}")

    # Set intensity to 0
    pointcloud[:, 3] = 0

    print(f"New mean intensity: {np.mean(pointcloud[:, 3])}")

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

    print(f"Original mean z: {np.mean(pointcloud[:, 2])}")
    sub = 4.4
    pointcloud[:, 2] -= sub
    print(f"Substracted {sub}, new mean z: {np.mean(pointcloud[:, 2])}")
    pointcloud[:, 2] = -pointcloud[:, 2]
    print(f"Inverted mean z: {np.mean(pointcloud[:, 2])}")
    return pointcloud

def save_pointcloud(file_path, pointcloud):
    """Save the translated point cloud to a .npy file."""
    np.save(file_path, pointcloud)
    print(f"Pointcloud saved to {file_path}")

def main():
    # file_path = input("Enter the path to the point cloud file (.bin or .npy): ").strip()
    file_path = "/home/brembilla/exp/shared_datasets/PNRR/ouster/frame_0000.npy"
    
    # output_path = f"{file_path[:-3]}_traslated.npy"
    output_path = f"zeroed_{file_path.split('/')[-1]}"

    if not os.path.exists(file_path):
        print("File does not exist. Please check the path.")
        return

    try:
        pointcloud = load_pointcloud(file_path)
        print(f"Loaded point cloud with shape: {pointcloud.shape}")

        # Translate point cloud to target mean z
        pointcloud = set_zero_intensity(pointcloud)

        # Save the translated point cloud
        save_pointcloud(output_path, pointcloud)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()