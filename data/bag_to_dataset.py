import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2
import os

bag_file = "/home/brembilla/exp/shared_datasets/PNRR/2024-12-09-14-38-14.bag"
topic = "/ouster/points"

def bag_to_np_dataset(bag_file, topic):
    # Create a folder to save frames
    output_folder = "/home/brembilla/exp/shared_datasets/PNRR/ouster"
    os.makedirs(output_folder, exist_ok=True)

    # Open the bag file
    bag = rosbag.Bag(bag_file)

    # Process each frame
    for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[topic])):
        pc_data = list(pc2.read_points(msg, field_names=["x", "y", "z", "intensity"], skip_nans=True))
        frame_array = np.array(pc_data, dtype=np.float32)
        np.save(os.path.join(output_folder, f"frame_{i:04d}.npy"), frame_array)
        print(f"Saved frame {i} with {len(frame_array)} points")

    bag.close()

    print(f"All frames saved to {output_folder}/")

if __name__ == "__main__":
  bag_to_np_dataset(bag_file, topic)