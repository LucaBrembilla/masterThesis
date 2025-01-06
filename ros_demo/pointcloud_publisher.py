#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import rosbag
import os

def publish_pointcloud():
    rospy.init_node('pointcloud_publisher', anonymous=True)
    pub = rospy.Publisher('/ouster/points', PointCloud2, queue_size=10)

    # Set the path to your ROS bag file
    bag_path = "/home/brembilla/exp/shared_datasets/PNRR/2024-12-09-14-38-14.bag"
    if not os.path.exists(bag_path):
        rospy.logerr(f"Bag file not found at {bag_path}")
        return

    # Open the bag file
    with rosbag.Bag(bag_path, 'r') as bag:
        # Specify the topic containing PointCloud2 messages
        topic_name = "/ouster/points"  # Replace with your topic name in the bag
        if topic_name not in bag.get_type_and_topic_info().topics:
            rospy.logerr(f"Topic {topic_name} not found in the bag file.")
            return

        # Read and publish messages
        rospy.loginfo("Publishing messages from the bag file...")
        rate = rospy.Rate(0.1)  # Adjust rate as needed (e.g., 0.1 Hz)
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            if rospy.is_shutdown():
                break
            rospy.loginfo("Publishing point cloud message...")
            rospy.loginfo(f"Message timestamp: {msg.header.stamp}")
            rospy.loginfo(f"Message frame_id: {msg.header.frame_id}")
            pub.publish(msg)
            rate.sleep()

if __name__ == '__main__':
    try:
        publish_pointcloud()
    except rospy.ROSInterruptException:
        pass
