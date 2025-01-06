#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os
import subprocess
from visualization_msgs.msg import Marker, MarkerArray
import math
import json
from ros_demo import process_point_cloud

# Paths to your OpenPCDet configuration and model checkpoint
CFG_FILE = '/home/brembilla/exp/tools/cfgs/kitti_models/pv_rcnn_ros.yaml'
CKPT = '/home/brembilla/exp/pretrained/pv_rcnn_8369.pth'  # Replace with the actual path to your checkpoint
OUTPUT_DIR = '/home/brembilla/exp/data/ros/'  # Directory to save .npy files

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_and_publish(predictions):
    
    marker_array = MarkerArray()

    for i, box in enumerate(predictions['pred_boxes']):
        
        x_center, y_center, z_center, x_size, y_size, z_size, yaw = box
        pred_label = predictions['pred_labels'][i]
        pred_score = predictions['pred_scores'][i]  
        # pred_cls_score = predictions['pred_cls_scores'][i]  
        # pred_iou_score = predictions['pred_iou_scores'][i] 

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_boxes"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Set pose and dimensions
        marker.pose.position.x = x_center
        marker.pose.position.y = y_center
        marker.pose.position.z = z_center
        marker.pose.orientation.z = math.sin(yaw / 2)
        marker.pose.orientation.w = math.cos(yaw / 2)
        marker.scale.x = x_size
        marker.scale.y = y_size
        marker.scale.z = z_size

        # Set color based on type
        if pred_label == 1:  # Car
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = pred_score  # Transparency
        elif pred_label == 2:  # Pedestrian
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = pred_score
        elif pred_label == 3:  # Cyclist
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = pred_score
        else: 
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = pred_score

        marker_array.markers.append(marker)

    bbox_publisher.publish(marker_array)
    rospy.loginfo(f"Published {len(marker_array.markers)} bounding boxes.")

def pointcloud_callback(msg):
    """
    Callback to save incoming point cloud data and call OpenPCDet for inference.
    """
    global bbox_publisher
    try:
        # Convert PointCloud2 to numpy array
        pointcloud = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=["x", "y", "z", "intensity"])))
        rospy.loginfo(f"Received point cloud with {pointcloud.shape[0]} points.")


        # Call the OpenPCDET script
        result = process_point_cloud(CFG_FILE, CKPT, pointcloud)

        rospy.loginfo("Inference completed successfully.")
        # rospy.loginfo(f"Output: {result}")

        parse_and_publish(result)

    except Exception as e:
        rospy.logerr(f"Error processing point cloud: {e}. Command output: {e.stderr}")

def main():
    global bbox_publisher
    rospy.init_node('pointcloud_subscriber', anonymous=True)
    bbox_publisher = rospy.Publisher('/ouster/bounding_boxes', MarkerArray, queue_size=10)
    rospy.Subscriber('/ouster/points', PointCloud2, pointcloud_callback)
    rospy.loginfo("PointCloud2 Subscriber Node Initialized.")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
