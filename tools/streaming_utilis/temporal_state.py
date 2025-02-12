import numpy as np
import torch

from tools.streaming_utilis.match_detections_to_tracks import match_detections_to_tracks

def update_temporal_state(pred_dicts, tracker=None, motion_model='linear', time_step=0.1, current_pose_mat=None):
    """
    Maintain temporal state using motion model predictions and compensate for sensor ego-motion.
    
    Args:
        pred_dicts: List of prediction dictionaries from current frame.
        tracker: Previous tracker state (None for first frame).
        motion_model: Type of motion prediction ('linear', 'kalman', 'constant_velocity').
        time_step: Time between frames (seconds).
        current_pose_mat: (4x4 numpy array) Current LiDAR pose as a homogeneous transformation matrix.
    
    Returns:
        updated_detections: Tensor of detections with motion-corrected boxes.
        updated_tracker: Updated tracking state.
    """
    current_boxes = pred_dicts[0]['pred_boxes']  # (N, 7) tensor [x,y,z,dx,dy,dz,heading]
    current_boxes_np = current_boxes.cpu().numpy()

    # Limit to 7 columns (x, y, z, dx, dy, dz, heading). For nuScenes
    if current_boxes_np[0].shape[0] > 7:
        current_velocities = current_boxes_np[:, 7:9]  # Extract velocities (vx, vy)
        current_boxes_np = current_boxes_np[:, :7]
    
    if tracker is None:
        tracker = {
            'track_ids': np.arange(len(current_boxes_np)).tolist(),
            'track_states': [
                {
                    'box': box.copy(),
                    'velocity': np.array([current_velocities[i][0], current_velocities[i][1], 0])  # Assuming z velocity = 0
                } for i,box in enumerate(current_boxes_np)
            ],
            'motion_model': motion_model,
            'last_timestamp': time_step,
            'lidar_pose': current_pose_mat  # Store current LiDAR pose.
        }
        return current_boxes, tracker
    
    # ---- Compensate for Ego-Motion ----
    # Get previous LiDAR pose (4x4 matrix) from tracker state.
    T_prev = tracker['lidar_pose']
    T_curr = current_pose_mat
    # Compute the relative transformation:
    # T_rel maps coordinates from the previous sensor frame into the current sensor frame.
    T_rel = np.linalg.inv(T_curr) @ T_prev

    # Update each previous track’s box by transforming its center using T_rel.
    # (Here we assume the box center is stored in the first three elements.)
    for track in tracker['track_states']:
        center = track['box'][:3]  # current center in previous sensor frame.
        center_hom = np.ones(4)
        center_hom[:3] = center
        new_center = T_rel @ center_hom  # new center in current sensor frame.
        track['box'][:3] = new_center[:3]
    
    # ---- Predict New Box Positions Using the Motion Model ----
    predicted_boxes = []
    for track in tracker['track_states']:
        current_box = track['box']
        velocity = track.get('velocity', None)
        
        # Simple motion prediction in sensor coordinates.
        if motion_model == 'linear':
            if velocity is not None:
                predicted_box = current_box.copy()
                predicted_box[:3] += velocity * time_step
            else:
                predicted_box = current_box.copy()
        elif motion_model == 'constant_velocity':
            vel = track.get('velocity', np.zeros(3))
            predicted_box = current_box.copy()
            predicted_box[:3] += vel * time_step
        elif motion_model == 'kalman':
            # Placeholder for Kalman filter prediction.
            predicted_box = current_box.copy()
        else:
            predicted_box = current_box.copy()
        
        predicted_boxes.append(predicted_box)
    
    # ---- Data Association ----
    matched_pairs = match_detections_to_tracks(
        current_boxes=current_boxes_np,
        predicted_boxes=np.array(predicted_boxes),
        iou_threshold=0.1
    )
    
    # Update tracks and handle new detections.
    updated_tracks = []
    matched_track_indices = set()
    matched_det_indices = set()
    
    # Generate new track IDs.
    new_id = max(tracker['track_ids']) + 1 if tracker['track_ids'] else 0
    
    nCars = 0
    for det_idx, track_idx in matched_pairs:
        if ( current_boxes_np[det_idx][3]>3 or current_boxes_np[det_idx][4]>3 ):
            nCars += 1
        prev_track = tracker['track_states'][track_idx]
        current_box = current_boxes_np[det_idx]
        
        # Calculate velocity based on the change in the box center.
        if prev_track['box'] is not None:
            velocity = np.array([current_velocities[det_idx][0], current_velocities[det_idx][1], 0])  # Assuming z velocity = 0
        else:
            velocity = np.zeros(3)
        
        updated_track = {
            'box': current_box.copy(),
            'velocity': velocity,
            'track_id': tracker['track_ids'][track_idx]
        }
        # (Optional) Place for a Kalman update.
        if motion_model == 'kalman':
            pass
        
        updated_tracks.append(updated_track)
        matched_track_indices.add(track_idx)
        matched_det_indices.add(det_idx)

    print(f"Number of matched cars: {nCars}")
    
    # Create new tracks for unmatched detections.
    unmatched_det_indices = set(range(len(current_boxes_np))) - matched_det_indices
    for det_idx in unmatched_det_indices:
        current_box = current_boxes_np[det_idx]
        updated_track = {
            'box': current_box.copy(),
            'velocity': np.array([current_velocities[det_idx][0], current_velocities[det_idx][1], 0]),
            'track_id': new_id
        }
        updated_tracks.append(updated_track)
        new_id += 1
    
    # Convert updated tracks to a detections tensor.
    updated_detections_np = np.array([track['box'] for track in updated_tracks])
    updated_detections = torch.tensor(updated_detections_np, device=current_boxes.device, dtype=current_boxes.dtype)
    
    # Update tracker state.
    updated_tracker = {
        'track_ids': [track['track_id'] for track in updated_tracks],
        'track_states': updated_tracks,
        'motion_model': motion_model,
        'last_timestamp': time_step,
        'lidar_pose': T_curr  # Save the current pose for the next frame.
    }
    
    return updated_detections, updated_tracker