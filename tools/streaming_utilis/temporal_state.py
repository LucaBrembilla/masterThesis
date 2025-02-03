import numpy as np
import torch

from tools.streaming_utilis.match_detections_to_tracks import match_detections_to_tracks

def update_temporal_state(pred_dicts, tracker=None, motion_model='linear', time_step=0.1):
    """
    Maintain temporal state using motion model predictions
    Args:
        pred_dicts: List of prediction dictionaries from current frame
        tracker: Previous tracker state (None for first frame)
        motion_model: Type of motion prediction ('linear', 'kalman', 'constant_velocity')
        time_step: Time between frames (seconds)
    Returns:
        updated_detections: List of detections with motion-corrected boxes
        updated_tracker: Updated tracking state
    """
    current_boxes = pred_dicts[0]['pred_boxes']  # (N, 7) tensor [x,y,z,dx,dy,dz,heading]
    current_boxes_np = current_boxes.cpu().numpy()

    # Limit to 7 columns (x, y, z, dx, dy, dz, heading). For nuScenes
    if current_boxes_np[0].shape[0] > 7:
        current_boxes_np = current_boxes_np[:, :7]
    
    if tracker is None:
        # Initialize tracker for first frame
        tracker = {
            'track_ids': np.arange(len(current_boxes_np)).tolist(),
            'track_states': [{'box': box[:7], 'velocity': None} for box in current_boxes_np],
            'motion_model': motion_model,
            'last_timestamp': time_step
        }
        return current_boxes, tracker
    
    # Predict next positions using motion model
    predicted_boxes = []
    for track in tracker['track_states']:
        current_box = track['box']
        velocity = track.get('velocity', None)
        
        if motion_model == 'linear':
            if velocity is not None:
                predicted_box = current_box.copy()
                predicted_box[:3] += velocity * time_step
            else:
                predicted_box = current_box.copy()
        elif motion_model == 'constant_velocity':
            # Assume constant velocity, initialize with zeros if not present
            vel = track.get('velocity', np.zeros(3))
            predicted_box = current_box.copy()
            predicted_box[:3] += vel * time_step
        elif motion_model == 'kalman':
            # Placeholder for Kalman filter prediction
            predicted_box = current_box.copy()
        else:
            predicted_box = current_box.copy()
        
        predicted_boxes.append(predicted_box)
    
    #print("Current boxes: ", current_boxes_np)
    #print("Predicted boxes from state: ", np.array(predicted_boxes))

    # Data association between predicted_boxes and current detections
    matched_pairs = match_detections_to_tracks(
        current_boxes=current_boxes_np,
        predicted_boxes=np.array(predicted_boxes),
        iou_threshold=0.3
    )
    
    #print("Matched pairs: ", matched_pairs)

    # Update tracks and handle new detections
    updated_tracks = []
    matched_track_indices = set()
    matched_det_indices = set()
    
    # Generate new track IDs
    new_id = max(tracker['track_ids']) + 1 if tracker['track_ids'] else 0
    
    for det_idx, track_idx in matched_pairs:
        prev_track = tracker['track_states'][track_idx]
        current_box = current_boxes_np[det_idx]
        
        # Calculate velocity
        if prev_track['box'] is not None:
            velocity = (current_box[:3] - prev_track['box'][:3]) / time_step
        else:
            velocity = np.zeros(3)
        
        updated_track = {
            'box': current_box,
            'velocity': velocity,
            'track_id': tracker['track_ids'][track_idx]
        }
        # Placeholder for Kalman update
        if motion_model == 'kalman':
            pass
        
        updated_tracks.append(updated_track)
        matched_track_indices.add(track_idx)
        matched_det_indices.add(det_idx)
    
    # Handle unmatched detections (new tracks)
    unmatched_det_indices = set(range(len(current_boxes_np))) - matched_det_indices
    for det_idx in unmatched_det_indices:
        current_box = current_boxes_np[det_idx]
        updated_track = {
            'box': current_box,
            'velocity': None,
            'track_id': new_id
        }
        updated_tracks.append(updated_track)
        new_id += 1
    
    # Convert updated tracks to tensor
    updated_detections_np = np.array([track['box'] for track in updated_tracks])
    updated_detections = torch.tensor(updated_detections_np, device=current_boxes.device, dtype=current_boxes.dtype)
    
    # Prepare updated tracker
    updated_tracker = {
        'track_ids': [track['track_id'] for track in updated_tracks],
        'track_states': updated_tracks,
        'motion_model': motion_model,
        'last_timestamp': time_step
    }
    
    return updated_detections, updated_tracker