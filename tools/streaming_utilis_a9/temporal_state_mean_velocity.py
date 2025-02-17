import numpy as np
import torch

from tools.streaming_utilis.match_detections_to_tracks import match_detections_to_tracks

def update_temporal_state(pred_dicts, tracker=None, motion_model='linear', time_step=0.1):
    """
    Maintain temporal state using motion model predictions.
    Args:
        pred_dicts: List of prediction dictionaries from current frame.
                   Each dictionary should have a 'pred_boxes' key containing a (N,7) tensor
                   [x, y, z, dx, dy, dz, heading].
        tracker: Previous tracker state (None for first frame).
        motion_model: Type of motion prediction ('linear', 'kalman', 'constant_velocity').
        time_step: Time between frames (seconds).
    Returns:
        updated_detections: Updated detections (as a torch tensor).
        updated_tracker: Updated tracking state.
    """
    current_boxes = pred_dicts[0]['pred_boxes']  # (N, 7) tensor
    current_boxes_np = current_boxes.cpu().numpy()
    mean_speed = 36.1  # mean velocity in m/s

    # Limit to 7 columns (x, y, z, dx, dy, dz, heading)
    if current_boxes_np[0].shape[0] > 7:
        current_boxes_np = current_boxes_np[:, :7]
    
    # If tracker is None, initialize it but do not return immediately.
    # This ensures that even on the first frame the boxes get updated.
    if tracker is None:
        tracker = {
            'track_ids': np.arange(len(current_boxes_np)).tolist(),
            'track_states': [
                {
                    'box': box.copy(),
                    'velocity': np.array([
                        mean_speed * np.cos(box[6]),
                        mean_speed * np.sin(box[6]),
                        0.0
                    ])
                }
                for box in current_boxes_np
            ],
            'motion_model': motion_model,
            'last_timestamp': time_step
        }
        return current_boxes, tracker
    
    # Predict next positions using the motion model
    predicted_boxes = []
    for track in tracker['track_states']:
        predicted_box = track['box']
        predicted_boxes.append(predicted_box)
    
    # Data association between predicted boxes and current detections
    matched_pairs = match_detections_to_tracks(
        current_boxes=current_boxes_np,
        predicted_boxes=np.array(predicted_boxes),
        iou_threshold=0.1
    )
    
    # Update tracks and handle new detections
    updated_tracks = []
    matched_det_indices = set()
    
    # Use existing track IDs; start new IDs from the max + 1.
    new_id = max(tracker['track_ids']) + 1 if tracker['track_ids'] else 0
    
    for det_idx, track_idx in matched_pairs:
        prev_track = tracker['track_states'][track_idx]
        prev_track['box'][:3] -= prev_track['velocity'] * time_step 
        current_box = current_boxes_np[det_idx]
        
        # Calculate new velocity based on displacement
        if prev_track['box'] is not None:
            velocity = (current_box[:3] - prev_track['box'][:3]) / time_step
        else:
            velocity = np.array([
                        mean_speed * np.cos(current_box[6]),
                        mean_speed * np.sin(current_box[6]),
                        0.0
                    ])
        
        updated_track = {
            'box': current_box,
            'velocity': velocity,
            'track_id': tracker['track_ids'][track_idx]
        }
        updated_tracks.append(updated_track)
        matched_det_indices.add(det_idx)
    
    # Handle unmatched detections (new tracks)
    unmatched_det_indices = set(range(len(current_boxes_np))) - matched_det_indices
    for det_idx in unmatched_det_indices:
        current_box = current_boxes_np[det_idx]
        updated_track = {
            'box': current_box,
            'velocity': np.array([
                mean_speed * np.cos(current_box[6]),
                mean_speed * np.sin(current_box[6]),
                0.0
            ]),
            'track_id': new_id
        }
        updated_tracks.append(updated_track)
        new_id += 1
    
    # Convert updated track boxes to tensor format for detections
    updated_detections_np = np.array([track['box'] for track in updated_tracks])
    updated_detections = torch.tensor(updated_detections_np, device=current_boxes.device, dtype=current_boxes.dtype)
    
    # Prepare the updated tracker state
    updated_tracker = {
        'track_ids': [track['track_id'] for track in updated_tracks],
        'track_states': updated_tracks,
        'motion_model': motion_model,
        'last_timestamp': time_step
    }
    
    return updated_detections, updated_tracker

def predict_from_state(tracker, time_step):
    if tracker is None:
        return None

    for track in tracker['track_states']:
        track['box'][0] += track['velocity'][0] * time_step
        track['box'][1] += track['velocity'][1] * time_step
        track['box'][2] += track['velocity'][2] * time_step

    return tracker

