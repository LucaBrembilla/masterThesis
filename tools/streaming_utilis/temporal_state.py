import numpy as np

from match_detections_to_tracks import match_detections_to_tracks


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
    current_boxes = [pred['pred_boxes'] for pred in pred_dicts]  # (N, 7) [x,y,z,dx,dy,dz,heading]
    current_scores = [pred['pred_scores'] for pred in pred_dicts]
    
    if tracker is None:
        # Initialize tracker for first frame
        tracker = {
            'track_ids': np.arange(len(current_boxes)),
            'track_states': current_boxes,
            'motion_model': motion_model,
            'last_timestamp': time_step
        }
        return current_boxes, tracker
    
    # Motion model implementation
    predicted_boxes = []
    for track in tracker['track_states']:
        if motion_model == 'linear':
            # Simple linear extrapolation (x = x0 + v*Δt)
            # Assume velocity is estimated from previous frames
            if 'velocity' not in track:
                predicted = track  # First frame, no motion
            else:
                predicted = track.copy()
                predicted[:3] += track['velocity'] * time_step
                
        elif motion_model == 'constant_velocity':
            # Classic constant velocity model
            predicted = track.copy()
            predicted[:3] += track.get('velocity', [0,0,0]) * time_step
            # Add noise/uncertainty growth
            # TODO
            
        elif motion_model == 'kalman':
            # Kalman filter prediction step
            # (Would need full Kalman implementation)
            # TODO
            predicted = track['kf'].predict()
            
        predicted_boxes.append(predicted)
    
    # Data association (simple IoU matching)
    matched_pairs = match_detections_to_tracks(
        current_boxes, 
        predicted_boxes,
        iou_threshold=0.3
    )
    
    # Update tracker state
    updated_tracks = []
    new_id = tracker['track_ids'][-1] + 1 if len(tracker['track_ids']) > 0 else 0
    
    for det_idx, track_idx in matched_pairs:
        # Update existing track
        updated_track = {
            'box': current_boxes[det_idx],
            'velocity': (current_boxes[det_idx][:3] - tracker['track_states'][track_idx][:3]) / time_step,
            'timestamp': time_step,
            'track_id': tracker['track_ids'][track_idx]
        }
        if motion_model == 'kalman':
            updated_track['kf'].update(current_boxes[det_idx])
            
        updated_tracks.append(updated_track)
    
    # Handle new detections
    unmatched_detections = set(range(len(current_boxes))) - {p[0] for p in matched_pairs}
    for det_idx in unmatched_detections:
        updated_tracks.append({
            'box': current_boxes[det_idx],
            # 'velocity': np.zeros(3),  # Unknown initial velocity
            'timestamp': time_step,
            'track_id': new_id
        })
        new_id += 1
        
    # Format output
    updated_detections = [track['box'] for track in updated_tracks]
    
    return updated_detections, {
        'track_ids': [t['track_id'] for t in updated_tracks],
        'track_states': updated_tracks,
        'motion_model': motion_model,
        'last_timestamp': time_step
    }

