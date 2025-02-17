import os
import json
import numpy as np

def load_frame_labels(folder):
    """
    Loads CAR labels from all JSON files in the folder.
    Each file is assumed to represent one frame.
    Returns a sorted list of frames, where each frame is a list of label dictionaries.
    Each label dict contains: timestamp, id, category, dims, loc, orient.
    """
    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    frames = []
    for fname in sorted(files):
        path = os.path.join(folder, fname)
        with open(path, 'r') as fp:
            data = json.load(fp)
        # Compute timestamp in seconds
        timestamp = data['timestamp_secs'] + data['timestamp_nsecs'] / 1e9
        frame_labels = []
        for label in data.get('labels', []):
            # We process only CAR objects (modify as needed)
            if label.get('category') != "CAR":
                continue
            dims = label['box3d']['dimension']  # length, width, height
            loc = label['box3d']['location']    # x, y, z
            orient = label['box3d']['orientation']  # rotationYaw, rotationPitch, rotationRoll
            frame_labels.append({
                'timestamp': timestamp,
                'id': label['id'],
                'category': label['category'],
                'dims': dims,
                'loc': loc,
                'orient': orient
            })
        # Only add frames that contain at least one CAR label
        if frame_labels:
            frames.append(frame_labels)
    # Sort frames by timestamp (assume each frame has at least one label)
    frames.sort(key=lambda f: f[0]['timestamp'])
    return frames

def match_labels(frame1, frame2, tol=0.2):
    """
    For each label in frame1, find the best matching label in frame2 based on:
      - Same category ("CAR")
      - Similar dimensions (difference less than tol in each dimension)
    Returns a list of tuples: (label_from_frame1, matched_label_from_frame2, euclidean_distance)
    """
    matches = []
    used_indices = set()
    for label1 in frame1:
        best_match = None
        best_dist = float('inf')
        for idx, label2 in enumerate(frame2):
            if idx in used_indices:
                continue
            if label2['category'] != label1['category']:
                continue
            d1 = label1['dims']
            d2 = label2['dims']
            # Check if dimensions are similar (within tol)
            if (abs(d1['length'] - d2['length']) > tol or
                abs(d1['width']  - d2['width'])  > tol or
                abs(d1['height'] - d2['height']) > tol):
                continue
            # Compute 2D distance (x-y) between the locations
            x1, y1 = label1['loc']['x'], label1['loc']['y']
            x2, y2 = label2['loc']['x'], label2['loc']['y']
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < best_dist:
                best_dist = dist
                best_match = label2
        if best_match is not None:
            # Mark this label in frame2 as used so it isn’t matched twice.
            for idx, candidate in enumerate(frame2):
                if candidate['id'] == best_match['id']:
                    used_indices.add(idx)
                    break
            matches.append((label1, best_match, best_dist))
    return matches

def compute_velocities(matches):
    """
    Given a list of matched label pairs from consecutive frames,
    compute the velocity (in the x-y plane) for each match.
    Also determine the orientation group based on rotationYaw from the first label.
    
    Grouping rule: 
      - If normalized yaw < π/2 or > 3π/2, assign group 0.
      - Otherwise, assign group 1.
    
    Returns a list of dicts with keys: speed, group, vx, vy, direction.
    The `direction` is computed (in radians) from the velocity vector using np.arctan2.
    """
    velocities = []
    for label1, label2, _ in matches:
        t1 = label1['timestamp']
        t2 = label2['timestamp']
        dt = t2 - t1
        if dt <= 0:
            continue
        x1, y1 = label1['loc']['x'], label1['loc']['y']
        x2, y2 = label2['loc']['x'], label2['loc']['y']
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        speed = np.sqrt(vx**2 + vy**2)
        # Compute the direction (angle in radians) of the velocity vector
        direction = np.arctan2(vy, vx)
        # Determine orientation group using the rotationYaw from label1.
        # Normalize yaw to [0, 2π)
        yaw = label1['orient']['rotationYaw'] % (2 * np.pi)
        group = 0 if (yaw < np.pi/2 or yaw > 3*np.pi/2) else 1
        velocities.append({'speed': speed, 'group': group, 'vx': vx, 'vy': vy, 'direction': direction})
    return velocities

def mean_angle(angles):
    """
    Compute the mean angle (in radians) from a list of angles using vector averaging.
    """
    if len(angles) == 0:
        return 0
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)

def main():
    folder = "/home/brembilla/exp/private_datasets/providentia/_labels"  # <-- update to the folder containing your JSON label files
    frames = load_frame_labels(folder)
    all_velocities = []

    # Loop over consecutive frame pairs to match labels and compute velocities.
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i+1]
        matches = match_labels(frame1, frame2, tol=0.01)
        velocities = compute_velocities(matches)
        all_velocities.extend(velocities)

    # Separate speeds and directions by group.
    group0_speeds = [v['speed'] for v in all_velocities if v['group'] == 0]
    group1_speeds = [v['speed'] for v in all_velocities if v['group'] == 1]
    
    group0_angles = [v['direction'] for v in all_velocities if v['group'] == 0]
    group1_angles = [v['direction'] for v in all_velocities if v['group'] == 1]

    mean_speed_group0 = np.mean(group0_speeds) if group0_speeds else 0
    mean_speed_group1 = np.mean(group1_speeds) if group1_speeds else 0

    # Compute the mean velocity direction (in radians) for each group.
    mean_direction_group0 = mean_angle(group0_angles) if group0_angles else 0
    mean_direction_group1 = mean_angle(group1_angles) if group1_angles else 0

    # Convert mean directions to degrees for easier interpretation.
    mean_direction_group0_deg = np.degrees(mean_direction_group0)
    mean_direction_group1_deg = np.degrees(mean_direction_group1)

    print("Mean speed for Group 0 (orientation near 0): {:.4f} m/s".format(mean_speed_group0))
    print("Mean velocity direction for Group 0: {:.2f}°".format(mean_direction_group0_deg))
    print("Mean speed for Group 1 (orientation near π): {:.4f} m/s".format(mean_speed_group1))
    print("Mean velocity direction for Group 1: {:.2f}°".format(mean_direction_group1_deg))

if __name__ == '__main__':
    main()
