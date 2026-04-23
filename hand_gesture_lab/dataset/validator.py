import numpy as np
from collections import deque

class Validator:
    def __init__(self, min_conf=0.5, max_jump_ratio=0.3, window_size=3):
        self.min_conf = min_conf
        self.max_jump_ratio = max_jump_ratio
        self.window_size = window_size
        
        # Buffers for temporal smoothing
        self.buffers = {
            'pose': deque(maxlen=window_size),
            'left': deque(maxlen=window_size),
            'right': deque(maxlen=window_size)
        }

    def validate_and_smooth(self, keypoints, confidence, img_shape, part_name):
        """
        Validates keypoints and applies moving average smoothing.
        part_name: 'pose', 'left', or 'right'
        """
        if keypoints is None or len(keypoints) == 0:
            return None
            
        if confidence is not None and confidence < self.min_conf:
            return None
            
        curr_kp = np.array(keypoints)
        h, w, _ = img_shape
        
        buffer = self.buffers[part_name]
        
        # Kinematic jump check
        if len(buffer) > 0:
            prev_kp = buffer[-1]
            # Use the first keypoint (e.g., wrist for hand, nose for pose) for jump check
            jump_dist = np.linalg.norm(curr_kp[0][:2] - prev_kp[0][:2])
            frame_diag = np.sqrt(h**2 + w**2)
            
            if jump_dist / frame_diag > self.max_jump_ratio:
                # Sudden jump implies detection error or huge movement. Break sequence.
                buffer.clear()
                buffer.append(curr_kp)
                return curr_kp
                
        buffer.append(curr_kp)
        
        # Apply moving average over the window
        smoothed_kp = np.mean(buffer, axis=0)
        return smoothed_kp

    def reset(self):
        for b in self.buffers.values():
            b.clear()
