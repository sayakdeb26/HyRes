import numpy as np
from . import config

class Validator:
    def __init__(self, ema_alpha=config.EMA_ALPHA, max_jump=config.MAX_JUMP_THRESHOLD, min_conf=config.MIN_HAND_CONFIDENCE):
        self.ema_alpha = ema_alpha
        self.max_jump = max_jump
        self.min_conf = min_conf
        self.prev_keypoints = {} # Store by hand ('left' or 'right')

    def validate_and_smooth(self, keypoints, confidence, hand_label, frame_shape):
        """
        Validates the keypoints and applies EMA smoothing.
        Returns smoothed keypoints if valid, else None.
        """
        if confidence < self.min_conf:
            self.prev_keypoints[hand_label] = None
            return None

        # Convert to numpy array
        curr_kp = np.array(keypoints)

        # Basic bounds check (are wrist coords inside frame?)
        h, w, _ = frame_shape
        wrist_x, wrist_y, _ = curr_kp[0]
        if not (0 <= wrist_x <= w and 0 <= wrist_y <= h):
            self.prev_keypoints[hand_label] = None
            return None

        # Temporal smoothing and jump check
        if hand_label in self.prev_keypoints and self.prev_keypoints[hand_label] is not None:
            prev_kp = self.prev_keypoints[hand_label]
            
            # Check for sudden jumps (using wrist position relative to frame size)
            wrist_jump = np.linalg.norm(curr_kp[0][:2] - prev_kp[0][:2])
            frame_diag = np.sqrt(h**2 + w**2)
            
            if wrist_jump / frame_diag > self.max_jump:
                # Too big of a jump, likely a false positive / misclassification
                self.prev_keypoints[hand_label] = None
                return None
            
            # Apply EMA
            smoothed_kp = self.ema_alpha * curr_kp + (1 - self.ema_alpha) * prev_kp
        else:
            smoothed_kp = curr_kp

        self.prev_keypoints[hand_label] = smoothed_kp
        return smoothed_kp.tolist()

    def reset(self):
        self.prev_keypoints = {}
