import numpy as np

class Validator:
    def __init__(self, min_conf=0.5, max_jump_ratio=0.3, ema_alpha=0.5):
        self.min_conf = min_conf
        self.max_jump_ratio = max_jump_ratio
        self.ema_alpha = ema_alpha
        
        self.prev_keypoints = None

    def validate_and_smooth(self, keypoints, confidence, img_shape):
        """
        Validates keypoints and applies EMA smoothing.
        Returns the smoothed keypoints or None if invalid.
        """
        if confidence < self.min_conf:
            self.prev_keypoints = None
            return None
            
        curr_kp = np.array(keypoints)
        h, w, _ = img_shape
        
        # Anatomy/Bounds check (wrist is inside the frame)
        wrist = curr_kp[0]
        if not (0 <= wrist[0] <= w and 0 <= wrist[1] <= h):
            self.prev_keypoints = None
            return None
            
        if self.prev_keypoints is not None:
            # Jump check (prevent swapping hands or false positives)
            prev_wrist = self.prev_keypoints[0]
            jump_dist = np.linalg.norm(wrist[:2] - prev_wrist[:2])
            frame_diag = np.sqrt(h**2 + w**2)
            
            if jump_dist / frame_diag > self.max_jump_ratio:
                # Sudden jump, invalidate sequence smoothing
                self.prev_keypoints = None
                return None
                
            # EMA Smoothing
            smoothed_kp = self.ema_alpha * curr_kp + (1 - self.ema_alpha) * self.prev_keypoints
        else:
            smoothed_kp = curr_kp
            
        self.prev_keypoints = smoothed_kp
        return smoothed_kp

    def reset(self):
        self.prev_keypoints = None
