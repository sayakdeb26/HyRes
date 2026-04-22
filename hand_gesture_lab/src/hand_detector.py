import mediapipe as mp
from . import config

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def process(self, roi_image, bbox):
        """
        Runs hand detection on the ROI image.
        bbox is [x_min, y_min, x_max, y_max] from the full frame.
        Returns global keypoints in absolute pixel coordinates.
        """
        results = self.hands.process(roi_image)
        
        if not results.multi_hand_landmarks:
            return None, 0.0

        # We only process the first detected hand per ROI
        hand_landmarks = results.multi_hand_landmarks[0]
        confidence = results.multi_handedness[0].classification[0].score

        roi_h, roi_w, _ = roi_image.shape
        x_min, y_min, _, _ = bbox

        global_keypoints = []
        for lm in hand_landmarks.landmark:
            # Convert normalized local coords to absolute local coords
            local_x = lm.x * roi_w
            local_y = lm.y * roi_h
            
            # Convert absolute local coords to absolute global coords
            global_x = local_x + x_min
            global_y = local_y + y_min
            
            global_keypoints.append([global_x, global_y, lm.z]) # Keeping z as relative

        return global_keypoints, confidence

    def close(self):
        self.hands.close()
