import mediapipe as mp
import numpy as np
import math

class KeypointPipeline:
    def __init__(self, min_roi_size=150, roi_padding_ratio=1.0):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.min_roi_size = min_roi_size
        self.roi_padding_ratio = roi_padding_ratio

    def _get_pixel_coords(self, landmark, img_shape):
        h, w, _ = img_shape
        if landmark.visibility < 0.5:
            return None
        return int(landmark.x * w), int(landmark.y * h)

    def _get_dynamic_roi(self, wrist, left_shoulder, right_shoulder, img_shape):
        """Calculates a square ROI around the wrist based on shoulder distance."""
        h, w, _ = img_shape
        
        if left_shoulder and right_shoulder:
            shoulder_dist = math.dist(left_shoulder, right_shoulder)
        else:
            shoulder_dist = self.min_roi_size * 2
            
        roi_size = max(self.min_size, int(shoulder_dist * self.roi_padding_ratio))
        half_size = roi_size // 2
        
        x_min = max(0, wrist[0] - half_size)
        y_min = max(0, wrist[1] - half_size)
        x_max = min(w, wrist[0] + half_size)
        y_max = min(h, wrist[1] + half_size)
        
        return x_min, y_min, x_max, y_max

    def process_frame(self, frame_rgb):
        """
        Extracts hand keypoints from an RGB frame.
        Returns a dictionary of hands {'left': keypoints, 'right': keypoints} and confidences.
        """
        img_h, img_w, _ = frame_rgb.shape
        
        # 1. Pose Estimation
        pose_results = self.pose.process(frame_rgb)
        if not pose_results.pose_landmarks:
            return {}, {}
            
        landmarks = pose_results.pose_landmarks.landmark
        
        left_shoulder = self._get_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value], frame_rgb.shape)
        right_shoulder = self._get_pixel_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value], frame_rgb.shape)
        
        wrist_coords = {
            'left': self._get_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value], frame_rgb.shape),
            'right': self._get_pixel_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value], frame_rgb.shape)
        }
        
        hands_kps = {}
        hands_conf = {}
        
        # 2. ROI Extraction & Hand Tracking
        for hand_label, wrist in wrist_coords.items():
            if not wrist:
                continue
                
            # Compute ROI
            x_min, y_min, x_max, y_max = self._get_dynamic_roi(wrist, left_shoulder, right_shoulder, frame_rgb.shape)
            
            # Crop
            roi_img = frame_rgb[y_min:y_max, x_min:x_max]
            if roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
                continue
                
            # Run Hands on ROI
            hand_results = self.hands.process(roi_img)
            
            if hand_results.multi_hand_landmarks:
                # We assume the first detected hand in this specific ROI is the target hand
                hand_lms = hand_results.multi_hand_landmarks[0]
                confidence = hand_results.multi_handedness[0].classification[0].score
                
                roi_h, roi_w, _ = roi_img.shape
                
                # Convert back to global coordinates
                global_kps = []
                for lm in hand_lms.landmark:
                    global_x = (lm.x * roi_w) + x_min
                    global_y = (lm.y * roi_h) + y_min
                    global_kps.append([global_x, global_y, lm.z])
                    
                hands_kps[hand_label] = global_kps
                hands_conf[hand_label] = confidence

        return hands_kps, hands_conf

    def close(self):
        self.pose.close()
        self.hands.close()
