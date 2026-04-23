import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os

class KeypointPipeline:
    def __init__(self, min_roi_size=150, roi_padding_ratio=1.2):
        # 1. Pose Estimation (GPU)
        pose_model_path = '/home/sayak/amr_gesture_ws/models/mediapipe/pose_landmarker.task'
        if not os.path.exists(pose_model_path):
            self.use_pose_tasks = False
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.use_pose_tasks = True
            base_options = python.BaseOptions(
                model_asset_path=pose_model_path,
                delegate=python.BaseOptions.Delegate.GPU
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose = vision.PoseLandmarker.create_from_options(options)
            
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        
        # 2. Hand Landmarker (GPU)
        model_path = '/home/sayak/amr_gesture_ws/models/mediapipe/hand_landmarker.task'
        if not os.path.exists(model_path):
            self.use_tasks = False
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.use_tasks = True
            base_options = python.BaseOptions(
                model_asset_path=model_path,
                delegate=python.BaseOptions.Delegate.GPU
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.hands = vision.HandLandmarker.create_from_options(options)
            
        self.min_roi_size = min_roi_size
        self.roi_padding_ratio = roi_padding_ratio
        
        # Temporal carry-forward buffers
        self.last_valid_hands = {'left': None, 'right': None}
        self.last_valid_conf = {'left': 0.0, 'right': 0.0}

    def _get_pixel_coords(self, landmark, img_shape):
        h, w, _ = img_shape
        if getattr(landmark, 'visibility', 1.0) < 0.3:
            return None
        return int(landmark.x * w), int(landmark.y * h)

    def _get_dynamic_roi(self, wrist, elbow, img_shape):
        """Calculates a square ROI around the midpoint of wrist and elbow."""
        h, w, _ = img_shape
        
        if wrist and elbow:
            center_x = (wrist[0] + elbow[0]) // 2
            center_y = (wrist[1] + elbow[1]) // 2
            dist = math.dist(wrist, elbow)
        elif wrist:
            center_x, center_y = wrist
            dist = self.min_roi_size
        else:
            return None
            
        roi_size = max(self.min_roi_size, int(dist * self.roi_padding_ratio))
        half_size = roi_size // 2
        
        x_min = max(0, center_x - half_size)
        y_min = max(0, center_y - half_size)
        x_max = min(w, center_x + half_size)
        y_max = min(h, center_y + half_size)
        
        if x_max <= x_min or y_max <= y_min:
            return None
            
        return x_min, y_min, x_max, y_max

    def process_frame(self, frame_rgb):
        """
        Extracts keypoints using GPU Pose + GPU Hand Refinement.
        Returns: pose_kps, hands_kps, hands_conf
        """
        img_h, img_w, _ = frame_rgb.shape
        
        # 1. Global Context (Pose)
        pose_kps = None
        landmarks = None
        
        if self.use_pose_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
            pose_results = self.pose.detect(mp_image)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks[0]
                pose_kps = []
                for lm in landmarks:
                    pose_kps.append([lm.x * img_w, lm.y * img_h, lm.z, lm.visibility])
        else:
            pose_results = self.pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                pose_kps = []
                for lm in landmarks:
                    pose_kps.append([lm.x * img_w, lm.y * img_h, lm.z, lm.visibility])
                
        if not pose_kps:
            return None, {}, {}
            
        # Get joints for ROI
        left_wrist = self._get_pixel_coords(landmarks[self.LEFT_WRIST], frame_rgb.shape)
        right_wrist = self._get_pixel_coords(landmarks[self.RIGHT_WRIST], frame_rgb.shape)
        left_elbow = self._get_pixel_coords(landmarks[self.LEFT_ELBOW], frame_rgb.shape)
        right_elbow = self._get_pixel_coords(landmarks[self.RIGHT_ELBOW], frame_rgb.shape)
        
        roi_targets = {
            'left': (left_wrist, left_elbow),
            'right': (right_wrist, right_elbow)
        }
        
        hands_kps = {}
        hands_conf = {}
        
        # 2. Local Hand Refinement
        for hand_label, (wrist, elbow) in roi_targets.items():
            refined_kps = None
            conf = 0.0
            
            roi_coords = self._get_dynamic_roi(wrist, elbow, frame_rgb.shape)
            
            if roi_coords:
                x_min, y_min, x_max, y_max = roi_coords
                roi_img = frame_rgb[y_min:y_max, x_min:x_max]
                
                if roi_img.shape[0] > 0 and roi_img.shape[1] > 0:
                    # Run HandLandmarker on ROI
                    if self.use_tasks:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(roi_img))
                        hand_results = self.hands.detect(mp_image)
                        
                        if hand_results.hand_landmarks:
                            hand_lms = hand_results.hand_landmarks[0] # Take most confident hand in ROI
                            conf = hand_results.handedness[0][0].score
                            roi_h, roi_w, _ = roi_img.shape
                            
                            refined_kps = []
                            for lm in hand_lms:
                                global_x = (lm.x * roi_w) + x_min
                                global_y = (lm.y * roi_h) + y_min
                                refined_kps.append([global_x, global_y, lm.z])
                    else:
                        hand_results = self.hands.process(roi_img)
                        if hand_results.multi_hand_landmarks:
                            hand_lms = hand_results.multi_hand_landmarks[0]
                            conf = hand_results.multi_handedness[0].classification[0].score
                            roi_h, roi_w, _ = roi_img.shape
                            
                            refined_kps = []
                            for lm in hand_lms.landmark:
                                global_x = (lm.x * roi_w) + x_min
                                global_y = (lm.y * roi_h) + y_min
                                refined_kps.append([global_x, global_y, lm.z])
                                
            # 3. Fallback logic
            if refined_kps is not None:
                hands_kps[hand_label] = refined_kps
                hands_conf[hand_label] = conf
                self.last_valid_hands[hand_label] = refined_kps
                self.last_valid_conf[hand_label] = conf
            elif self.last_valid_hands[hand_label] is not None:
                # Temporal carry-forward only
                hands_kps[hand_label] = self.last_valid_hands[hand_label]
                hands_conf[hand_label] = self.last_valid_conf[hand_label] * 0.9 # Decay confidence
                
        return pose_kps, hands_kps, hands_conf

    def close(self):
        if hasattr(self.pose, 'close'):
            self.pose.close()
        if hasattr(self.hands, 'close'):
            self.hands.close()
