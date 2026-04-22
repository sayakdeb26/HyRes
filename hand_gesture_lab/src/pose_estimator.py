import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, image):
        """
        Processes an image and returns pose landmarks.
        Image must be in RGB format.
        """
        results = self.pose.process(image)
        return results

    def extract_key_points(self, results, image_shape):
        """
        Extracts wrist, elbow, and shoulder coordinates in pixels for both arms.
        Returns a dictionary with left and right arm keypoints.
        """
        if not results.pose_landmarks:
            return None

        h, w, _ = image_shape
        landmarks = results.pose_landmarks.landmark

        def get_pixel_coords(landmark_idx):
            lm = landmarks[landmark_idx]
            if lm.visibility < 0.5:
                return None
            return int(lm.x * w), int(lm.y * h)

        keypoints = {
            'left': {
                'wrist': get_pixel_coords(self.mp_pose.PoseLandmark.LEFT_WRIST.value),
                'elbow': get_pixel_coords(self.mp_pose.PoseLandmark.LEFT_ELBOW.value),
                'shoulder': get_pixel_coords(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            },
            'right': {
                'wrist': get_pixel_coords(self.mp_pose.PoseLandmark.RIGHT_WRIST.value),
                'elbow': get_pixel_coords(self.mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                'shoulder': get_pixel_coords(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            }
        }
        return keypoints

    def close(self):
        self.pose.close()
