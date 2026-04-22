import numpy as np
import cv2
from . import config

class ROIExtractor:
    def __init__(self, padding_ratio=config.ROI_PADDING_RATIO, min_size=config.MIN_ROI_SIZE):
        self.padding_ratio = padding_ratio
        self.min_size = min_size

    def calculate_shoulder_distance(self, keypoints):
        """Calculates pixel distance between left and right shoulders."""
        ls = keypoints['left']['shoulder']
        rs = keypoints['right']['shoulder']
        if ls and rs:
            return np.linalg.norm(np.array(ls) - np.array(rs))
        return self.min_size * 2 # fallback

    def get_roi(self, image_shape, wrist_coords, shoulder_distance):
        """
        Calculates bounding box [x_min, y_min, x_max, y_max] around the wrist.
        """
        if not wrist_coords:
            return None
        
        h, w, _ = image_shape
        roi_size = max(self.min_size, int(shoulder_distance * self.padding_ratio))
        half_size = roi_size // 2

        x_min = max(0, wrist_coords[0] - half_size)
        y_min = max(0, wrist_coords[1] - half_size)
        x_max = min(w, wrist_coords[0] + half_size)
        y_max = min(h, wrist_coords[1] + half_size)

        return [x_min, y_min, x_max, y_max]
        
    def extract(self, image, keypoints):
        """
        Extracts the ROIs for both hands.
        Returns a dictionary with 'left' and 'right' ROI bounding boxes and cropped images.
        """
        if not keypoints:
            return None

        shoulder_dist = self.calculate_shoulder_distance(keypoints)
        
        rois = {}
        for hand in ['left', 'right']:
            wrist = keypoints[hand]['wrist']
            bbox = self.get_roi(image.shape, wrist, shoulder_dist)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                crop = image[y_min:y_max, x_min:x_max]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    rois[hand] = {
                        'bbox': bbox,
                        'image': crop
                    }
        return rois
