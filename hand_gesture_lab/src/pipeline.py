import cv2
import time
from .pose_estimator import PoseEstimator
from .roi_extractor import ROIExtractor
from .hand_detector import HandDetector
from .validator import Validator
from .feature_builder import FeatureBuilder
from .visualizer import Visualizer

class GesturePipeline:
    def __init__(self, inference_engine=None):
        self.pose_estimator = PoseEstimator()
        self.roi_extractor = ROIExtractor()
        self.hand_detector = HandDetector()
        self.validator = Validator()
        self.feature_builder = FeatureBuilder()
        self.visualizer = Visualizer()
        self.inference_engine = inference_engine
        
        self.prev_time = time.time()

    def process_frame(self, frame, visualize=True):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Pose Estimation
        pose_results = self.pose_estimator.process(image_rgb)
        pose_keypoints = self.pose_estimator.extract_key_points(pose_results, frame.shape)
        
        # 2. ROI Extraction
        rois = self.roi_extractor.extract(frame, pose_keypoints)
        
        predictions = {'left': "None", 'right': "None"}
        
        if rois:
            for hand_label, roi_data in rois.items():
                bbox = roi_data['bbox']
                roi_image = roi_data['image']
                
                # Convert ROI to RGB for Hand processing
                roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
                
                # 3. Hand Detection
                raw_keypoints, confidence = self.hand_detector.process(roi_rgb, bbox)
                
                # 4. Validation & Smoothing
                smoothed_keypoints = self.validator.validate_and_smooth(
                    raw_keypoints, confidence, hand_label, frame.shape
                )
                
                if visualize:
                    self.visualizer.draw_roi(frame, bbox, f"{hand_label} ROI")
                    if smoothed_keypoints:
                        self.visualizer.draw_skeleton(frame, smoothed_keypoints)
                
                # 5. Feature Building
                features = None
                if smoothed_keypoints:
                    features = self.feature_builder.build(smoothed_keypoints)
                    
                # 6. Inference
                if self.inference_engine:
                    pred = self.inference_engine.process(hand_label, features)
                    predictions[hand_label] = pred
                    
        # Update inference even if no ROI to maintain buffer zeros
        else:
            if self.inference_engine:
                predictions['left'] = self.inference_engine.process('left', None)
                predictions['right'] = self.inference_engine.process('right', None)
            
        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time
        
        if visualize:
            self.visualizer.draw_fps(frame, fps)
            for hand_label, pred in predictions.items():
                if pred != "None":
                    self.visualizer.draw_prediction(frame, pred, hand_label)
                    
        return frame, predictions

    def close(self):
        self.pose_estimator.close()
        self.hand_detector.close()
