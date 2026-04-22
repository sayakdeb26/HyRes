import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.pose_estimator import PoseEstimator
from src.roi_extractor import ROIExtractor
from src.hand_detector import HandDetector
from src.validator import Validator
from src.feature_builder import FeatureBuilder
from dataset.jester_loader import JesterLoader

def extract_features_from_video(video_path, pose_estimator, roi_extractor, hand_detector, validator, feature_builder):
    cap = cv2.VideoCapture(video_path)
    features_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = pose_estimator.process(image_rgb)
        pose_keypoints = pose_estimator.extract_key_points(pose_results, frame.shape)
        
        rois = roi_extractor.extract(frame, pose_keypoints)
        
        frame_features = None
        # Process the first valid hand found (or right hand preferred)
        if rois:
            hand_label = 'right' if 'right' in rois else 'left'
            roi_data = rois[hand_label]
            
            bbox = roi_data['bbox']
            roi_image = roi_data['image']
            roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
            
            raw_keypoints, confidence = hand_detector.process(roi_rgb, bbox)
            smoothed_keypoints = validator.validate_and_smooth(raw_keypoints, confidence, hand_label, frame.shape)
            
            if smoothed_keypoints:
                frame_features = feature_builder.build(smoothed_keypoints)
                
        if frame_features is None:
            frame_features = np.zeros(config.FEATURE_DIM)
            
        features_list.append(frame_features)
        
    cap.release()
    return features_list

def pad_or_truncate(features, target_length=config.SEQUENCE_LENGTH):
    """
    Ensures the sequence length is exactly target_length.
    If shorter, pads with zeros at the beginning.
    If longer, takes the middle target_length frames.
    """
    seq_len = len(features)
    
    if seq_len == 0:
        return np.zeros((target_length, config.FEATURE_DIM))
        
    if seq_len < target_length:
        pad_len = target_length - seq_len
        padding = np.zeros((pad_len, config.FEATURE_DIM))
        return np.vstack((padding, features))
    elif seq_len > target_length:
        start_idx = (seq_len - target_length) // 2
        return np.array(features[start_idx : start_idx + target_length])
    else:
        return np.array(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing video files")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to Jester CSV file")
    parser.add_argument('--output_dir', type=str, default="data/processed", help="Output directory for .npy files")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    loader = JesterLoader(args.data_dir, args.csv_path)
    items = loader.get_video_paths_and_labels()
    
    # Save label map
    loader.save_label_map(os.path.join(args.output_dir, "label_map.json"))
    
    # Initialize pipeline components
    pose_estimator = PoseEstimator()
    roi_extractor = ROIExtractor()
    hand_detector = HandDetector()
    validator = Validator()
    feature_builder = FeatureBuilder()
    
    all_features = []
    all_labels = []
    
    print(f"Processing {len(items)} videos...")
    for vid_path, label in tqdm(items):
        validator.reset() # Reset EMA between videos
        
        try:
            features = extract_features_from_video(
                vid_path, pose_estimator, roi_extractor, hand_detector, validator, feature_builder
            )
            processed_seq = pad_or_truncate(features)
            
            all_features.append(processed_seq)
            all_labels.append(label)
        except Exception as e:
            print(f"Error processing {vid_path}: {e}")
            
    # Save bulk arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    np.save(os.path.join(args.output_dir, "X.npy"), X)
    np.save(os.path.join(args.output_dir, "y.npy"), y)
    print(f"Saved {X.shape[0]} sequences to {args.output_dir}")
    
    # Cleanup
    pose_estimator.close()
    hand_detector.close()

if __name__ == "__main__":
    main()
