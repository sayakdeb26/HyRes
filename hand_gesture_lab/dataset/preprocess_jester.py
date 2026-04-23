import os
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.frame_loader import FrameLoader
from dataset.csv_loader import CSVLoader
from dataset.keypoint_pipeline import KeypointPipeline
from dataset.validator import Validator
from dataset.feature_builder import FeatureBuilder
from dataset.utils import setup_logger
from src.visualizer import Visualizer  # Re-use visualizer for debug mode

logger = setup_logger()

global_pipeline = None

def init_worker():
    global global_pipeline
    global_pipeline = KeypointPipeline()

def process_video_task(folder_path, label_idx, seq_len, stride, downsample, debug, debug_out_dir):
    """
    Task function for processing a single video folder (frames).
    Returns a single sequence of shape (seq_len, 84).
    """
    vid_id = os.path.basename(folder_path)
    
    try:
        # 1. Frame Extraction (Uniform Sampling / Padding)
        loader = FrameLoader(target_size=(256, 256), seq_len=seq_len)
        global global_pipeline
        pipeline = global_pipeline
        validator = Validator()
        feature_builder = FeatureBuilder()
        visualizer = Visualizer() if debug else None
        
        if debug and debug_out_dir:
            vid_debug_dir = os.path.join(debug_out_dir, vid_id)
            os.makedirs(vid_debug_dir, exist_ok=True)
        
        frame_features = []
        valid_frames_count = 0
        total_frames = 0
        
        extracted_frames = loader.extract_frames(folder_path)
        if not extracted_frames or len(extracted_frames) != seq_len:
            return [], []
            
        for frame_rgb, frame_path in extracted_frames:
            total_frames += 1
            
            # 2. Keypoint Pipeline (Holistic + Hands)
            pose_kps, hands_kps, hands_conf = pipeline.process_frame(frame_rgb)
            
            # 3. Validation Layer
            val_pose = validator.validate_and_smooth(pose_kps, 1.0, frame_rgb.shape, 'pose')
            
            left_kps = None
            if 'left' in hands_kps:
                left_kps = validator.validate_and_smooth(hands_kps['left'], hands_conf['left'], frame_rgb.shape, 'left')
                
            right_kps = None
            if 'right' in hands_kps:
                right_kps = validator.validate_and_smooth(hands_kps['right'], hands_conf['right'], frame_rgb.shape, 'right')
                
            if val_pose is not None or left_kps is not None or right_kps is not None:
                valid_frames_count += 1
                
            # 4. Feature Construction (60 Dims)
            base_feat = feature_builder.build(val_pose, left_kps, right_kps)
            frame_features.append(base_feat)
            
            # Debug visualization
            if debug and visualizer:
                debug_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(vid_debug_dir, os.path.basename(frame_path))
                cv2.imwrite(out_path, debug_frame)
                
        if debug:
            logger.info(f"[{vid_id}] Valid frames: {valid_frames_count}/{total_frames}")
        
        seq_features = np.array(frame_features) # (seq_len, 60)
        
        # 5. Velocity Calculation (24 Dims)
        # Select: L/R Wrists (12-17), L/R Index, Middle, Thumb tips (21-29, 39-47)
        vel_indices = list(range(12, 18)) + list(range(21, 30)) + list(range(39, 48))
        velocities = np.zeros((seq_len, 24), dtype=np.float32)
        
        for t in range(1, seq_len):
            velocities[t] = seq_features[t, vel_indices] - seq_features[t-1, vel_indices]
            
        # Assembly (30, 84)
        final_sequence = np.concatenate([seq_features, velocities], axis=1)
        
        if np.isnan(final_sequence).any():
            logger.warning(f"NaN found in sequence {vid_id}, skipping.")
            return [], []
            
        return [final_sequence], [label_idx]
            
    except Exception as e:
        logger.error(f"Error processing {folder_path}: {str(e)}")
        return [], []

import multiprocessing as mp

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    parser = argparse.ArgumentParser(description="Jester Preprocessor for JPG Sequences")
    parser.add_argument('--input_dir', type=str, required=True, help="Train/Validation directory containing video_id folders")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to Train.csv / Validation.csv")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for processed data")
    parser.add_argument('--seq_len', type=int, default=30, help="Sequence length")
    parser.add_argument('--stride', type=int, default=5, help="Sliding window stride")
    parser.add_argument('--downsample', type=int, default=1, help="Downsample factor for frames")
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 4, help="Number of parallel workers")
    parser.add_argument('--sample_ratio', type=float, default=1.0, help="Fraction of dataset to process (e.g. 0.3 for 30%)")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode (saves visual output)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    debug_out_dir = os.path.join(args.output_dir, "debug_frames") if args.debug else None
    if args.debug:
        os.makedirs(debug_out_dir, exist_ok=True)
        logger.info(f"DEBUG MODE ENABLED. Intermediate frames will be saved to {debug_out_dir}")
        
    logger.info(f"Loading CSV mapping from {args.csv_path}")
    csv_loader = CSVLoader(args.csv_path)
    vid_to_label = csv_loader.get_vid_to_label_map()
    
    csv_loader.save_label_map(os.path.join(args.output_dir, "label_map.json"))
    logger.info("Saved label_map.json")
    
    # Collect video tasks
    video_tasks = []
    
    # Iterate through input_dir (folders are video_ids)
    for folder_name in os.listdir(args.input_dir):
        folder_path = os.path.join(args.input_dir, folder_name)
        if os.path.isdir(folder_path):
            vid_id = folder_name.strip()
            if vid_id in vid_to_label:
                label_idx = vid_to_label[vid_id]
                video_tasks.append((folder_path, label_idx))
            else:
                if args.debug:
                    logger.warning(f"Video {vid_id} found in directory but not in CSV.")
                    
    if args.sample_ratio < 1.0:
        import random
        random.seed(42) # Ensure reproducible sampling
        sample_size = int(len(video_tasks) * args.sample_ratio)
        video_tasks = random.sample(video_tasks, sample_size)
        logger.info(f"Subsampled dataset to {args.sample_ratio*100}% ({len(video_tasks)} videos).")
                    
    logger.info(f"Found {len(video_tasks)} valid videos to process using {args.workers} workers.")
    
    all_X = []
    all_y = []
    
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
        future_to_vid = {
            executor.submit(
                process_video_task, 
                folder_path, 
                label_idx, 
                args.seq_len, 
                args.stride, 
                args.downsample, 
                args.debug,
                debug_out_dir
            ): folder_path
            for folder_path, label_idx in video_tasks
        }
        
        for i, future in enumerate(tqdm(as_completed(future_to_vid), total=len(video_tasks), desc="Processing Sequence Folders")):
            folder_path = future_to_vid[future]
            try:
                seqs, labels = future.result()
                if seqs:
                    all_X.extend(seqs)
                    all_y.extend(labels)
            except Exception as e:
                logger.error(f"Task exception for {folder_path}: {str(e)}")
                
    if not all_X:
        logger.error("No valid sequences were extracted. Check your data.")
        return
        
    X_arr = np.array(all_X, dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.int64)
    
    # Final Validation
    if np.isnan(X_arr).any():
        logger.error("FATAL ERROR: NaNs detected in X_arr!")
        return
        
    logger.info(f"Extracted {len(X_arr)} sequences of shape {X_arr.shape[1:]}")
    
    np.save(os.path.join(args.output_dir, "X.npy"), X_arr)
    np.save(os.path.join(args.output_dir, "y.npy"), y_arr)
    logger.info(f"Saved successfully to {args.output_dir}")

if __name__ == "__main__":
    main()
