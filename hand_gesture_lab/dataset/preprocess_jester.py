import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.frame_loader import FrameLoader
from dataset.csv_loader import CSVLoader
from dataset.keypoint_pipeline import KeypointPipeline
from dataset.validator import Validator
from dataset.feature_builder import FeatureBuilder
from dataset.utils import setup_logger, sliding_window
from src.visualizer import Visualizer  # Re-use visualizer for debug mode

logger = setup_logger()

def process_video_task(folder_path, label_idx, seq_len, stride, downsample, debug, debug_out_dir):
    """
    Task function for processing a single video folder (frames).
    """
    vid_id = os.path.basename(folder_path)
    
    try:
        loader = FrameLoader(downsample_factor=downsample)
        pipeline = KeypointPipeline()
        validator = Validator()
        feature_builder = FeatureBuilder()
        visualizer = Visualizer() if debug else None
        
        if debug and debug_out_dir:
            vid_debug_dir = os.path.join(debug_out_dir, vid_id)
            os.makedirs(vid_debug_dir, exist_ok=True)
        
        frame_features = []
        valid_frames_count = 0
        total_frames = 0
        
        for result in loader.extract_frames(folder_path):
            if result is None:
                continue
                
            frame_rgb, frame_path = result
            total_frames += 1
            
            hands_kps, hands_conf = pipeline.process_frame(frame_rgb)
            
            hand_label = None
            if 'right' in hands_kps:
                hand_label = 'right'
            elif 'left' in hands_kps:
                hand_label = 'left'
                
            feature_vec = None
            smoothed_kps = None
            
            if hand_label:
                raw_kps = hands_kps[hand_label]
                conf = hands_conf[hand_label]
                
                smoothed_kps = validator.validate_and_smooth(raw_kps, conf, frame_rgb.shape)
                if smoothed_kps is not None:
                    feature_vec = feature_builder.build(smoothed_kps)
                    valid_frames_count += 1
                    
            if feature_vec is None:
                feature_vec = np.zeros(63, dtype=np.float32)
                
            frame_features.append(feature_vec)
            
            # Debug visualization
            if debug and visualizer:
                # Convert back to BGR for saving
                debug_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if smoothed_kps is not None:
                    visualizer.draw_skeleton(debug_frame, smoothed_kps)
                
                # Draw ROI if possible (requires modifying pipeline to return ROI, but skip for simplicity here or re-calculate)
                # Just draw text for now
                status = "VALID" if smoothed_kps is not None else "INVALID"
                color = (0, 255, 0) if smoothed_kps is not None else (0, 0, 255)
                cv2.putText(debug_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                out_path = os.path.join(vid_debug_dir, os.path.basename(frame_path))
                cv2.imwrite(out_path, debug_frame)
                
        pipeline.close()
        
        if debug:
            logger.info(f"[{vid_id}] Valid frames: {valid_frames_count}/{total_frames}")
        
        # Sequences
        sequences = sliding_window(frame_features, seq_len=seq_len, stride=stride)
        
        if len(sequences) > 0:
            labels = [label_idx] * len(sequences)
            return sequences, labels
        else:
            return [], []
            
    except Exception as e:
        logger.error(f"Error processing {folder_path}: {str(e)}")
        return [], []

def main():
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
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
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
        
        for future in tqdm(as_completed(future_to_vid), total=len(video_tasks), desc="Processing Sequence Folders"):
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
