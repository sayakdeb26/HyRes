#!/usr/bin/env python3
import os
import glob
import json
import logging
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("processing_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("jester_preprocessor")

# --- Constants ---
SEQ_LEN = 30
TARGET_SIZE = (256, 256)

POSE_INDICES = [11, 12, 13, 14, 15, 16] # L/R Shoulder, L/R Elbow, L/R Wrist
HAND_INDICES = list(range(21))

# --- Functions ---

def process_video(video_id, video_dir, label_id):
    """
    Process a single video directory.
    1. Load frames.
    2. Sample/pad to SEQ_LEN.
    3. Extract Holistic keypoints.
    4. Construct features.
    Returns: video_id, features (SEQ_LEN, F), label_id, error_msg
    """
    try:
        mp_holistic = mp.solutions.holistic
        
        # 1. Frame Loading & Sampling
        frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        if not frame_files:
            return video_id, None, label_id, "No frames found"
            
        num_frames = len(frame_files)
        
        # Uniform sampling or padding
        if num_frames >= SEQ_LEN:
            indices = np.linspace(0, num_frames - 1, SEQ_LEN, dtype=int)
        else:
            # Pad with the last frame
            indices = np.arange(num_frames)
            pad_count = SEQ_LEN - num_frames
            indices = np.pad(indices, (0, pad_count), 'edge')
            
        # 2. Extract Keypoints
        raw_kps = []
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            for idx in indices:
                img_path = frame_files[idx]
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                frame = cv2.resize(frame, TARGET_SIZE)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = holistic.process(frame_rgb)
                
                # Extract coordinates [x, y, z]
                pose_kps = np.zeros((len(POSE_INDICES), 3))
                left_kps = np.zeros((21, 3))
                right_kps = np.zeros((21, 3))
                
                # Pose
                if results.pose_landmarks:
                    for i, p_idx in enumerate(POSE_INDICES):
                        lm = results.pose_landmarks.landmark[p_idx]
                        pose_kps[i] = [lm.x, lm.y, lm.z]
                
                # Left Hand
                if results.left_hand_landmarks:
                    for i, lm in enumerate(results.left_hand_landmarks.landmark):
                        left_kps[i] = [lm.x, lm.y, lm.z]
                        
                # Right Hand
                if results.right_hand_landmarks:
                    for i, lm in enumerate(results.right_hand_landmarks.landmark):
                        right_kps[i] = [lm.x, lm.y, lm.z]
                        
                raw_kps.append({
                    'pose': pose_kps,
                    'left': left_kps,
                    'right': right_kps
                })
                
        if len(raw_kps) != SEQ_LEN:
             pad_kps = raw_kps[-1] if raw_kps else {'pose':np.zeros((6,3)), 'left':np.zeros((21,3)), 'right':np.zeros((21,3))}
             while len(raw_kps) < SEQ_LEN:
                 raw_kps.append(pad_kps)

        # 3. Feature Construction
        seq_features = []
        
        for t in range(SEQ_LEN):
            curr_kp = raw_kps[t]
            
            # Shoulder center (pose[0]=L shoulder, pose[1]=R shoulder)
            l_shoulder = curr_kp['pose'][0]
            r_shoulder = curr_kp['pose'][1]
            if np.all(l_shoulder == 0) and np.all(r_shoulder == 0):
                shoulder_center = np.zeros(3)
            else:
                shoulder_center = (l_shoulder + r_shoulder) / 2.0
                
            # Relative coordinates
            rel_pose = curr_kp['pose'] - shoulder_center
            rel_left = curr_kp['left'] - shoulder_center
            rel_right = curr_kp['right'] - shoulder_center
            
            coords = np.concatenate([rel_pose.flatten(), rel_left.flatten(), rel_right.flatten()]) # 48*3 = 144
            
            # Arm extension (pose[4]=L wrist, pose[5]=R wrist)
            l_wrist = curr_kp['pose'][4]
            r_wrist = curr_kp['pose'][5]
            
            ext_l = np.linalg.norm(l_wrist - shoulder_center) if not np.all(l_wrist == 0) else 0.0
            ext_r = np.linalg.norm(r_wrist - shoulder_center) if not np.all(r_wrist == 0) else 0.0
            
            seq_features.append({
                'coords': coords,
                'ext_l': ext_l,
                'ext_r': ext_r,
                'l_wrist': l_wrist,
                'r_wrist': r_wrist
            })
            
        # Compile temporal features
        final_features = []
        for t in range(SEQ_LEN):
            curr = seq_features[t]
            prev = seq_features[max(0, t-1)]
            
            # Relative coords (144)
            coords = curr['coords']
            
            # Velocity (144)
            v_t = coords - prev['coords']
            
            # Arm extension (2) & Delta arm (2)
            arm_ext = np.array([curr['ext_l'], curr['ext_r']])
            delta_arm = arm_ext - np.array([prev['ext_l'], prev['ext_r']])
            
            # Direction features for wrists (dx, dy normalized)
            # L wrist
            dx_l = curr['l_wrist'][0] - prev['l_wrist'][0]
            dy_l = curr['l_wrist'][1] - prev['l_wrist'][1]
            norm_l = np.sqrt(dx_l**2 + dy_l**2) + 1e-6
            dir_l = np.array([dx_l/norm_l, dy_l/norm_l])
            
            # R wrist
            dx_r = curr['r_wrist'][0] - prev['r_wrist'][0]
            dy_r = curr['r_wrist'][1] - prev['r_wrist'][1]
            norm_r = np.sqrt(dx_r**2 + dy_r**2) + 1e-6
            dir_r = np.array([dx_r/norm_r, dy_r/norm_r])
            
            feat_t = np.concatenate([
                coords,        # 144
                v_t,           # 144
                arm_ext,       # 2
                delta_arm,     # 2
                dir_l,         # 2
                dir_r          # 2
            ]) # Total F = 296
            
            final_features.append(feat_t)
            
        final_features = np.array(final_features, dtype=np.float32) # (30, 296)
        
        # Sequence Normalization (Zero Mean, Unit Variance)
        mean = np.mean(final_features, axis=0, keepdims=True)
        std = np.std(final_features, axis=0, keepdims=True) + 1e-6
        final_features = (final_features - mean) / std
        
        # Check NaNs
        if np.isnan(final_features).any():
            return video_id, None, label_id, "NaNs generated in features"
            
        return video_id, final_features, label_id, None
        
    except Exception as e:
        return video_id, None, label_id, f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='/home/sayak/datasets/jester/Train.csv')
    parser.add_argument('--data_dir', type=str, default='/home/sayak/datasets/jester/Train')
    parser.add_argument('--out_dir', type=str, default='./data/processed_full')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help="Limit number of videos for testing")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load CSV
    logger.info(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    
    # Label Map
    label_map = dict(zip(df['label'], df['label_id']))
    with open(os.path.join(args.out_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
        
    videos = df['video_id'].astype(str).tolist()
    labels = df['label_id'].astype(int).tolist()
    
    if args.limit > 0:
        videos = videos[:args.limit]
        labels = labels[:args.limit]
        
    logger.info(f"Total videos to process: {len(videos)}")
    
    # 2. Multiprocessing
    X_list = []
    y_list = []
    failed = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for vid, lbl in zip(videos, labels):
            v_dir = os.path.join(args.data_dir, vid)
            futures[executor.submit(process_video, vid, v_dir, lbl)] = vid
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
            vid, features, lbl, err = future.result()
            if err:
                logger.error(f"Video {vid} failed: {err}")
                failed += 1
            else:
                X_list.append(features)
                y_list.append(lbl)
                
    elapsed = time.time() - start_time
    
    # 3. Save Outputs
    if not X_list:
        logger.error("No valid sequences processed.")
        return
        
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    
    logger.info("Saving X.npy and y.npy...")
    np.save(os.path.join(args.out_dir, 'X.npy'), X_arr)
    np.save(os.path.join(args.out_dir, 'y.npy'), y_arr)
    
    # 4. Summary
    speed = len(videos) / elapsed
    
    logger.info("\n" + "="*40)
    logger.info("=== Output Summary ===")
    logger.info(f"Total sequences processed : {len(X_arr)}")
    logger.info(f"Total failures            : {failed}")
    logger.info(f"Processing time           : {elapsed:.2f} s")
    logger.info(f"Processing speed          : {speed:.2f} videos/sec")
    logger.info(f"Output shape X            : {X_arr.shape}")
    logger.info(f"Output shape y            : {y_arr.shape}")
    logger.info(f"Feature dimension F       : {X_arr.shape[2]}")
    logger.info("="*40)
    
    # Class distribution
    unique, counts = np.unique(y_arr, return_counts=True)
    logger.info("Class distribution:")
    for u, c in zip(unique, counts):
        logger.info(f"  Class {u:2d}: {c} samples")

if __name__ == "__main__":
    main()
