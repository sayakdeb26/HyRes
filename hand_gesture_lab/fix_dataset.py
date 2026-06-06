import numpy as np
import os
from tqdm import tqdm

def fix_dataset():
    data_dir = '/home/sayak/HyRes/hand_gesture_lab/data/processed/train'
    x_path = os.path.join(data_dir, 'X.npy')
    
    print("Loading original X.npy...")
    X = np.load(x_path).astype(np.float32)
    base = X.copy()  # Use all 84 dims
    
    N, T, F = base.shape
    
    print("Repairing missing keypoints (forward fill)...")
    for i in tqdm(range(N)):
        last_pose = None
        last_left = None
        last_right = None
        last_vel = None
        
        for t in range(T):
            frame = base[i, t]
            
            # Check if parts are missing (all values identical in the 18-dim blocks)
            pose_missing = np.all(frame[0:18] == frame[0])
            left_missing = np.all(frame[18:36] == frame[18])
            right_missing = np.all(frame[36:54] == frame[36])
            
            # Forward fill the entire 84 dims if pose is missing
            if pose_missing and last_pose is not None:
                base[i, t, 0:18] = last_pose
                base[i, t, 54:84] = last_vel  # Wrist and precomputed velocities
            elif not pose_missing:
                last_pose = frame[0:18].copy()
                last_vel = frame[54:84].copy()
                
            if left_missing and last_left is not None:
                base[i, t, 18:36] = last_left
            elif not left_missing:
                last_left = frame[18:36].copy()
                
            if right_missing and last_right is not None:
                base[i, t, 36:54] = last_right
            elif not right_missing:
                last_right = frame[36:54].copy()
                
    print(f"Fixed dataset shape: {base.shape}")
    out_path = os.path.join(data_dir, 'X_fixed.npy')
    np.save(out_path, base)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    fix_dataset()
