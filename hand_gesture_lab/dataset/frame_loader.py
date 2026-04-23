import os
import glob
import cv2
import numpy as np

class FrameLoader:
    def __init__(self, target_size=(256, 256), seq_len=30):
        """
        target_size: Resize all frames to (W, H)
        seq_len: Exact number of frames to return per video
        """
        self.target_size = target_size
        self.seq_len = seq_len

    def extract_frames(self, folder_path):
        """
        Returns exactly `seq_len` RGB frames from a folder of sequential JPG images.
        Resizes to `target_size`.
        """
        search_pattern = os.path.join(folder_path, "*.jpg")
        frame_paths = sorted(glob.glob(search_pattern))
        
        if not frame_paths:
            return []
            
        num_frames = len(frame_paths)
        
        # Determine indices for uniform sampling or padding
        if num_frames > self.seq_len:
            indices = np.linspace(0, num_frames - 1, self.seq_len, dtype=int)
        else:
            indices = list(range(num_frames))
            last_idx = num_frames - 1
            indices.extend([last_idx] * (self.seq_len - num_frames))
            
        selected_frames = []
        
        for idx in indices:
            frame_path = frame_paths[idx]
            try:
                frame_bgr = cv2.imread(frame_path)
                if frame_bgr is None:
                    continue
                
                frame_resized = cv2.resize(frame_bgr, self.target_size)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                selected_frames.append((frame_rgb, frame_path))
                
            except Exception as e:
                pass
                
        # Ensure we return exactly seq_len frames even if some reads failed
        if 0 < len(selected_frames) < self.seq_len:
            last_valid = selected_frames[-1]
            selected_frames.extend([last_valid] * (self.seq_len - len(selected_frames)))
            
        return selected_frames
