import os
import glob
import cv2

class FrameLoader:
    def __init__(self, downsample_factor=1):
        """
        downsample_factor: Extract every Nth frame (e.g., 2 means every 2nd frame)
        """
        self.downsample_factor = downsample_factor

    def extract_frames(self, folder_path):
        """
        Yields RGB frames from a folder of sequential JPG images.
        """
        # Find all jpgs and sort them
        search_pattern = os.path.join(folder_path, "*.jpg")
        frame_paths = sorted(glob.glob(search_pattern))
        
        if not frame_paths:
            yield None
            return
            
        for i, frame_path in enumerate(frame_paths):
            if i % self.downsample_factor != 0:
                continue
                
            try:
                # Read using cv2
                frame_bgr = cv2.imread(frame_path)
                if frame_bgr is None:
                    continue
                    
                # Convert to RGB since MediaPipe expects RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_rgb, frame_path
                
            except Exception as e:
                # Handle unreadable images gracefully
                pass
