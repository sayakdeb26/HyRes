import cv2

class VideoLoader:
    def __init__(self, target_fps=None):
        self.target_fps = target_fps

    def extract_frames(self, video_path):
        """
        Yields RGB frames from a video file.
        Gracefully handles corrupted or unreadable videos.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield None
            return

        # Original video FPS (can be used to skip frames to reach target_fps)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = 1
        
        if self.target_fps and orig_fps and orig_fps > self.target_fps:
            frame_skip = int(round(orig_fps / self.target_fps))
            if frame_skip < 1:
                frame_skip = 1

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_skip == 0:
                    # Convert to RGB since MediaPipe expects RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield frame_rgb
                    
                frame_count += 1
        except Exception as e:
            # Catching OpenCV assertion errors or corrupted chunk errors
            pass
        finally:
            cap.release()
