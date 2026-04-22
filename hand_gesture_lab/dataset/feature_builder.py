import numpy as np

class FeatureBuilder:
    def __init__(self, target_dim=63):
        self.target_dim = target_dim

    def build(self, keypoints):
        """
        Normalizes and flattens keypoints into a feature vector.
        keypoints: numpy array of shape (21, 3) representing [x, y, z].
        """
        if keypoints is None:
            return np.zeros(self.target_dim, dtype=np.float32)
            
        kp_array = np.array(keypoints, dtype=np.float32)
        
        # 1. Translation: Wrist becomes origin (0, 0, 0)
        wrist = kp_array[0]
        relative_kp = kp_array - wrist
        
        # 2. Scaling: Scale by distance from wrist (0) to middle finger MCP (9)
        middle_mcp = relative_kp[9]
        scale = np.linalg.norm(middle_mcp[:2]) # Using 2D distance for scaling
        
        if scale > 0:
            normalized_kp = relative_kp / scale
        else:
            normalized_kp = relative_kp
            
        # 3. Flatten
        feature_vector = normalized_kp.flatten()
        return feature_vector
