import numpy as np

class FeatureBuilder:
    def __init__(self):
        pass

    def build(self, keypoints):
        """
        Normalizes and flattens keypoints into a feature vector.
        keypoints is a list of [x, y, z] for 21 landmarks.
        """
        if not keypoints:
            return np.zeros(63)
            
        kp_array = np.array(keypoints)
        
        # 1. Wrist relative normalization
        wrist = kp_array[0]
        relative_kp = kp_array - wrist
        
        # 2. Scale normalization
        # Use distance from wrist (0) to middle finger MCP (9) as scale anchor
        middle_mcp = relative_kp[9]
        scale = np.linalg.norm(middle_mcp[:2]) # Use 2D distance for scale
        
        if scale > 0:
            normalized_kp = relative_kp / scale
        else:
            normalized_kp = relative_kp

        # 3. Flatten
        feature_vector = normalized_kp.flatten()
        return feature_vector
