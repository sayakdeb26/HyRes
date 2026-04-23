import numpy as np

class FeatureBuilder:
    def __init__(self):
        # Pose: L/R Shoulder(11,12), L/R Elbow(13,14), L/R Wrist(15,16)
        self.pose_indices = [11, 12, 13, 14, 15, 16]
        # Hands: Wrist(0), Thumb Tip(4), Index Tip(8), Middle Tip(12), Ring Tip(16), Pinky Tip(20)
        self.hand_indices = [0, 4, 8, 12, 16, 20]
        self.target_dim = 60

    def build(self, pose_kps, left_kps, right_kps):
        """
        Constructs the 60-dim base feature vector.
        """
        features = []
        
        # 1. Setup global origin (Shoulder midpoint)
        if pose_kps is not None and len(pose_kps) > 12:
            left_shoulder = np.array(pose_kps[11][:3])
            right_shoulder = np.array(pose_kps[12][:3])
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        else:
            shoulder_mid = np.zeros(3)
            
        # 2. Pose (18 dims)
        if pose_kps is not None and len(pose_kps) > 16:
            for idx in self.pose_indices:
                pt = np.array(pose_kps[idx][:3]) - shoulder_mid
                features.extend(pt.tolist())
        else:
            features.extend([0.0] * 18)
            
        # 3. Hands (36 dims)
        for hand_kps in [left_kps, right_kps]:
            if hand_kps is not None and len(hand_kps) > 20:
                for idx in self.hand_indices:
                    pt = np.array(hand_kps[idx][:3]) - shoulder_mid
                    features.extend(pt.tolist())
            else:
                features.extend([0.0] * 18)
                
        # 4. Motion (Relative Wrist Displacement) (6 dims)
        if pose_kps is not None and len(pose_kps) > 16:
            l_wrist = np.array(pose_kps[15][:3]) - shoulder_mid
            r_wrist = np.array(pose_kps[16][:3]) - shoulder_mid
            features.extend(l_wrist.tolist())
            features.extend(r_wrist.tolist())
        else:
            features.extend([0.0] * 6)
            
        # 5. Standardization
        feat_arr = np.array(features, dtype=np.float32)
        mean = np.mean(feat_arr)
        std = np.std(feat_arr)
        if std > 1e-6:
            feat_arr = (feat_arr - mean) / std
            
        return feat_arr

