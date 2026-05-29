import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from collections import deque

class SequenceBufferNode(Node):
    def __init__(self):
        super().__init__('sequence_buffer_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/keypoints',
            self.keypoints_callback,
            10)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/sequence', 10)
        
        self.seq_len = 30
        self.buffer = deque(maxlen=self.seq_len)
        self.latest_features = None
        
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info('Sequence Buffer Node started (Sub: /keypoints, Pub: /sequence)')

    def keypoints_callback(self, msg):
        features = np.array(msg.data, dtype=np.float32)
        if len(features) != 144:
            self.get_logger().warning(f'Expected 144 features, got {len(features)}. Ignoring.', throttle_duration_sec=2.0)
            return
            
        self.latest_features = features

    def compute_advanced_features(self, raw_buffer):
        seq_features = []
        for t in range(self.seq_len):
            curr_kp = raw_buffer[t]
            pose_kps = curr_kp[0:18].reshape(6, 3)
            left_kps = curr_kp[18:81].reshape(21, 3)
            right_kps = curr_kp[81:144].reshape(21, 3)
            
            l_shoulder = pose_kps[0]
            r_shoulder = pose_kps[1]
            if np.all(l_shoulder == 0) and np.all(r_shoulder == 0):
                shoulder_center = np.zeros(3)
            else:
                shoulder_center = (l_shoulder + r_shoulder) / 2.0
                
            rel_pose = pose_kps - shoulder_center
            rel_left = left_kps - shoulder_center
            rel_right = right_kps - shoulder_center
            coords = np.concatenate([rel_pose.flatten(), rel_left.flatten(), rel_right.flatten()])
            
            l_wrist = pose_kps[4]
            r_wrist = pose_kps[5]
            
            ext_l = np.linalg.norm(l_wrist - shoulder_center) if not np.all(l_wrist == 0) else 0.0
            ext_r = np.linalg.norm(r_wrist - shoulder_center) if not np.all(r_wrist == 0) else 0.0
            
            seq_features.append({
                'coords': coords,
                'ext_l': ext_l,
                'ext_r': ext_r,
                'l_wrist': l_wrist,
                'r_wrist': r_wrist
            })
            
        final_features = []
        for t in range(self.seq_len):
            curr = seq_features[t]
            prev = seq_features[max(0, t-1)]
            
            coords = curr['coords']
            v_t = coords - prev['coords']
            
            arm_ext = np.array([curr['ext_l'], curr['ext_r']])
            delta_arm = arm_ext - np.array([prev['ext_l'], prev['ext_r']])
            
            dx_l = curr['l_wrist'][0] - prev['l_wrist'][0]
            dy_l = curr['l_wrist'][1] - prev['l_wrist'][1]
            norm_l = np.sqrt(dx_l**2 + dy_l**2) + 1e-6
            dir_l = np.array([dx_l/norm_l, dy_l/norm_l])
            
            dx_r = curr['r_wrist'][0] - prev['r_wrist'][0]
            dy_r = curr['r_wrist'][1] - prev['r_wrist'][1]
            norm_r = np.sqrt(dx_r**2 + dy_r**2) + 1e-6
            dir_r = np.array([dx_r/norm_r, dy_r/norm_r])
            
            feat_t = np.concatenate([
                coords, v_t, arm_ext, delta_arm, dir_l, dir_r
            ])
            final_features.append(feat_t)
            
        final_features = np.array(final_features, dtype=np.float32)
        mean = np.mean(final_features, axis=0, keepdims=True)
        std = np.std(final_features, axis=0, keepdims=True) + 1e-6
        final_features = (final_features - mean) / std
        return final_features

    def timer_callback(self):
        if self.latest_features is None:
            return
            
        self.buffer.append(self.latest_features)
        
        if len(self.buffer) == self.seq_len:
            raw_buffer = np.array(self.buffer, dtype=np.float32)
            seq_features = self.compute_advanced_features(raw_buffer) # (30, 296)
            
            out_msg = Float32MultiArray()
            out_msg.data = seq_features.flatten().tolist()
            self.publisher_.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SequenceBufferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
