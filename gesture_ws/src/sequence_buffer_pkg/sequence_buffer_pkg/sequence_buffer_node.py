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
            '/gesture/features',
            self.features_callback,
            10)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/gesture/sequence', 10)
        
        self.seq_len = 30
        self.buffer = deque(maxlen=self.seq_len)
        
        # Indices for velocity calculation (24 dims total)
        # Wrists (12-17), Index, Middle, Thumb tips (21-29, 39-47)
        self.vel_indices = list(range(12, 18)) + list(range(21, 30)) + list(range(39, 48))
        
        self.get_logger().info('Sequence Buffer Node started (Publishing to /gesture/sequence)')

    def features_callback(self, msg):
        features = np.array(msg.data, dtype=np.float32)
        if len(features) != 60:
            self.get_logger().warning(f'Expected 60 features, got {len(features)}. Ignoring.', throttle_duration_sec=2.0)
            return
            
        self.buffer.append(features)
        
        # Only publish when the buffer is full (sliding window)
        if len(self.buffer) == self.seq_len:
            seq_features = np.array(self.buffer) # (30, 60)
            
            # Calculate velocities
            velocities = np.zeros((self.seq_len, 24), dtype=np.float32)
            for t in range(1, self.seq_len):
                velocities[t] = seq_features[t, self.vel_indices] - seq_features[t-1, self.vel_indices]
                
            # Concatenate to get (30, 84)
            final_sequence = np.concatenate([seq_features, velocities], axis=1)
            
            # Publish flattened array
            out_msg = Float32MultiArray()
            out_msg.data = final_sequence.flatten().tolist()
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
