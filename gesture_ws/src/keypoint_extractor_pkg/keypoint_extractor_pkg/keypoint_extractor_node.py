import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np

class KeypointExtractorNode(Node):
    def __init__(self):
        super().__init__('keypoint_extractor_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.publisher_ = self.create_publisher(Float32MultiArray, '/keypoints', 10)
        self.bridge = CvBridge()
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        self.pose_indices = [11, 12, 13, 14, 15, 16]
        
        # Temporal buffers for missing keypoints
        self.last_valid_kps = {
            'pose': np.zeros((6, 3)),
            'left': np.zeros((21, 3)),
            'right': np.zeros((21, 3))
        }
        
        self.get_logger().info('Keypoint Extractor Node started (Sub: /camera/image_raw, Pub: /keypoints)')

    def extract_features(self, results):
        pose_kps = np.zeros((6, 3))
        left_kps = np.zeros((21, 3))
        right_kps = np.zeros((21, 3))
        
        if results.pose_landmarks:
            for i, p_idx in enumerate(self.pose_indices):
                lm = results.pose_landmarks.landmark[p_idx]
                pose_kps[i] = [lm.x, lm.y, lm.z]
            self.last_valid_kps['pose'] = pose_kps
        else:
            pose_kps = self.last_valid_kps['pose']
            
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                left_kps[i] = [lm.x, lm.y, lm.z]
            self.last_valid_kps['left'] = left_kps
        else:
            left_kps = self.last_valid_kps['left']
            
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                right_kps[i] = [lm.x, lm.y, lm.z]
            self.last_valid_kps['right'] = right_kps
        else:
            right_kps = self.last_valid_kps['right']
            
        # Compile all 48 absolute keypoints (144 dims)
        abs_coords = np.concatenate([pose_kps.flatten(), left_kps.flatten(), right_kps.flatten()])
        return abs_coords.astype(np.float32)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Same resolution scaling as offline
            cv_image = cv2.resize(cv_image, (256, 256))
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            results = self.holistic.process(image_rgb)
            features = self.extract_features(results)
            
            feature_msg = Float32MultiArray()
            feature_msg.data = features.tolist()
            self.publisher_.publish(feature_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = KeypointExtractorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
