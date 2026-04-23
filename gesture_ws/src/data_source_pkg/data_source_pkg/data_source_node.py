import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class DataSourceNode(Node):
    def __init__(self):
        super().__init__('data_source_node')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback) # 30 FPS
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()
        self.get_logger().info('Data Source Node started (Publishing to /camera/image_raw)')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
        else:
            self.get_logger().warning('Failed to capture frame from camera.', throttle_duration_sec=1.0)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataSourceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
