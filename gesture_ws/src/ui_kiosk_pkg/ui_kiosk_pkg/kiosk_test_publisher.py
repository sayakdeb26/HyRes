#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vlm_interfaces.msg import ConfirmRequest
import time

class KioskTestPublisher(Node):
    def __init__(self):
        super().__init__('kiosk_test_publisher')
        self.pub = self.create_publisher(ConfirmRequest, '/vlm/confirm_request', 10)
        self.timer = self.create_timer(5.0, self.timer_cb)
        self.session = 0
        self.get_logger().info("Kiosk test publisher ready.")

    def timer_cb(self):
        msg = ConfirmRequest()
        self.session += 1
        msg.session_id = f"test_{self.session:04d}"
        msg.candidate_label = "wave"
        msg.candidate_conf = 0.72
        msg.clip_path = "/home/sayak/amr_kiosk_media/test.mp4"
        self.pub.publish(msg)
        self.get_logger().info(f"Published fake ConfirmRequest for {msg.clip_path}")
        time.sleep(0.1)

def main():
    rclpy.init()
    node = KioskTestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
