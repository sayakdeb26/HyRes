#!/usr/bin/env python3
import os, time, sys, uuid, shutil
import rclpy
from rclpy.node import Node
from vlm_interfaces.msg import ConfirmRequest

MEDIA_DIR = "/home/sayak/amr_kiosk_media"

class Tester(Node):
    def __init__(self):
        super().__init__('kiosk_test_publisher')
        self.pub = self.create_publisher(ConfirmRequest, "/vlm/confirm_request", 10)
        self.timer = self.create_timer(5.0, self.fire)
        self.count = 0
        self.get_logger().info("Test publisher up. Put a small .mp4 into /home/sayak/amr_kiosk_media to reuse.")
    def fire(self):
        # pick any .mp4 in MEDIA_DIR or bail
        mp4 = None
        for f in os.listdir(MEDIA_DIR):
            if f.lower().endswith(".mp4"):
                mp4 = os.path.join(MEDIA_DIR, f)
                break
        if not mp4:
            self.get_logger().warn("No .mp4 in media_dir; drop one file to test.")
            return
        sid = str(uuid.uuid4())[:8]
        msg = ConfirmRequest()
        msg.session_id = sid
        msg.candidate_label = f"gesture_{self.count%3}"
        msg.candidate_conf = 0.6 + 0.1*(self.count%3)
        msg.clip_path = mp4
        self.pub.publish(msg)
        self.count += 1
        self.get_logger().info(f"Sent ConfirmRequest {sid} with {os.path.basename(mp4)}")
def main():
    rclpy.init()
    rclpy.spin(Tester())
    rclpy.shutdown()
if __name__ == "__main__":
    main()
