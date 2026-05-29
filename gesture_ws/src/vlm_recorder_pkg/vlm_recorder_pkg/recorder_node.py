#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from amr_interfaces.msg import UnknownGesture
from vlm_interfaces.msg import RecorderRequest
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import collections
import time
import json
import os
import threading

class RecorderNode(Node):
    def __init__(self):
        super().__init__('recorder_node')

        # Parameters
        self.declare_parameter('input_topic', '/frames/simplified')
        self.declare_parameter('save_dir', '/home/sayak/amr_gesture_ws/data/runtime_clips')
        self.declare_parameter('window_seconds', 4.0)
        self.declare_parameter('pre_secs', 2.0)
        self.declare_parameter('post_secs', 2.0)

        self.input_topic = self.get_parameter('input_topic').value
        self.save_dir = self.get_parameter('save_dir').value
        self.window_seconds = self.get_parameter('window_seconds').value
        self.pre_secs = self.get_parameter('pre_secs').value
        self.post_secs = self.get_parameter('post_secs').value

        # Create save directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # State
        self.bridge = CvBridge()
        self.frame_buffer = collections.deque() # Stores (timestamp, cv_image)
        self.lock = threading.Lock()

        # Subscribers
        self.sub_img = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10)
        
        self.sub_request = self.create_subscription(
            RecorderRequest,
            '/recorder/request',
            self.request_callback,
            10)
        
        # Deduplication
        self.active_sessions = set()

        # Publisher
        self.pub_clip_ready = self.create_publisher(String, '/recorder/clip_ready', 10)

        self.get_logger().info(f'Recorder Node started. Saving to {self.save_dir}')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = time.time() # Use system time for simplicity in correlation
            
            with self.lock:
                self.frame_buffer.append((timestamp, cv_image))
                
                # Prune old frames
                while self.frame_buffer and (timestamp - self.frame_buffer[0][0] > self.window_seconds):
                    self.frame_buffer.popleft()
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

    def request_callback(self, msg):
        # Handle RecorderRequest from Bridge
        session_id = msg.session_id
        window_id = msg.window_id
        
        # Deduplication
        with self.lock:
            if session_id in self.active_sessions:
                self.get_logger().warn(f"Session {session_id} already recording. Ignoring.")
                return
            self.active_sessions.add(session_id)

        # Use t_center from message if available, else current time
        # msg.t_center is a ROS Time object. Convert to float seconds.
        t_center = time.time()
        if msg.t_center.sec > 0:
            t_center = msg.t_center.sec + msg.t_center.nanosec * 1e-9
            
        # Use params from message if valid, else defaults
        pre = msg.pre_secs if msg.pre_secs > 0 else self.pre_secs
        post = msg.post_secs if msg.post_secs > 0 else self.post_secs

        self.get_logger().info(f"Recording clip for session {session_id} (t={t_center:.2f}, pre={pre}, post={post})")
        
        threading.Thread(target=self.save_clip_thread, args=(t_center, session_id, window_id, pre, post)).start()

    def save_clip_thread(self, t_center, session_id, window_id, pre_secs, post_secs):
        # Wait for post_secs
        time.sleep(post_secs + 0.5) # Extra buffer

        # Collect frames
        frames_to_save = []
        with self.lock:
            # Copy buffer to avoid holding lock too long
            for ts, img in self.frame_buffer:
                if (ts >= t_center - pre_secs) and (ts <= t_center + post_secs):
                    frames_to_save.append(img)
        
        if not frames_to_save:
            self.get_logger().warn(f"No frames found for session {session_id}")
            with self.lock:
                self.active_sessions.discard(session_id)
            return

        # Save to MP4
        filename = f"clip_{session_id}_{window_id}.mp4"
        filepath = os.path.join(self.save_dir, filename)
        
        height, width, layers = frames_to_save[0].shape
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        temp_filepath = filepath + ".raw.mp4"
        try:
            out = cv2.VideoWriter(temp_filepath, codec, 10.0, (width, height))
            for frame in frames_to_save:
                out.write(frame)
            out.release()
            
            # Ensure flush
            time.sleep(0.2)

            # Transcode to browser-friendly H.264
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", temp_filepath, "-vcodec", "libx264", "-pix_fmt", "yuv420p", filepath],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Cleanup temp raw file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
            self.get_logger().info(f"Saved browser-compatible H.264 clip: {filepath}")

            # Publish clip_ready
            info = {
                "session_id": session_id,
                "window_id": window_id,
                "clip_path": filepath,
                "t_center": str(t_center),
                "pre_secs": pre_secs,
                "post_secs": post_secs,
                "success": True
            }
            
            msg = String()
            msg.data = json.dumps(info)
            self.pub_clip_ready.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Failed to save clip: {e}")
        finally:
            with self.lock:
                self.active_sessions.discard(session_id)

def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
