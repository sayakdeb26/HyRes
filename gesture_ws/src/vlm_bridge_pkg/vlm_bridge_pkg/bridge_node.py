#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from amr_interfaces.msg import UnknownGesture, ConfirmRequest, ConfirmReply, Intent, TrainingExample, VlmCallStart
from vlm_interfaces.msg import RecorderRequest
from vlm_interfaces.srv import InferClip
from std_msgs.msg import String
import json
import time

class BridgeNode(Node):
    def __init__(self):
        super().__init__('bridge_node')

        # Parameters
        self.declare_parameter('confirm_timeout_s', 20.0)
        self.declare_parameter('wait_clip_timeout_s', 5.0)
        self.declare_parameter('default_clip', '')

        self.confirm_timeout_s = self.get_parameter('confirm_timeout_s').value
        self.wait_clip_timeout_s = self.get_parameter('wait_clip_timeout_s').value
        self.default_clip = self.get_parameter('default_clip').value

        # State Machine
        self.current_session_id = ""
        self.current_window_id = -1
        self.awaiting_clip = False
        self.awaiting_confirmation = False
        self.clip_wait_start_time = 0.0
        self.confirm_wait_start_time = 0.0
        self.current_unknown_msg = None # Store the initial UnknownGesture
        self.timestamp_cache = {}
        self.vlm_predictions = {}

        # Store clip paths: (session_id, window_id) -> clip_path
        self.clip_map = {}

        # Subscribers
        self.sub_unknown = self.create_subscription(
            UnknownGesture,
            '/lstm/unknown',
            self.unknown_callback,
            10)
        
        self.sub_clip = self.create_subscription(
            String,
            '/recorder/clip_ready',
            self.clip_callback,
            10)
        
        self.sub_reply = self.create_subscription(
            ConfirmReply,
            '/ui/confirm_reply',
            self.reply_callback,
            10)
        self.sub_meta = self.create_subscription(
            String,
            '/video_meta',
            self.meta_callback,
            10)
        self.current_video_id = "unknown"
        self.current_true_label = "unknown"

        # Publishers
        self.pub_confirm = self.create_publisher(ConfirmRequest, '/vlm/confirm_request', 10)
        self.pub_intent = self.create_publisher(Intent, '/intents_raw', 10)
        self.pub_training = self.create_publisher(TrainingExample, '/db/training_example', 10)
        self.pub_req = self.create_publisher(RecorderRequest, '/recorder/request', 10)
        self.pub_vlm_start = self.create_publisher(VlmCallStart, '/vlm/call_start', 10)
        self.pub_reply = self.create_publisher(ConfirmReply, '/ui/confirm_reply', 10)

        # Service Client
        self.cli_infer = self.create_client(InferClip, '/vlm/infer')
        if not self.cli_infer.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('VLM service not available yet, waiting...')

        # Timer for timeouts
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info('Bridge Node started')

    def meta_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.current_video_id = data.get("video_id", "unknown")
            self.current_true_label = data.get("label", "unknown")
        except Exception as e:
            self.get_logger().error(f"Error parsing meta: {e}")

    def log_confidence(self, video_id, true_label, pred_label, confidence):
        import csv
        import os
        os.makedirs('/home/sayak/HybridTestBed/experiment_results/confidence', exist_ok=True)
        csv_path = '/home/sayak/HybridTestBed/experiment_results/confidence/confidence_logs.csv'
        header = ["video_id", "true_label", "predicted_label", "confidence", "correct"]
        file_exists = os.path.exists(csv_path)
        correct = 1 if true_label == pred_label else 0
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([video_id, true_label, pred_label, confidence, correct])

    def log_escalation(self, true_label, lstm_pred, lstm_conf, escalation_flag, vlm_pred, final_pred, correctness):
        import csv
        import os
        os.makedirs('/home/sayak/HybridTestBed/experiment_results/escalation', exist_ok=True)
        csv_path = '/home/sayak/HybridTestBed/experiment_results/escalation/hybrid_escalation_log.csv'
        header = ["true_label", "lstm_prediction", "confidence", "escalation_flag", "vlm_prediction", "final_prediction", "correctness"]
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([true_label, lstm_pred, lstm_conf, escalation_flag, vlm_pred, final_pred, correctness])

    def unknown_callback(self, msg):
        if self.current_session_id == "":
            # New session
            self.current_session_id = msg.session_id
            self.current_window_id = msg.window_id
            self.current_unknown_msg = msg
            self.awaiting_clip = True
            self.clip_wait_start_time = time.time()
            self.get_logger().info(f"Starting new session {self.current_session_id}. Waiting for clip.")
            self.timestamp_cache[msg.session_id] = {
                't0': msg.t0,
                't1': msg.t1,
                't2': msg.t2,
                't3': msg.t3,
                't4': 0,
                't5': 0
            }
            
            # Publish RecorderRequest
            req = RecorderRequest()
            req.session_id = msg.session_id
            req.window_id = str(msg.window_id)
            # Use defaults (0 means use recorder defaults)
            req.pre_secs = 0.0
            req.post_secs = 0.0
            req.t_center.sec = 0 # Use current time in recorder
            self.pub_req.publish(req)
            self.get_logger().info(f"RecorderRequest sent for {msg.session_id}")
            
            # Check if clip already exists
            key = (msg.session_id, msg.window_id)
            if key in self.clip_map:
                self.process_vlm(self.clip_map[key])
        elif msg.session_id != self.current_session_id:
            self.get_logger().warn(f"Ignored UnknownGesture {msg.session_id} while busy with {self.current_session_id}")

    def clip_callback(self, msg):
        try:
            data = json.loads(msg.data)
            sid = data['session_id']
            wid = int(data['window_id'])
            path = data['clip_path']
            
            self.clip_map[(sid, wid)] = path
            
            if self.awaiting_clip and sid == self.current_session_id and wid == self.current_window_id:
                self.process_vlm(path)
        except Exception as e:
            self.get_logger().error(f"Error parsing clip msg: {e}")

    def process_vlm(self, clip_path):
        self.awaiting_clip = False
        self.get_logger().info(f"Clip found: {clip_path}. Calling VLM.")
        
        # Check if VLM service is available
        if not self.cli_infer.service_is_ready():
            self.get_logger().error("VLM service not ready! Resetting session.")
            self.reset_session()
            return
        
        req = InferClip.Request()
        req.clip_path = clip_path
        req.label_hint = self.current_unknown_msg.label
        req.session_id = self.current_session_id
        
        # Publish VlmCallStart for metrics timing (BEFORE async call)
        vlm_start_msg = VlmCallStart()
        vlm_start_msg.stamp = self.get_clock().now().to_msg()
        vlm_start_msg.session_id = self.current_session_id
        vlm_start_msg.clip_path = clip_path
        self.pub_vlm_start.publish(vlm_start_msg)
        
        t4 = self.get_clock().now().nanoseconds
        if self.current_session_id in self.timestamp_cache:
            self.timestamp_cache[self.current_session_id]['t4'] = t4
        
        try:
            future = self.cli_infer.call_async(req)
            future.add_done_callback(lambda f: self.vlm_response_callback(f, clip_path))
        except Exception as e:
            self.get_logger().error(f"Failed to call VLM service: {e}")
            self.reset_session()

    def vlm_response_callback(self, future, clip_path):
        try:
            resp = future.result()
            t5 = self.get_clock().now().nanoseconds
            if self.current_session_id in self.timestamp_cache:
                self.timestamp_cache[self.current_session_id]['t5'] = t5
                
            if not resp or resp.label == "ERROR":
                self.get_logger().error(f"VLM returned error or empty response")
                self.reset_session()
                return
                
            self.vlm_predictions[self.current_session_id] = resp.label
                
            self.get_logger().info(f"VLM Response: {resp.label}, {resp.confidence}")
            
            # Send ConfirmRequest only if valid
            req = ConfirmRequest()
            req.stamp = self.get_clock().now().to_msg()
            req.session_id = self.current_session_id
            req.window_id = self.current_window_id
            req.candidate_label = resp.label
            req.candidate_conf = resp.confidence
            req.hint = resp.rationale
            req.source = "vlm"
            
            self.pub_confirm.publish(req)
            
            self.awaiting_confirmation = True
            self.confirm_wait_start_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f"VLM call failed: {e}")
            self.reset_session()

    def reply_callback(self, msg):
        if self.awaiting_confirmation and msg.session_id == self.current_session_id:
            t6 = self.get_clock().now().nanoseconds
            t_dict = self.timestamp_cache.get(msg.session_id, {})
            t0 = t_dict.get('t0', 0)
            t1 = t_dict.get('t1', 0)
            t2 = t_dict.get('t2', 0)
            t3 = t_dict.get('t3', 0)
            t4 = t_dict.get('t4', 0)
            t5 = t_dict.get('t5', 0)
            
            vlm_pred = self.vlm_predictions.get(msg.session_id, "unknown")
            
            if msg.approved and msg.final_label:
                self.get_logger().info(f"Approved: {msg.final_label}")
                
                # Publish Intent
                intent = Intent()
                intent.stamp = self.get_clock().now().to_msg()
                intent.session_id = msg.session_id
                intent.label = msg.final_label
                intent.confidence = 1.0 # User approved
                intent.source = "ui+vlm"
                intent.t0 = t0
                intent.t1 = t1
                intent.t2 = t2
                intent.t3 = t3
                intent.t4 = t4
                intent.t5 = t5
                intent.t6 = t6
                intent.latency_ms = int((t6 - t0) / 1000000) if t0 > 0 else 0
                self.pub_intent.publish(intent)
                
                # Publish TrainingExample
                te = TrainingExample()
                te.stamp = self.get_clock().now().to_msg()
                te.session_id = msg.session_id
                te.window_id = msg.window_id
                te.label = msg.final_label
                
                key = (msg.session_id, msg.window_id)
                te.clip_path = self.clip_map.get(key, "")
                te.confidence = 1.0
                te.source = "ui+vlm"
                self.pub_training.publish(te)
                
                # Log confidence and escalation
                self.log_confidence(self.current_video_id, self.current_true_label, msg.final_label, 1.0)
                self.log_escalation(self.current_true_label, self.current_unknown_msg.label, self.current_unknown_msg.confidence, 1, vlm_pred, msg.final_label, 1 if self.current_true_label == msg.final_label else 0)
            else:
                self.get_logger().info("Rejected by operator")
                self.log_confidence(self.current_video_id, self.current_true_label, "REJECTED", 0.0)
                self.log_escalation(self.current_true_label, self.current_unknown_msg.label, self.current_unknown_msg.confidence, 1, vlm_pred, "REJECTED", 0)
            
            # Clean up cache
            self.timestamp_cache.pop(msg.session_id, None)
            self.vlm_predictions.pop(msg.session_id, None)
            
            self.current_session_id = "" # Clear to prevent double negative ConfirmReply
            self.reset_session()

    def timer_callback(self):
        now = time.time()
        
        if self.awaiting_clip:
            if now - self.clip_wait_start_time > self.wait_clip_timeout_s:
                self.get_logger().warn("Clip timeout")
                if self.default_clip:
                     self.get_logger().info(f"Using default clip: {self.default_clip}")
                     self.process_vlm(self.default_clip)
                else:
                     self.reset_session()
        
        if self.awaiting_confirmation:
            if now - self.confirm_wait_start_time > self.confirm_timeout_s:
                self.get_logger().warn("Confirmation timeout")
                self.reset_session()

    def reset_session(self):
        if self.current_session_id:
            reply = ConfirmReply()
            reply.session_id = self.current_session_id
            reply.approved = False
            reply.final_label = ""
            self.pub_reply.publish(reply)
            self.get_logger().info(f"Published fallback negative ConfirmReply for session {self.current_session_id}")
            
            # Log escalation timeout/rejection
            vlm_pred = self.vlm_predictions.get(self.current_session_id, "unknown")
            self.log_confidence(self.current_video_id, self.current_true_label, "TIMEOUT", 0.0)
            if self.current_unknown_msg:
                self.log_escalation(self.current_true_label, self.current_unknown_msg.label, self.current_unknown_msg.confidence, 1, vlm_pred, "TIMEOUT", 0)
            
            self.timestamp_cache.pop(self.current_session_id, None)
            self.vlm_predictions.pop(self.current_session_id, None)
            
        self.current_session_id = ""
        self.current_window_id = -1
        self.awaiting_clip = False
        self.awaiting_confirmation = False
        self.current_unknown_msg = None
        self.get_logger().info("Session reset. Ready for next.")

def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
