import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import json
import torch
import sys
import os
import time
import csv
from amr_interfaces.msg import UnknownGesture, ConfirmReply, Intent, InstrumentedSequence

# Append path to load GestureLSTM
sys.path.append('/home/sayak/HybridTestBed/hand_gesture_lab')
from train import GestureLSTM

class LSTMInferenceNode(Node):
    def __init__(self):
        super().__init__('lstm_inference_node')
        self.subscription = self.create_subscription(
            InstrumentedSequence,
            '/sequence',
            self.sequence_callback,
            10)
        self.publisher_ = self.create_publisher(String, '/gesture/prediction', 10)
        
        self.sub_meta = self.create_subscription(
            String,
            '/video_meta',
            self.meta_callback,
            10)
        self.declare_parameter('confidence_threshold', 0.60)
        self.current_video_id = "unknown"
        self.current_true_label = "unknown"
        
        # Force disable cuDNN to fix local mismatch (9.21.1 vs 9.19.0)
        torch.backends.cudnn.enabled = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cooldown State (VLM/Uncertainty Branch)
        self.paused = False
        self.pause_until_time = 0.0
        self.waiting_for_confirmation = False
        self.current_unknown_session = ""
        self.cooldown_known = 2.5
        self.cooldown_unknown = 90.0
        self.session_counter = 0

        self.sub_reply = self.create_subscription(ConfirmReply, '/ui/confirm_reply', self.on_confirm_reply, 10)
        self.pub_unknown = self.create_publisher(UnknownGesture, '/lstm/unknown', 10)
        self.pub_intent = self.create_publisher(Intent, '/intents_raw', 10)
        
        # Load Model
        self.get_logger().info("Loading PyTorch LSTM model...")
        self.model = GestureLSTM(input_dim=296, num_classes=6)
        weights_path = '/home/sayak/HybridTestBed/hand_gesture_lab/weights/best_lstm_model.pth'
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.get_logger().info(f"Successfully loaded weights from {weights_path}")
        else:
            self.get_logger().warning(f"Weights file not found at {weights_path}. Predictions will be untrained!")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.labels = ["Swipe Left", "Swipe Right", "Rolling Forward", 
                       "Rolling Backward", "Thumb Down", "Stop Sign"]
        
        self.get_logger().info('LSTM Inference Node started (Sub: /sequence, Pub: /gesture/prediction)')

    def on_confirm_reply(self, msg: ConfirmReply):
        if self.waiting_for_confirmation and msg.session_id == self.current_unknown_session:
            self.waiting_for_confirmation = False
            self.paused = False
            self.get_logger().info(f"Resuming gesture recognition after VLM confirmation")

    def meta_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.current_video_id = data.get("video_id", "unknown")
            self.current_true_label = data.get("label", "unknown")
        except Exception as e:
            self.get_logger().error(f"Error parsing meta: {e}")

    def log_confidence(self, video_id, true_label, pred_label, confidence):
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
        os.makedirs('/home/sayak/HybridTestBed/experiment_results/escalation', exist_ok=True)
        csv_path = '/home/sayak/HybridTestBed/experiment_results/escalation/hybrid_escalation_log.csv'
        header = ["true_label", "lstm_prediction", "confidence", "escalation_flag", "vlm_prediction", "final_prediction", "correctness"]
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([true_label, lstm_pred, lstm_conf, escalation_flag, vlm_pred, final_pred, correctness])

    def sequence_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) != 8880: # 30 * 296
            self.get_logger().warning(f'Expected 8880 elements (30x296), got {len(data)}. Ignoring.', throttle_duration_sec=2.0)
            return
            
        t0 = msg.t0
        t1 = msg.t1
        t2 = msg.t2
        t3 = self.get_clock().now().nanoseconds
        
        sequence = data.reshape((1, 30, 296))
        
        with torch.no_grad():
            tensor_seq = torch.tensor(sequence).to(self.device)
            outputs = self.model(tensor_seq)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
            label_idx = pred.item()
            confidence = conf.item()
            
            # Confidence gating and Cooldown/VLM logic
            now = time.time()
            if self.paused and now < self.pause_until_time:
                return
            elif self.paused and now >= self.pause_until_time:
                self.paused = False
                self.get_logger().info("Auto-resuming after cooldown")

            if self.waiting_for_confirmation:
                return

            threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
            if confidence > threshold:
                label_str = self.labels[label_idx]
                self.paused = True
                self.pause_until_time = now + self.cooldown_known
                self.get_logger().info(f"Detected {label_str} ({confidence:.2f}). Pausing for {self.cooldown_known}s")
                
                t6 = self.get_clock().now().nanoseconds
                
                msg_intent = Intent()
                msg_intent.stamp = self.get_clock().now().to_msg()
                msg_intent.session_id = f"lstm_{int(now)}"
                msg_intent.label = label_str
                msg_intent.confidence = confidence
                msg_intent.source = "lstm"
                msg_intent.t0 = t0
                msg_intent.t1 = t1
                msg_intent.t2 = t2
                msg_intent.t3 = t3
                msg_intent.t4 = t3
                msg_intent.t5 = t3
                msg_intent.t6 = t6
                msg_intent.latency_ms = int((t6 - t0) / 1000000) if t0 > 0 else 0
                self.pub_intent.publish(msg_intent)
                
                # Log confidence directly
                self.log_confidence(self.current_video_id, self.current_true_label, label_str, confidence)
            else:
                label_str = self.labels[label_idx]
                self.session_counter += 1
                session_id = f"sess_{int(now)}_{self.session_counter}"
                
                msg_unknown = UnknownGesture()
                msg_unknown.stamp = self.get_clock().now().to_msg()
                msg_unknown.session_id = session_id
                msg_unknown.window_id = 0
                msg_unknown.label = label_str
                msg_unknown.confidence = confidence
                msg_unknown.hint = ""
                msg_unknown.source = "lstm"
                msg_unknown.t0 = t0
                msg_unknown.t1 = t1
                msg_unknown.t2 = t2
                msg_unknown.t3 = t3
                msg_unknown.t4 = 0
                msg_unknown.t5 = 0
                msg_unknown.t6 = 0
                self.pub_unknown.publish(msg_unknown)
                
                self.waiting_for_confirmation = True
                self.current_unknown_session = session_id
                self.pause_until_time = now + self.cooldown_unknown
                self.get_logger().info(f"UNCERTAIN gesture detected (Top-1 LSTM: {label_str} | Conf: {confidence:.2f}). Triggering VLM branch.")
        
        prediction = {
            "label": label_str,
            "confidence": round(confidence, 3)
        }
        
        out_msg = String()
        out_msg.data = json.dumps(prediction)
        self.publisher_.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LSTMInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
