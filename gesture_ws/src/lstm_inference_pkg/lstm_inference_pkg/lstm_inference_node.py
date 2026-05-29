import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import json
import torch
import sys
import os
import time
from amr_interfaces.msg import UnknownGesture, ConfirmReply, Intent

# Append path to load GestureLSTM
sys.path.append('/home/sayak/HybridTestBed/hand_gesture_lab')
from train import GestureLSTM

class LSTMInferenceNode(Node):
    def __init__(self):
        super().__init__('lstm_inference_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/sequence',
            self.sequence_callback,
            10)
        self.publisher_ = self.create_publisher(String, '/gesture/prediction', 10)
        
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

    def sequence_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        if len(data) != 8880: # 30 * 296
            self.get_logger().warning(f'Expected 8880 elements (30x296), got {len(data)}. Ignoring.', throttle_duration_sec=2.0)
            return
            
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

            if confidence > 0.6:
                label_str = self.labels[label_idx]
                self.paused = True
                self.pause_until_time = now + self.cooldown_known
                self.get_logger().info(f"Detected {label_str} ({confidence:.2f}). Pausing for {self.cooldown_known}s")
                
                msg_intent = Intent()
                msg_intent.stamp = self.get_clock().now().to_msg()
                msg_intent.session_id = f"lstm_{int(now)}"
                msg_intent.label = label_str
                msg_intent.confidence = confidence
                msg_intent.source = "lstm"
                msg_intent.latency_ms = 0
                self.pub_intent.publish(msg_intent)
            else:
                label_str = "UNCERTAIN"
                self.session_counter += 1
                session_id = f"sess_{int(now)}_{self.session_counter}"
                
                msg_unknown = UnknownGesture()
                msg_unknown.stamp = self.get_clock().now().to_msg()
                msg_unknown.session_id = session_id
                msg_unknown.window_id = 0
                msg_unknown.label = "UNCERTAIN"
                msg_unknown.confidence = confidence
                msg_unknown.hint = ""
                msg_unknown.source = "lstm"
                self.pub_unknown.publish(msg_unknown)
                
                self.waiting_for_confirmation = True
                self.current_unknown_session = session_id
                self.pause_until_time = now + self.cooldown_unknown
                self.get_logger().info(f"UNCERTAIN gesture detected. Triggering VLM branch (session: {session_id}).")
        
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
