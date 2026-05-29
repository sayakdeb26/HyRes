#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from amr_interfaces.msg import TrainingExample, Intent
import sqlite3
import os
import json

class CentralDBNode(Node):
    def __init__(self):
        super().__init__('central_db_node')

        # Parameters
        self.db_dir = '/home/sayak/amr_db'
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        
        self.db_path = os.path.join(self.db_dir, 'amr_gestures.db')
        self.init_db()

        # Subscribers
        self.sub_te = self.create_subscription(
            TrainingExample,
            '/db/training_example',
            self.te_callback,
            10)
        
        self.sub_intent = self.create_subscription(
            Intent,
            '/intents_raw',
            self.intent_callback,
            10)

        self.get_logger().info(f'Central DB Node started. DB: {self.db_path}')

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS training_examples
                     (timestamp REAL, session_id TEXT, window_id INTEGER, label TEXT, clip_path TEXT, confidence REAL, source TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS intents
                     (timestamp REAL, session_id TEXT, label TEXT, confidence REAL, latency_ms INTEGER, source TEXT)''')
        conn.commit()
        conn.close()

    def te_callback(self, msg):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            ts = msg.stamp.sec + msg.stamp.nanosec * 1e-9
            c.execute("INSERT INTO training_examples VALUES (?,?,?,?,?,?,?)",
                      (ts, msg.session_id, msg.window_id, msg.label, msg.clip_path, msg.confidence, msg.source))
            conn.commit()
            conn.close()
            self.get_logger().info(f"Logged TrainingExample: {msg.session_id}")
        except Exception as e:
            self.get_logger().error(f"DB Error: {e}")

    def intent_callback(self, msg):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            ts = msg.stamp.sec + msg.stamp.nanosec * 1e-9
            c.execute("INSERT INTO intents VALUES (?,?,?,?,?,?)",
                      (ts, msg.session_id, msg.label, msg.confidence, msg.latency_ms, msg.source))
            conn.commit()
            conn.close()
            # self.get_logger().info(f"Logged Intent: {msg.session_id}")
        except Exception as e:
            self.get_logger().error(f"DB Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CentralDBNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
