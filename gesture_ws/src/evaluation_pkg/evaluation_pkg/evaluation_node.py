import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class EvaluationNode(Node):
    def __init__(self):
        super().__init__('evaluation_node')
        self.subscription = self.create_subscription(
            String,
            '/gesture/prediction',
            self.prediction_callback,
            10)
        
        self.get_logger().info('Evaluation Node started (Listening on /gesture/prediction)')

    def prediction_callback(self, msg):
        self.get_logger().info(f'Prediction Received: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
