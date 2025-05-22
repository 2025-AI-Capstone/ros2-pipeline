import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from collections import deque
import requests
import time
import json

class FallAlertNode(Node):
    def __init__(self):
        super().__init__('fall_alert_node')

        self.subscription = self.create_subscription(
            String,
            'falldetector/falldets',
            self.fall_callback,
            10
        )

        self.alert_publisher = self.create_publisher(String, 'fall_alert', 10)

        self.fall_history = deque()
        self.fall_window_sec = 5
        self.threshold_ratio = 0.5
        self.last_alert_time = 0
        self.alert_cooldown = 10

        self.user_id = 1

    def fall_callback(self, msg: String):
        data = json.loads(msg.data)
        now = time.time()
        confidence_score = data.get('confidence_score',[])
        self.send_alert(confidence_score)

    def send_alert(self, confidence_score: float):
        alert_msg = String()
        alert_msg.data = "Fall detected. Sending alert to server."
        self.alert_publisher.publish(alert_msg)
        self.get_logger().warn("published message: " ,alert_msg.data)

        data = {
            "user_id": self.user_id,
            "event_type": "fall",
            "status": "unconfirmed",
            "confidence_score": confidence_score
        }

        try:
            response = requests.post("http://localhost:8000/event-logs", json=data)
            if response.status_code == 200:
                self.get_logger().info("Alert successfully sent to server.")
            else:
                self.get_logger().error(f"Server error: {response.status_code} - {response.text}")
        except Exception as e:
            self.get_logger().error(f"Failed to send alert to server: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FallAlertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('fall_alert_node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
