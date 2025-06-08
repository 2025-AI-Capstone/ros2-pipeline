import rclpy
from rclpy.node import Node
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

        self.alert_publisher = self.create_publisher(String, 'fall_alert/warning', 10)

        # 최근 낙상 여부 기록 (timestamp, is_fall)
        self.fall_history = deque()
        self.fall_window_sec = 5 
        self.threshold_ratio = 0.5  # 50% 이상이면 alert
        self.last_alert_time = 0
        self.alert_cooldown = 180  # 최소 180초 간격으로 alert

        self.user_id = 1

    def fall_callback(self, msg: String):
        now = time.time()

        try:
            data = json.loads(msg.data)
            is_fall = data.get('is_fall', False)
            confidence_score = data.get('confidence_score', 0.0)
        except Exception as e:
            self.get_logger().error(f"Invalid message format: {e}")
            return

        self.fall_history.append((now, is_fall))
        self._clean_old_entries(now)

        fall_count = sum(1 for t, fall in self.fall_history if fall)
        total_count = len(self.fall_history)
        fall_ratio = fall_count / total_count if total_count > 0 else 0.0

        if fall_ratio >= self.threshold_ratio:
            if now - self.last_alert_time > self.alert_cooldown:
                self.last_alert_time = now
                self.send_alert(confidence_score)

    def _clean_old_entries(self, current_time):
        while self.fall_history and current_time - self.fall_history[0][0] > self.fall_window_sec:
            self.fall_history.popleft()

    def send_alert(self, confidence_score: float):
        alert_msg = String()
        alert_msg.data = "Fall detected. Sending alert to server."
        self.alert_publisher.publish(alert_msg)
        self.get_logger().info(f"Published message: {alert_msg.data}")

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
                self.get_logger().info(f"Server error: {response.status_code} - {response.text}")
        except Exception as e:
            self.get_logger().info(f"Failed to send alert to server: {e}")


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
