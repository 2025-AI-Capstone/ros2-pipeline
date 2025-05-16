import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from collections import deque
import requests
import time

class FallAlertNode(Node):
    def __init__(self):
        super().__init__('fall_alert_node')

        self.subscription = self.create_subscription(
            JointState,
            'falldetector/falldets',
            self.fall_callback,
            10
        )

        self.alert_publisher = self.create_publisher(String, 'fall_alert/warning', 10)

        self.fall_history = deque()
        self.fall_window_sec = 5
        self.threshold_ratio = 0.5
        self.last_alert_time = 0
        self.alert_cooldown = 10

        self.user_id = 1  # 사용자 ID (필요 시 파라미터화 가능)

    def fall_callback(self, msg: JointState):
        now = time.time()
        fall_detected = any(name == 'FALL' for name in msg.name)
        self.fall_history.append((now, fall_detected))

        while self.fall_history and self.fall_history[0][0] < now - self.fall_window_sec:
            self.fall_history.popleft()

        if len(self.fall_history) >= 3:
            fall_count = sum(1 for t, is_fall in self.fall_history if is_fall)
            ratio = fall_count / len(self.fall_history)

            if ratio >= self.threshold_ratio and (now - self.last_alert_time) > self.alert_cooldown:
                confidence = round(ratio * 100, 1)
                self.send_alert(confidence)
                self.last_alert_time = now

    def send_alert(self, confidence_score: float):
        alert_msg = String()
        alert_msg.data = "Fall detected. Sending alert to server."
        self.alert_publisher.publish(alert_msg)
        self.get_logger().info("published message:{alert_msg.data}")

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
