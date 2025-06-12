import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from custom_msgs.msg import CustomBoolean
from collections import deque
import requests
import time
import json

FASTAPI_BASE_URL = "http://localhost:8000"
LOGIN_ENDPOINT = f"{FASTAPI_BASE_URL}/login"
EVENT_LOG_ENDPOINT = f"{FASTAPI_BASE_URL}/event-logs"

USERNAME = "홍길동"
PASSWORD = "1234"

class FallAlertNode(Node):
    def __init__(self):
        super().__init__('fall_alert_node')

        self.subscription = self.create_subscription(
            CustomBoolean,
            'falldetector/falldets',
            self.fall_callback,
            10
        )
        self.alert_publisher = self.create_publisher(String, 'fall_alert/warning', 10)

        self.fall_history = deque()
        self.fall_window_sec = 4
        self.threshold_ratio = 0.8
        self.last_alert_time = 0
        self.alert_cooldown = 180

        self.user_id = 1
        self.session_id = None

        self.login()
        self.get_logger().info("FallAlertNode initialized with session handling.")

    def login(self):
        try:
            payload = {"name": USERNAME, "password": PASSWORD}
            res = requests.post(LOGIN_ENDPOINT, json=payload, timeout=5)
            res.raise_for_status()
            self.session_id = res.cookies.get("session_id")
            if not self.session_id:
                raise ValueError("No session_id in login response")
            self.get_logger().info(f"Login successful, session_id: {self.session_id}")
        except Exception as e:
            self.get_logger().error(f"Login failed: {e}")
            self.session_id = None

    def fall_callback(self, msg: CustomBoolean):
        now = time.time()
        is_fall = msg.is_fall.data

        self.fall_history.append((now, is_fall))
        self._clean_old_entries(now)

        fall_count = sum(1 for t, fall in self.fall_history if fall)
        total_count = len(self.fall_history)
        fall_ratio = fall_count / total_count if total_count > 0 else 0.0

        if fall_ratio >= self.threshold_ratio:
            if now - self.last_alert_time > self.alert_cooldown:
                self.last_alert_time = now
                self.send_alert(fall_ratio)

    def _clean_old_entries(self, current_time):
        while self.fall_history and current_time - self.fall_history[0][0] > self.fall_window_sec:
            self.fall_history.popleft()

    def send_alert(self, fall_ratio):
        if not self.session_id:
            self.get_logger().warn("No valid session. Trying to re-login.")
            self.login()
            if not self.session_id:
                self.get_logger().error("Alert aborted due to failed login.")
                return

        alert_msg = String()
        alert_msg.data = "Fall detected. Sending alert to server."
        self.alert_publisher.publish(alert_msg)
        self.get_logger().info(f"Published message: {alert_msg.data}")

        data = {
            "user_id": self.user_id,
            "event_type": "fall",
            "confidence_score": fall_ratio,
            "status": "낙상이 감지되었습니다."
        }

        headers = {"Cookie": f"session_id={self.session_id}"}
        try:
            response = requests.post(EVENT_LOG_ENDPOINT, json=data, headers=headers, timeout=5)
            if response.status_code == 200:
                self.get_logger().info("Alert successfully sent to server.")
            elif response.status_code == 401:
                self.get_logger().warn("Session expired. Re-authenticating...")
                self.session_id = None
                self.send_alert(fall_ratio)  # 재시도
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
