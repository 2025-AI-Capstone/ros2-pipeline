import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
from datetime import datetime, time, timedelta
import json

FASTAPI_BASE_URL = "http://localhost:8000"
LOGIN_ENDPOINT = f"{FASTAPI_BASE_URL}/login"
ROUTINE_ENDPOINT = f"{FASTAPI_BASE_URL}/routines/me"

USERNAME = "홍길동"
PASSWORD = "1234"

class SchedulerNode(Node):
    def __init__(self):
        super().__init__('routine_scheduler')
        self.publisher_ = self.create_publisher(String, 'routine_alarm_trigger', 10)
        self.timer = self.create_timer(30.0, self.check_routines)
        self.session_id = None
        self.cached_routines = []
        self.triggered_alarms = {}
        self.login()
        self.get_logger().info("Routine Scheduler Node initialized.")

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

    def fetch_routines(self):
        if not self.session_id:
            self.get_logger().warn("No valid session. Trying to re-login.")
            self.login()
            if not self.session_id:
                return []

        try:
            headers = {"Cookie": f"session_id={self.session_id}"}
            res = requests.get(ROUTINE_ENDPOINT, headers=headers, timeout=5)
            if res.status_code == 401:
                self.get_logger().warn("Session expired. Re-authenticating.")
                self.session_id = None
                return self.fetch_routines()

            res.raise_for_status()
            self.cached_routines = res.json()
            return self.cached_routines
        except Exception as e:
            self.get_logger().error(f"Routine fetch failed: {e}")
            return []

    def check_routines(self):
        routines = self.fetch_routines()
        now = datetime.now()

        for r in routines:
            rid = r.get('id')
            title = r.get('title')
            alarm_str = r.get('alarm_time')
            repeat = r.get('repeat_type', 'daily')
            if not rid or not alarm_str:
                continue

            try:
                alarm_t = time.fromisoformat(alarm_str)
            except ValueError:
                continue

            if self.triggered_alarms.get(rid) == now.date():
                continue

            diff = now - datetime.combine(now.date(), alarm_t)
            if timedelta(seconds=-30) < diff <= timedelta(seconds=30):
                msg = String()
                msg.data = json.dumps({
                    "id": rid,
                    "title": title,
                    "description": r.get("description", ""),
                    "alarm_time": alarm_str,
                    "repeat_type": repeat,
                    "triggered_at": now.isoformat()
                })
                self.publisher_.publish(msg)
                self.triggered_alarms[rid] = now.date()
                self.get_logger().info(f"Alarm triggered for {title} | Published: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = SchedulerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
