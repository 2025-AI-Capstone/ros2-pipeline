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

        # 구독: 낙상 결과 토픽
        self.subscription = self.create_subscription(
            JointState,
            'falldetector/falldets',
            self.fall_callback,
            10
        )

        # 발행: 낙상 경고 메시지
        self.alert_publisher = self.create_publisher(String, 'fall_alert', 10)

        # 최근 5초간의 FALL 상태 기록 (timestamp, is_fall)
        self.fall_history = deque()

        # 파라미터
        self.fall_window_sec = 5         # 검사 시간 (초)
        self.threshold_ratio = 0.5       # 낙상 판단 비율
        self.last_alert_time = 0
        self.alert_cooldown = 10         # 알림 간 최소 간격 (초)

    def fall_callback(self, msg: JointState):
        now = time.time()

        # 현재 메시지에서 FALL 감지 여부 판단
        fall_detected = any(name == 'FALL' for name in msg.name)
        self.fall_history.append((now, fall_detected))

        # 5초 이내 데이터만 유지
        while self.fall_history and self.fall_history[0][0] < now - self.fall_window_sec:
            self.fall_history.popleft()

        # 낙상 비율 계산
        if len(self.fall_history) >= 10:  # 최소 메시지 수
            fall_count = sum(1 for t, is_fall in self.fall_history if is_fall)
            ratio = fall_count / len(self.fall_history)
            self.get_logger().info(f'FALL ratio: {ratio:.2f}')

            # 낙상으로 판단되면 알림 전송
            if ratio >= self.threshold_ratio and (now - self.last_alert_time) > self.alert_cooldown:
                self.send_alert()
                self.last_alert_time = now

    def send_alert(self):
        alert_msg = String()
        alert_msg.data = "⚠️ Fall Detected! Sending alert to server..."

        # ROS2 토픽으로 알림
        self.alert_publisher.publish(alert_msg)
        self.get_logger().warn(alert_msg.data)

        # 서버에 HTTP POST 요청
        try:
            response = requests.post("http://localhost:8080/api/fall_alert", json={"message": alert_msg.data})
            self.get_logger().info(f"Alert sent to server. Response: {response.status_code}")
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
