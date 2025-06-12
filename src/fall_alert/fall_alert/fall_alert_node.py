import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from custom_msgs.msg import CustomBoolean

from collections import deque
import threading
import time
import requests
import redis

FASTAPI_BASE_URL = "http://localhost:8000"
LOGIN_ENDPOINT = f"{FASTAPI_BASE_URL}/login"
EVENT_LOG_ENDPOINT = f"{FASTAPI_BASE_URL}/event-logs"

USERNAME = "홍길동"
PASSWORD = "1234"
REDIS_HOST = "localhost"
REDIS_STATE_KEY = "fall_alert_enabled"
REDIS_CHANNEL = "fall_alert_toggle"


class FallAlertNode(Node):
    def __init__(self):
        super().__init__('fall_alert_node')

        # ROS2 구독/퍼블리셔
        self.subscription = self.create_subscription(CustomBoolean, 'falldetector/falldets', self.fall_callback, 10)
        self.alert_publisher = self.create_publisher(String, 'fall_alert/warning', 10)

        # 낙상 감지 상태 관련 변수
        self.fall_history = deque()
        self.fall_window_sec = 4
        self.threshold_ratio = 0.8
        self.last_alert_time = 0
        self.alert_cooldown = 60

        self.user_id = 1
        self.session_id = None

        # Redis에서 초기 알림 상태 로드
        self.alert_enabled = self.load_initial_alert_status()
        self.get_logger().info(f"초기 낙상 알림 상태: {'활성화' if self.alert_enabled else '비활성화'}")

        # Redis Pub/Sub 수신 스레드 실행
        threading.Thread(target=self.redis_listener, daemon=True).start()

        # 로그인
        self.login()
        self.get_logger().info("FallAlertNode initialized with session handling.")

    def load_initial_alert_status(self):
        try:
            r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
            val = r.get(REDIS_STATE_KEY)
            if val:
                return val.decode().strip().lower() == "true"
        except Exception as e:
            self.get_logger().warn(f"Redis 상태 불러오기 실패: {e}")
        return True  # 기본값은 True

    def redis_listener(self):
        try:
            r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
            pubsub = r.pubsub()
            pubsub.subscribe(REDIS_CHANNEL)
            self.get_logger().info("Redis 채널 구독 시작")

            for message in pubsub.listen():
                if message['type'] != 'message':
                    continue
                cmd = message['data'].decode().strip().lower()
                self.get_logger().info(f"[Redis] 수신 명령: {cmd}")

                if cmd == "enable":
                    self.alert_enabled = True
                elif cmd == "disable":
                    self.alert_enabled = False
                else:
                    self.get_logger().warn(f"[Redis] 알 수 없는 명령: {cmd}")
        except Exception as e:
            self.get_logger().error(f"Redis 리스너 오류: {e}")

    def login(self):
        try:
            payload = {"name": USERNAME, "password": PASSWORD}
            res = requests.post(LOGIN_ENDPOINT, json=payload, timeout=5)
            res.raise_for_status()
            self.session_id = res.cookies.get("session_id")
            if not self.session_id:
                raise ValueError("세션 ID 없음")
            self.get_logger().info("로그인 성공")
        except Exception as e:
            self.get_logger().error(f"로그인 실패: {e}")
            self.session_id = None

    def fall_callback(self, msg: CustomBoolean):
        now = time.time()
        is_fall = msg.is_fall.data

        self.fall_history.append((now, is_fall))
        self._clean_old_entries(now)

        fall_count = sum(1 for _, fall in self.fall_history if fall)
        total_count = len(self.fall_history)
        fall_ratio = fall_count / total_count if total_count > 0 else 0.0

        if fall_ratio >= self.threshold_ratio and (now - self.last_alert_time > self.alert_cooldown):
            self.last_alert_time = now
            self.send_alert(fall_ratio)

    def _clean_old_entries(self, current_time):
        while self.fall_history and current_time - self.fall_history[0][0] > self.fall_window_sec:
            self.fall_history.popleft()

    def send_alert(self, fall_ratio):
        if not self.alert_enabled:
            self.get_logger().info("낙상 알림이 비활성화되어 있어 메시지를 발행하지 않음.")
            return

        if not self.session_id:
            self.get_logger().warn("유효한 세션 없음. 재로그인 시도.")
            self.login()
            if not self.session_id:
                self.get_logger().error("세션 복구 실패. 알림 중단.")
                return

        alert_msg = String()
        alert_msg.data = "Fall detected. Sending alert to server."
        self.alert_publisher.publish(alert_msg)
        self.get_logger().info(f"Published: {alert_msg.data}")

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
                self.get_logger().info("서버로 알림 전송 완료")
            elif response.status_code == 401:
                self.get_logger().warn("세션 만료. 재인증 시도 중...")
                self.session_id = None
                self.send_alert(fall_ratio)
            else:
                self.get_logger().error(f"서버 오류: {response.status_code} - {response.text}")
        except Exception as e:
            self.get_logger().error(f"서버 알림 전송 실패: {e}")


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
