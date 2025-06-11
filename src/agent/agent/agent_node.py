import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty

from agent.modules.workflow import run_workflow
from agent.modules.agent_components import initialize_agent_components
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import requests
import json

FASTAPI_BASE_URL = "http://localhost:8000"
LOGIN_ENDPOINT = f"{FASTAPI_BASE_URL}/login"
EVENT_LOG_ENDPOINT = f"{FASTAPI_BASE_URL}/event-logs"

USERNAME = "홍길동"
PASSWORD = "1234"

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        load_dotenv()
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.agent_components = initialize_agent_components(self.llm)
        self.fall_alert = False
        self.fall_response = "낙상이 감지되었습니다. 비상연락이 필요하신가요?"
        self.user_id = 1
        self.session_id = None

        self.login()

        # STT 텍스트 입력
        self.create_subscription(String, 'stt/speech_text', self.text_callback, 10)
        # 낙상 감지 알림
        self.create_subscription(String, 'fall_alert/warning', self.fall_alert_callback, 10)
        # 루틴 알림
        self.create_subscription(String, 'routine_alarm_trigger', self.routine_callback, 10)

        # 응답 발행
        self.response_publisher = self.create_publisher(String, 'agent/response', 10)
        self.stt_trigger_pub = self.create_publisher(Empty, 'agent/trigger', 10)

        self.get_logger().info("Agent Node ready")

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

    def text_callback(self, msg: String):
        input_text = msg.data

        answer = run_workflow(
            input=input_text,
            llm=self.llm,
            fall_alert=self.fall_alert,
            agent_components=self.agent_components
        )

        self.fall_alert = False
        if answer is None:
            answer = "죄송합니다. 응답을 생성할 수 없습니다."
        elif not isinstance(answer, str):
            answer = str(answer)    
        
        out = String()
        out.data = answer
        self.response_publisher.publish(out)

        # 서버 로그 전송 - message를 문자열로 변경
        log_message = f"Query: {input_text} | Answer: {answer}"
        self.send_log("talk", log_message)
        self.get_logger().info(f"Published answer: {answer}")

    def fall_alert_callback(self, msg: String):
        out = String()
        out.data = self.fall_response
        self.response_publisher.publish(out)
        self.get_logger().info(f"Published fall alert response: {self.fall_response}")
        self.fall_alert = True
        self.stt_trigger_pub.publish(Empty())
        
        # 낙상 알림 로그 전송
        log_message = f"Fall alert detected | Response: {self.fall_response}"
        self.send_log("fall_alert", log_message)

    def routine_callback(self, msg: String):
        try:
            routine = json.loads(msg.data)
            title = routine.get("title", "루틴")
            alarm_time = routine.get("alarm_time", None)

            response_text = f"지금은 '{title}' 시간입니다."

            out = String()
            out.data = response_text
            self.response_publisher.publish(out)
            self.get_logger().info(f"Published routine alert: {response_text}")

            # 루틴 알림 로그 전송
            log_message = f"Routine alert: {title} | Time: {alarm_time} | Response: {response_text}"
            self.send_log("routine", log_message)

        except Exception as e:
            self.get_logger().error(f"Failed to handle routine message: {e}")

    def send_log(self, event_type, message, confidence_score=0.8):
        """서버 스키마에 맞게 로그 전송"""
        if not self.session_id:
            self.get_logger().warn("No valid session. Trying to re-login.")
            self.login()
            if not self.session_id:
                self.get_logger().error("Failed to send log due to missing session.")
                return

        data = {
            "user_id": self.user_id,
            "event_type": event_type,
            "message": message,  # 문자열로 변경
            "status": "completed",  # 필수 필드 추가
            "confidence_score": confidence_score
        }

        headers = {"Cookie": f"session_id={self.session_id}"}

        try:
            response = requests.post(EVENT_LOG_ENDPOINT, json=data, headers=headers, timeout=5)
            if response.status_code == 200:
                self.get_logger().info("Successfully sent log to server.")
            elif response.status_code == 401:
                self.get_logger().warn("Session expired. Re-authenticating.")
                self.session_id = None
                self.send_log(event_type, message, confidence_score)  # 재시도
            else:
                self.get_logger().error(f"Server error: {response.status_code} - {response.text}")
        except Exception as e:
            self.get_logger().error(f"Failed to send log to server: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('AgentNode interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()