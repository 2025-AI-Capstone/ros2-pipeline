import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty, Bool
from typing import Dict, Any

from agent.modules.workflow import run_workflow
from agent.modules.agent_components import initialize_agent_components
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import requests
import json

# Constants for login and event logging endpoints
LOGIN_ENDPOINT = "http://localhost:8000/login"
EVENT_LOG_ENDPOINT = "http://localhost:8000/event-logs"
USERNAME = "홍길동" # Example username, adjust if needed
PASSWORD = "1234" # Example password, adjust if needed

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        load_dotenv()
        self.user_id = os.getenv("USER_ID", "default_user") # Get USER_ID from env
        # initialize LangGraph LLM + components
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.agent_components = initialize_agent_components(self.llm)
        self.fall_alert = False
        self.fall_response = "낙상이 감지되었습니다. 비상연락이 필요하신가요?"
        self.session_id = None # Initialize session_id

        self.login()

        # STT 텍스트 입력
        self.create_subscription(String, 'stt/speech_text', self.text_callback, 10)
        # 낙상 감지 알림
        self.create_subscription(String, 'fall_alert/warning', self.fall_alert_callback, 10)
        # 루틴 알림
        self.create_subscription(String, 'routine_alarm_trigger', self.routine_callback, 10)

        # 응답 발행
        self.response_publisher = self.create_publisher(String, 'agent/response', 10)
        self.alert_publisher = self.create_publisher(String, 'agent/alert', 10)

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

        out = String()
        out.data = answer.strip()
        self.response_publisher.publish(out)

        # Prepare log data as a dictionary for the new send_log signature
        log_data = {'query':input_text,'answer':answer.strip(), 'event_type': 'talk'}
        self.send_log(log_data)

        self.get_logger().info(f"Published answer: {answer}")

    def fall_alert_callback(self, msg: String):
        out = String()
        out.data = self.fall_response
        self.alert_publisher.publish(out)
        self.get_logger().info(f"Published fall alert response: {self.fall_response}")
        self.fall_alert = True
        
        # Prepare log data as a dictionary for the new send_log signature
        log_data = {'query':"Fall alert detected", 'answer': self.fall_response, 'event_type': 'fall_alert'}
        self.send_log(log_data)

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

            # Prepare log data as a dictionary for the new send_log signature
            log_data = {'query':"Routine alert","answer": response_text, 'event_type': 'routine'}
            self.send_log(log_data)

        except Exception as e:
            self.get_logger().error(f"Failed to handle routine message: {e}")

    def send_log(self, log_data: Dict[str, Any]):
        """서버 스키마에 맞게 로그 전송"""
        if not self.session_id:
            self.get_logger().warn("No valid session. Trying to re-login.")
            self.login()
            if not self.session_id:
                self.get_logger().error("Failed to send log due to missing session.")
                return

        data = {
            "user_id": self.user_id,
            "event_type": log_data.get("event_type", "default"), # Use event_type from log_data or default
            "message": json.dumps(log_data),
            "status": "agent",
            "confidence_score": log_data.get("confidence_score", 0)
        }

        headers = {"Cookie": f"session_id={self.session_id}"}

        try:
            response = requests.post(EVENT_LOG_ENDPOINT, json=data, headers=headers, timeout=5)
            if response.status_code == 200:
                self.get_logger().info("Successfully sent log to server.")
            elif response.status_code == 401:
                self.get_logger().warn("Session expired. Re-authenticating.")
                self.session_id = None
                self.send_log(log_data)  # Retry with new session
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