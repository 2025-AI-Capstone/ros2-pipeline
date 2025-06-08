import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty, Bool

from agent.modules.workflow import run_workflow
from agent.modules.agent_components import initialize_agent_components
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import requests
import json

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        load_dotenv()
        # initialize LangGraph LLM + components
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.agent_components = initialize_agent_components(self.llm)
        self.fall_alert = False
        self.fall_response = "낙상이 감지되었습니다. 비상연락이 필요하신가요?"
        # STT text subscription
        self.create_subscription(String, 'stt/speech_text', self.text_callback, 10)
        # fall detection subscription
        self.create_subscription(String, 'fall_alert/warning', self.fall_alert_callback, 10)
        self.create_subscription(String, 'routine_alarm_trigger', self.routine_callback, 10)
        # publish LLM answer
        self.response_publisher = self.create_publisher(String, 'agent/response', 10)
        # publish STT trigger
        self.stt_trigger_pub = self.create_publisher(Empty, 'agent/trigger', 10)

        self.get_logger().info("Agent Node ready")

    def text_callback(self, msg: String):
        input_text = msg.data
        # invoke the full LangGraph workflow
        answer = run_workflow(
            input=input_text,
            llm=self.llm,
            fall_alert= self.fall_alert,
            agent_components=self.agent_components
        )
        # reset fall flag
        self.fall_alert = False

        # publish result
        out = String()
        out.data = answer.content.strip()
        self.response_publisher.publish(out)
        log = {'query':input_text,'answer':answer.content.strip()}
        self.send_log(log)
        self.get_logger().info(f"Published answer: {answer.content}")

    def fall_alert_callback(self, msg: String):
        out = String()
        out.data = self.fall_response
        self.response_publisher.publish(out)
        self.get_logger().info(f"Published fall alert response: {self.fall_response}")
        self.fall_alert = True
        self.get_logger().info("Fall detected → triggering STT")
        self.stt_trigger_pub.publish(Empty())

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

            # 로그로도 전송
            self.send_log({
                "type": "routine",
                "received": msg.data,
                "response": response_text
            })

        except Exception as e:
            self.get_logger().error(f"Failed to handle routine message: {e}")

    def send_log(self, log):


        data = {
            "user_id": self.user_id,
            "event_type": "talk",
            "message": log,
            "confidence_score": 0
        }
        
        try:
            response = requests.post("http://localhost:8000/event-logs", json=data)
            if response.status_code == 200:
                self.get_logger().info("Successfully sent to server.")
            else:
                self.get_logger().info(f"Server error: {response.status_code} - {response.text}")
        except Exception as e:
            self.get_logger().info(f"Failed to send log to server: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
