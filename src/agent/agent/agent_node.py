import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty, Bool

from workflow import run_workflow
from agent_components import initialize_agent_components

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        # initialize LangGraph LLM + components
        model_id = "Qwen/Qwen2.5-3B-Instruct"
        self.agent_components = initialize_agent_components(model_id)
        self.llm = self.agent_components["llm"]
        self.fall_alert = False

        # STT text subscription
        self.create_subscription(String, 'stt/speech_text', self.text_callback, 10)
        # fall detection subscription
        self.create_subscription(Bool, 'assistant/fall_alert', self.fall_alert_callback, 10)
        # publish LLM answer
        self.response_publisher = self.create_publisher(String, 'agent/response', 10)
        # publish STT trigger
        self.stt_trigger_pub = self.create_publisher(Empty, 'stt/trigger', 10)

        self.get_logger().info("Agent Node ready")

    def text_callback(self, msg: String):
        input_text = msg.data
        # invoke the full LangGraph workflow
        answer = run_workflow(
            input=input_text,
            llm=self.llm,
            fall_alert=self.fall_alert,
            agent_components=self.agent_components
        )
        # reset fall flag
        self.fall_alert = False

        # publish result
        out = String()
        out.data = answer
        self.response_publisher.publish(out)
        self.get_logger().info(f"Published answer: {answer}")

    def fall_alert_callback(self, msg: Bool):
        if msg.data:
            self.fall_alert = True
            self.get_logger().info("Fall detected → triggering STT")
            self.stt_trigger_pub.publish(Empty())

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
