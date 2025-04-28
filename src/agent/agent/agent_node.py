import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty, Bool
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        # 초기화 변수 설정
        self.start_time = time.time()
        
        # LLM 모델 로드
        self.llm = self.load_kcgpt2_llm()
        self.prompt_template = PromptTemplate(
            input_variables=["input_text"],
            template="다음 질문에 답해주세요: {input_text} \n\n 답변:"
        )

        # 구독: STT 텍스트
        self.create_subscription(
            String,
            'stt/speech_text',
            self.text_callback,
            10
        )

        # 구독: 낙상 알림
        self.create_subscription(
            Bool,
            'assistant/fall_alert',
            self.fall_alert_callback,
            10
        )

        # 발행: Agent 응답
        self.response_publisher = self.create_publisher(
            String,
            'agent/response',
            10
        )
        # 발행: STT 트리거
        self.stt_trigger_pub = self.create_publisher(
            Empty,
            'stt/trigger',
            10
        )

        self.get_logger().info("Agent Node 초기화 완료")
        
    def text_callback(self, msg: String):
        """사용자 음성이 텍스트로 변환되었을 때 호출되는 콜백"""
        input_text = msg.data
        self.get_logger().info(f"받은 텍스트: {input_text}")
        formatted_prompt = self.prompt_template.format(input_text=input_text)
        # LLM을 사용하여 응답 생성
        try:
            response = self.llm.invoke(formatted_prompt)
            self.get_logger().info(f"생성된 응답: {response}")

            # 응답 발행
            response_msg = String()
            response_msg.data = response
            self.response_publisher.publish(response_msg)
            self.get_logger().info("publish succeed")
        except Exception as e:
            self.get_logger().error(f"응답 생성 오류: {e}")

    def fall_alert_callback(self, msg: Bool):
        """낙상 감지 시 STT 트리거 메시지 발행"""
        if msg.data:
            self.get_logger().info('Fall alert received, triggering STT')
            self.stt_trigger_pub.publish(Empty())

    def load_kcgpt2_llm(self):
        """한국어 GPT-2 모델 로드"""
        try:
            model_id = "EleutherAI/polyglot-ko-1.3b"
            self.get_logger().info(f"모델 로드 중: {model_id}")

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=60,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                return_full_text=False
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            self.get_logger().info("모델 로드 완료")
            return llm
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {e}")
            return None

    def destroy_node(self):
        """노드 종료 시 정리 작업"""
        self.get_logger().info("Agent Node shutdown")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 Agent Node가 중지되었습니다')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
