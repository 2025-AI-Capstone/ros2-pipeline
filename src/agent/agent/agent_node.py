import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from std_srvs.srv import SetBool
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
        # 구독 설정
        self.subscription = self.create_subscription(
            String,
            'stt/speech_text',
            self.text_callback,
            10
        )
        
        # 발행자 설정
        self.response_publisher = self.create_publisher(String, 'agent/response', 10)
                
        
        self.get_logger().info("Agent Node 초기화 완료")
        
    def text_callback(self, msg):
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
            
        except Exception as e:
            self.get_logger().error(f"응답 생성 오류: {str(e)}")
    
    def destroy_node(self):
        """노드 종료 시 정리 작업"""
        self.get_logger().info("노드 정리 작업 수행")
        super().destroy_node()


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
                temperature=0.7
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            self.get_logger().info("모델 로드 완료")
            return llm
            
        except Exception as e:
            self.get_logger().error(f"모델 로드 실패: {str(e)}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 에이전트 노드가 중지되었습니다')
    except Exception as e:
        node.get_logger().error(f'예외 발생: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()