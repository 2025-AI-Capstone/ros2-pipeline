import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gtts import gTTS
import pygame
import tempfile
import os

class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')
        
        # 구독자 설정
        self.subscription = self.create_subscription(
            String,
            'speech_text',
            self.text_callback,
            10
        )
        
        # Pygame 초기화 (오디오 재생용)
        pygame.mixer.init()
        
        # 임시 파일 경로
        self.temp_dir = tempfile.gettempdir()
        self.temp_file = os.path.join(self.temp_dir, 'temp_speech.mp3')
        
        self.get_logger().info('TTS Node is ready')
        
    def text_callback(self, msg):
        try:
            self.get_logger().info(f'Received text: {msg.data}')
            
            # gTTS를 사용하여 텍스트를 음성으로 변환
            tts = gTTS(text=msg.data, lang='ko')
            
            # 임시 파일로 저장
            tts.save(self.temp_file)
            
            # 음성 재생
            pygame.mixer.music.load(self.temp_file)
            pygame.mixer.music.play()
            
            # 재생이 끝날 때까지 대기
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # 임시 파일 삭제
            os.remove(self.temp_file)
            
        except Exception as e:
            self.get_logger().error(f'TTS error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 