import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gtts import gTTS
import pygame
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path

class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')

        # LangGraph 혹은 LLM 응답 텍스트 구독
        self.subscription = self.create_subscription(
            String,
            'agent/response',
            self.text_callback,
            10
        )

        # Pygame 초기화
        pygame.mixer.init()

        # 임시 mp3 저장 경로
        self.temp_dir = tempfile.gettempdir()
        self.temp_file = os.path.join(self.temp_dir, 'temp_speech.mp3')

        # 루틴 경로 및 데이터 로딩
        self.routine_file_path = Path(__file__).parent / 'routine_data.json'
        self.routines = self.load_routines()
        self.announced_times = set()  # 루틴 중복 방지

        # 타이머: 루틴 확인 주기 30초
        self.timer = self.create_timer(30.0, self.check_routines)

        self.get_logger().info('TTS Node is ready')

    def load_routines(self):
        if not self.routine_file_path.exists():
            self.get_logger().warn(f"No routine_data.json found at {self.routine_file_path}")
            return []
        try:
            with open(self.routine_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f'Error loading routine data: {str(e)}')
            return []

    def check_routines(self):
        now = datetime.now().strftime('%H:%M')
        for routine in self.routines:
            routine_time = routine.get('time')
            routine_text = routine.get('text')

            if routine_time == now and routine_time not in self.announced_times:
                self.get_logger().info(f'⏰ Routine triggered: {routine_text}')
                self.announced_times.add(routine_time)
                self.speak(routine_text)

    def text_callback(self, msg):
        try:
            self.get_logger().info(f'Received text: {msg.data}')
            self.speak(msg.data)
        except Exception as e:
            self.get_logger().error(f'TTS error: {str(e)}')

    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='ko')
            tts.save(self.temp_file)

            pygame.mixer.music.load(self.temp_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            os.remove(self.temp_file)

        except Exception as e:
            self.get_logger().error(f'TTS speak error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
