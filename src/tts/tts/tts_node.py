import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty
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

        self.subscription = self.create_subscription(
            String,
            'agent/response',
            self.text_callback,
            10
        )
        self.alert_subscription = self.create_subscription(
            String,
            'agent/alert',
            self.alert_callback,
            10
        )
        self.stt_trigger_pub = self.create_publisher(Empty, 'agent/trigger', 10)
        # Pygame 초기화
        pygame.mixer.init()

        # 임시 mp3 저장 경로
        self.temp_dir = tempfile.gettempdir()
        self.temp_file = os.path.join(self.temp_dir, 'temp_speech.mp3')


        self.get_logger().info('TTS Node is ready')


    def text_callback(self, msg):
        try:
            self.get_logger().info(f'Received text: {msg.data}')
            self.speak(msg.data)
        except Exception as e:
            self.get_logger().error(f'TTS error: {str(e)}')

    def alert_callback(self, msg):
        try:
            self.get_logger().info(f'Received text: {msg.data}')
            self.speak_alert(msg.data)
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

    def speak_alert(self, text):
        try:
            tts = gTTS(text=text, lang='ko')
            tts.save(self.temp_file)

            pygame.mixer.music.load(self.temp_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            os.remove(self.temp_file)
            self.stt_trigger_pub.publish(Empty())
            
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
