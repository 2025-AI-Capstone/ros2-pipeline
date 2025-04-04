import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import pyaudio
import wave
import numpy as np
import threading
import time
import pvporcupine
import struct
import os
from dotenv import load_dotenv

class STTNode(Node):
    def __init__(self):
        super().__init__('stt')
        
        # 환경변수 로드
        load_dotenv()
        
        # 퍼블리셔 설정
        self.publisher = self.create_publisher(String, 'speech_text', 10)
        
        # Whisper 모델 로드
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")
        
        # Porcupine 초기화 (wake word: "포커스")
        access_key = os.getenv('PICOVOICE_ACCESS_KEY')
        if not access_key:
            self.get_logger().error('PICOVOICE_ACCESS_KEY environment variable not set')
            raise RuntimeError('PICOVOICE_ACCESS_KEY not set')
            
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=['focus']
        )
        
        # 오디오 설정
        self.CHUNK = 512  # Porcupine 권장 크기
        self.FORMAT = pyaudio.paInt16  # Porcupine 요구사항
        self.CHANNELS = 1
        self.RATE = self.porcupine.sample_rate  # Porcupine 샘플레이트 사용
        self.RECORD_SECONDS = 5
        
        # 상태 변수
        self.is_listening = False
        
        # PyAudio 초기화
        self.audio = pyaudio.PyAudio()
        
        # 녹음 스레드 시작
        self.recording_thread = threading.Thread(target=self.process_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.get_logger().info('STT Node is ready. Waiting for wake word "포커스"...')
    
    def process_audio(self):
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        while rclpy.ok():
            # 오디오 데이터 읽기
            pcm = stream.read(self.CHUNK)
            pcm = struct.unpack_from("h" * self.CHUNK, pcm)
            
            # Wake word 감지
            keyword_index = self.porcupine.process(pcm)
            
            if keyword_index >= 0:
                self.get_logger().info('Wake word detected! Starting to listen...')
                self.is_listening = True
                
                # 음성 녹음 시작
                frames = []
                for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                    data = stream.read(self.CHUNK)
                    frames.append(data)
                
                self.get_logger().info('Recording finished. Processing...')
                
                # 녹음된 데이터를 numpy array로 변환
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0  # 정규화
                
                # Whisper로 음성 인식
                try:
                    result = self.model.transcribe(audio_data)
                    text = result["text"].strip()
                    
                    if text:
                        # 인식된 텍스트 퍼블리시
                        msg = String()
                        msg.data = text
                        self.publisher.publish(msg)
                        self.get_logger().info(f'Published: {text}')
                
                except Exception as e:
                    self.get_logger().error(f'Transcription error: {str(e)}')
                
                self.is_listening = False
                self.get_logger().info('Waiting for wake word...')
            
            time.sleep(0.01)  # CPU 부하 감소
    
    def __del__(self):
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
        if hasattr(self, 'audio'):
            self.audio.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 