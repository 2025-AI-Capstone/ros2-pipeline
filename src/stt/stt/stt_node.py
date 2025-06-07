import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty
import whisper
import pyaudio
import numpy as np
import threading
import time
import pvporcupine
import struct
import os
from dotenv import load_dotenv
import logging
import datetime

class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')

        # 로깅 설정
        self.setup_logging()

        # 환경변수 로딩
        load_dotenv()
        self.logger.info('Environment variables loaded')

        # 발화 텍스트 퍼블리시 토픽
        self.publisher = self.create_publisher(String, 'stt/speech_text', 10)

        # 외부 트리거 수신
        self.trigger_sub = self.create_subscription(
            Empty,
            'agent/trigger',
            self.trigger_callback,
            10
        )

        # Whisper 모델 로드
        start_time = time.time()
        self.model = whisper.load_model("base")
        load_time = time.time() - start_time
        self.logger.info(f'Whisper model loaded in {load_time:.2f} seconds')

        # 사용자 정의 wakeword 경로 지정
        access_key = os.getenv('PICOVOICE_ACCESS_KEY')
        keyword_path = os.getenv('WAKEWORD_PATH', './src/stt/wake_word/포커스_ko_linux_v3_0_0.ppn')

        if not access_key:
            self.logger.error('PICOVOICE_ACCESS_KEY not set')
            raise RuntimeError('PICOVOICE_ACCESS_KEY not set')

        if not os.path.exists(keyword_path):
            self.logger.error(f'Wakeword .ppn file not found: {keyword_path}')
            raise FileNotFoundError(f'Wakeword file not found: {keyword_path}')

        # Porcupine 초기화
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[keyword_path]
            )
            self.logger.info(f'Porcupine initialized with keyword file: {keyword_path}')
        except Exception as e:
            self.logger.error(f'Failed to initialize Porcupine: {e}')
            raise

        # 오디오 설정
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = self.porcupine.sample_rate
        self.RECORD_SECONDS = 7

        self.wake_word_count = 0
        self.publish_count = 0

        # PyAudio 및 스레드 실행
        self.audio = pyaudio.PyAudio()
        self.recording_thread = threading.Thread(target=self.process_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        self.logger.info('STT Node initialized and listening')

    def setup_logging(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'stt_node_{timestamp}.log'
        self.logger = logging.getLogger('stt_node')
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.info(f'Logging initialized to {log_filename}')

    def trigger_callback(self, msg: Empty):
        self.logger.info('STT trigger received')
        transcription = self.listen_and_transcribe(self.RECORD_SECONDS)
        if transcription:
            out = String()
            out.data = transcription
            self.publisher.publish(out)
            self.publish_count += 1
            self.logger.info(f'Published triggered STT #{self.publish_count}: {transcription}')

    def process_audio(self):
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            self.logger.info('Audio stream opened for wakeword detection')
            while rclpy.ok():
                pcm = stream.read(self.CHUNK, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("h" * self.CHUNK, pcm)
                keyword_index = self.porcupine.process(pcm_unpacked)
                if keyword_index >= 0:
                    self.wake_word_count += 1
                    self.logger.info(f'Wake word detected #{self.wake_word_count}')
                    transcription = self.listen_and_transcribe(self.RECORD_SECONDS)
                    if transcription:
                        msg = String()
                        msg.data = transcription
                        self.publisher.publish(msg)
                        self.publish_count += 1
                        self.logger.info(f'Published wake STT #{self.publish_count}: {transcription}')
                time.sleep(0.01)
        except Exception as e:
            self.logger.critical(f'Audio processing error: {e}', exc_info=True)
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
                self.logger.info('Audio stream closed')

    def listen_and_transcribe(self, record_seconds: int) -> str:
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            frames = []
            total_chunks = int(self.RATE / self.CHUNK * record_seconds)
            for _ in range(total_chunks):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            stream.stop_stream()
            stream.close()
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_data)
            text = result.get('text', '').strip()
            self.logger.info(f'Transcription result: {text}')
            return text
        except Exception as e:
            self.logger.error(f'Transcription error: {e}', exc_info=True)
            return ""

    def __del__(self):
        self.logger.info('STT Node shutting down')
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        self.logger.info('Cleanup complete')


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
