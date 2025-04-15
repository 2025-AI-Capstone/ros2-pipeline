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
import logging
import datetime

class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        
        # set logging
        self.setup_logging()
        
        # load enviromental variables
        load_dotenv()
        self.logger.info('Environment variables loaded')
        
        # set publisher
        self.publisher = self.create_publisher(String, 'speech_text', 10)
        self.logger.info('Publisher created for topic "speech_text" with queue size 10')
        
        # load openai whisper model(base)
        self.get_logger().info('Loading Whisper model...')
        self.logger.info('Starting to load Whisper "base" model')
        start_time = time.time()
        self.model = whisper.load_model("base")
        load_time = time.time() - start_time
        self.logger.info(f'Whisper model loaded in {load_time:.2f} seconds')
        
        access_key = os.getenv('PICOVOICE_ACCESS_KEY')
        if not access_key:
            self.logger.error('PICOVOICE_ACCESS_KEY environment variable not set')
            raise RuntimeError('PICOVOICE_ACCESS_KEY not set')
            
        self.logger.info('Initializing Porcupine wake word detector')
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=['computer']
            )
            self.logger.info(f'Porcupine initialized successfully with keyword "focus"')
            self.logger.info(f'Porcupine sample rate: {self.porcupine.sample_rate} Hz')
        except Exception as e:
            self.logger.error(f'Failed to initialize Porcupine: {str(e)}')
            raise
        
        # 오디오 설정
        self.CHUNK = 512  # Porcupine Chunk
        self.FORMAT = pyaudio.paInt16  # Porcupine Format
        self.CHANNELS = 1
        self.RATE = self.porcupine.sample_rate  # Porcupine sample rate
        self.RECORD_SECONDS = 10
        self.logger.info(f'Audio configuration: CHUNK={self.CHUNK}, FORMAT=Int16, CHANNELS={self.CHANNELS}, '
                         f'RATE={self.RATE}, RECORD_SECONDS={self.RECORD_SECONDS}')
        
        self.is_listening = False
        self.wake_word_count = 0
        self.transcription_count = 0
        self.publish_count = 0
        
        # Initialize PyAudio
        self.logger.info('Initializing PyAudio')
        self.audio = pyaudio.PyAudio()
        
        self.log_audio_devices()
        
        # start recording thread
        self.logger.info('Starting audio processing thread')
        self.recording_thread = threading.Thread(target=self.process_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.get_logger().info('STT Node is ready. Waiting for wake word "Computer"...')
        self.logger.info('STT Node initialization complete, waiting for wake word')
    
    def setup_logging(self):

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'stt_node_{timestamp}.log'
        
        # 로거 설정
        self.logger = logging.getLogger('stt_node')
        self.logger.setLevel(logging.DEBUG)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f'Logging initialized, writing to {log_filename}')
    
    def log_audio_devices(self):
        """사용 가능한 오디오 입력 장치를 로깅"""
        self.logger.info("Available audio input devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # 입력 장치만 표시
                self.logger.info(f"  Device {i}: {device_info['name']}")
                self.logger.info(f"    Input channels: {device_info['maxInputChannels']}")
                self.logger.info(f"    Default sample rate: {device_info['defaultSampleRate']}")
        
        # 기본 입력 장치 정보
        default_input = self.audio.get_default_input_device_info()
        self.logger.info(f"Default input device: {default_input['name']} (index: {default_input['index']})")
    
    def process_audio(self):
        self.logger.info('Starting audio processing loop')
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            self.logger.info('Audio stream opened successfully')

            while rclpy.ok():
                try:
                    pcm = stream.read(self.CHUNK, exception_on_overflow=False)
                    pcm_unpacked = struct.unpack_from("h" * self.CHUNK, pcm)
                    keyword_index = self.porcupine.process(pcm_unpacked)

                    if keyword_index >= 0:
                        self.wake_word_count += 1
                        self.logger.info(f'Wake word detected! (#{self.wake_word_count}) 시작합니다.')

                        # 리팩토링된 함수로 음성 녹음 및 Whisper 처리
                        transcription = self.listen_and_transcribe(self.RECORD_SECONDS)

                        if transcription:
                            msg = String()
                            msg.data = transcription
                            self.publisher.publish(msg)
                            self.publish_count += 1

                            self.logger.info(f'Published #{self.publish_count}: {transcription}')
                        else:
                            self.logger.warning("텍스트 인식 실패: 결과 없음")
    
                except Exception as e:
                    self.logger.error(f'Error in audio loop: {str(e)}', exc_info=True)

                time.sleep(0.01)

        except Exception as e:
            self.logger.critical(f'Fatal error in audio setup: {str(e)}', exc_info=True)

        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
                self.logger.info('Audio stream closed')

    def listen_and_transcribe(self, record_seconds: int) -> str:

        try:
            self.logger.info(f"{record_seconds}초 동안 녹음 시작")
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            frames = []
            total_chunks = int(self.RATE / self.CHUNK * record_seconds)
            for chunk_idx in range(total_chunks):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            self.logger.info("녹음 종료. Whisper로 처리 시작...")

            # numpy array로 변환
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0

            # Whisper 인식
            result = self.model.transcribe(audio_data)
            text = result["text"].strip()

            self.logger.info(f"인식 결과: {text}")
            return text

        except Exception as e:
            self.logger.error(f"[listen_and_transcribe] 오류 발생: {str(e)}", exc_info=True)
            return ""

    def __del__(self):
        self.logger.info('STT Node shutting down')
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
            self.logger.info('Porcupine instance deleted')
        if hasattr(self, 'audio'):
            self.audio.terminate()
            self.logger.info('PyAudio terminated')
        self.logger.info('STT Node shutdown complete')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = STTNode()
        rclpy.spin(node)
        node.destroy_node()
    except Exception as e:
        logging.critical(f'Unhandled exception in main: {str(e)}', exc_info=True)
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()