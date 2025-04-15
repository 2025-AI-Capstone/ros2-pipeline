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
        
        # 로깅 설정
        self.setup_logging()
        
        # 환경변수 로드
        load_dotenv()
        self.logger.info('Environment variables loaded')
        
        # 퍼블리셔 설정
        self.publisher = self.create_publisher(String, 'speech_text', 10)
        self.logger.info('Publisher created for topic "speech_text" with queue size 10')
        
        # Whisper 모델 로드
        self.get_logger().info('Loading Whisper model...')
        self.logger.info('Starting to load Whisper "base" model')
        start_time = time.time()
        self.model = whisper.load_model("base")
        load_time = time.time() - start_time
        self.logger.info(f'Whisper model loaded in {load_time:.2f} seconds')
        
        # Porcupine 초기화 (wake word: "포커스")
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
        self.CHUNK = 512  # Porcupine 권장 크기
        self.FORMAT = pyaudio.paInt16  # Porcupine 요구사항
        self.CHANNELS = 1
        self.RATE = self.porcupine.sample_rate  # Porcupine 샘플레이트 사용
        self.RECORD_SECONDS = 10
        self.logger.info(f'Audio configuration: CHUNK={self.CHUNK}, FORMAT=Int16, CHANNELS={self.CHANNELS}, '
                         f'RATE={self.RATE}, RECORD_SECONDS={self.RECORD_SECONDS}')
        
        # 상태 변수
        self.is_listening = False
        self.wake_word_count = 0
        self.transcription_count = 0
        self.publish_count = 0
        
        # PyAudio 초기화
        self.logger.info('Initializing PyAudio')
        self.audio = pyaudio.PyAudio()
        
        # 사용 가능한 입력 장치 로깅
        self.log_audio_devices()
        
        # 녹음 스레드 시작
        self.logger.info('Starting audio processing thread')
        self.recording_thread = threading.Thread(target=self.process_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.get_logger().info('STT Node is ready. Waiting for wake word "포커스"...')
        self.logger.info('STT Node initialization complete, waiting for wake word')
    
    def setup_logging(self):
        # 로그 파일명에 날짜와 시간 포함
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
                # 오디오 데이터 읽기
                try:
                    pcm = stream.read(self.CHUNK, exception_on_overflow=False)
                    pcm_unpacked = struct.unpack_from("h" * self.CHUNK, pcm)
                    
                    # 오디오 레벨 주기적으로 로깅 (CPU 사용량을 고려하여 로깅 빈도 조절)
                    if self.wake_word_count % 100 == 0:
                        audio_level = np.abs(np.array(pcm_unpacked, dtype=np.int16)).mean()
                        self.logger.debug(f'Current audio level: {audio_level:.2f}')
                    
                    # Wake word 감지
                    keyword_index = self.porcupine.process(pcm_unpacked)
                    
                    if keyword_index >= 0:
                        self.wake_word_count += 1
                        self.logger.info(f'Wake word detected! (Count: {self.wake_word_count}) Starting to listen...')
                        self.is_listening = True
                        self.get_logger().info('Wake word detected! Starting to listen...')
                        
                        # 시작 시간 기록
                        recording_start_time = time.time()
                        self.logger.info(f'Recording started at {datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}')
                        
                        # 음성 녹음 시작
                        frames = []
                        total_chunks = int(self.RATE / self.CHUNK * self.RECORD_SECONDS)
                        for chunk_idx in range(total_chunks):
                            data = stream.read(self.CHUNK, exception_on_overflow=False)
                            frames.append(data)
                            
                            # 진행 상황 로깅 (10% 간격으로)
                            if chunk_idx % int(total_chunks / 10) == 0:
                                progress = int((chunk_idx / total_chunks) * 100)
                                self.logger.debug(f'Recording progress: {progress}%')
                        
                        recording_duration = time.time() - recording_start_time
                        self.logger.info(f'Recording finished in {recording_duration:.2f} seconds. Processing...')
                        self.get_logger().info('Recording finished. Processing...')
                        
                        # 녹음된 데이터를 numpy array로 변환
                        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                        self.logger.debug(f'Raw audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}')
                        
                        # 오디오 통계 로깅
                        audio_min = np.min(audio_data)
                        audio_max = np.max(audio_data)
                        audio_mean = np.mean(np.abs(audio_data))
                        self.logger.info(f'Audio statistics - Min: {audio_min}, Max: {audio_max}, Mean abs: {audio_mean:.2f}')
                        
                        # 정규화
                        audio_data = audio_data.astype(np.float32) / 32768.0
                        self.logger.debug(f'Normalized audio data range: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]')
                        
                        # Whisper로 음성 인식
                        try:
                            self.logger.info('Starting Whisper transcription')
                            transcription_start_time = time.time()
                            
                            result = self.model.transcribe(audio_data)
                            
                            transcription_duration = time.time() - transcription_start_time
                            self.transcription_count += 1
                            
                            text = result["text"].strip()
                            self.logger.info(f'Transcription #{self.transcription_count} completed in {transcription_duration:.2f} seconds')
                            self.logger.info(f'Raw transcription result: "{text}"')
                            
                            # 세그먼트 정보 로깅
                            if "segments" in result:
                                self.logger.debug(f'Number of segments: {len(result["segments"])}')
                                for i, segment in enumerate(result["segments"]):
                                    self.logger.debug(f'Segment {i+1}: {segment.get("text", "").strip()} '
                                                     f'(conf: {segment.get("confidence", 0):.4f}, '
                                                     f'start: {segment.get("start", 0):.2f}s, '
                                                     f'end: {segment.get("end", 0):.2f}s)')
                            
                            if text:
                                # 인식된 텍스트 퍼블리시
                                msg = String()
                                msg.data = text
                                publish_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                self.publisher.publish(msg)
                                self.publish_count += 1
                                
                                self.logger.info(f'Published message #{self.publish_count} at {publish_time}')
                                self.logger.info(f'Published text: "{text}"')
                                self.get_logger().info(f'Published: {text}')
                            else:
                                self.logger.warning('Transcription produced empty text, nothing published')
                        
                        except Exception as e:
                            self.logger.error(f'Transcription error: {str(e)}', exc_info=True)
                            self.get_logger().error(f'Transcription error: {str(e)}')
                        
                        self.is_listening = False
                        self.logger.info('Finished processing, waiting for next wake word')
                        self.get_logger().info('Waiting for wake word...')
                
                except Exception as e:
                    self.logger.error(f'Error in audio processing loop: {str(e)}', exc_info=True)
                
                time.sleep(0.01)  # CPU 부하 감소
        
        except Exception as e:
            self.logger.critical(f'Fatal error in audio processing: {str(e)}', exc_info=True)
        finally:
            self.logger.info('Audio processing thread stopping')
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
                self.logger.info('Audio stream closed')
    
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