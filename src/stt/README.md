# STT Node

ROS2 Speech-to-Text Node with Wake Word Detection

## 환경 설정

1. `.env` 파일 생성:
```bash
cp .env.example .env
```

2. Picovoice 액세스 키 설정:
- https://console.picovoice.ai/ 에서 계정 생성
- 액세스 키 발급
- `.env` 파일에 발급받은 키 입력:
```
PICOVOICE_ACCESS_KEY=your_access_key_here
```

## 의존성 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
# ROS2 워크스페이스에서
colcon build --packages-select stt
source install/setup.bash
ros2 run stt stt
```

## 사용 방법

1. "포커스" 라고 말하면 wake word가 감지됨
2. Wake word 감지 후 5초 동안 음성 입력
3. 입력된 음성이 텍스트로 변환되어 'speech_text' 토픽으로 발행됨 