#!/bin/bash

# 1. 빌드 및 환경 설정
echo "빌드를 시작합니다..."
colcon build --symlink-install
if [ $? -ne 0 ]; then
  echo "빌드 실패!"
  exit 1
fi
echo "빌드 완료!"

echo "환경 설정을 적용합니다..."
source install/setup.bash
if [ $? -ne 0 ]; then
  echo "환경 설정 실패!"
  exit 1
fi
echo "환경 설정 완료!"

# 2. 노드 실행 (순서 주의)
echo "노드를 실행합니다..."
ros2 run video_publisher video_publisher &
video_publisher_pid=$!
echo "video_publisher 노드 실행 (PID: $video_publisher_pid)"

ros2 run detector detector &
detector_pid=$!
echo "detector 노드 실행 (PID: $detector_pid)"

ros2 run tracker tracker &
tracker_pid=$!
echo "tracker 노드 실행 (PID: $tracker_pid)"

ros2 run falldetector falldetector &
falldetector_pid=$!
echo "falldetector 노드 실행 (PID: $falldetector_pid)"

# 3. 노드 실행 결과 확인 및 에러 처리
trap "kill $video_publisher_pid $detector_pid $tracker_pid $falldetector_pid; exit 1" INT TERM ERR

# 4. 백그라운드 작업들이 종료될 때까지 대기
wait

echo "모든 노드가 정상적으로 종료되었습니다."

exit 0