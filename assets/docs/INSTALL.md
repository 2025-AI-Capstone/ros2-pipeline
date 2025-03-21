## Installation

### WSL 설치
- PowerShell을 관리자 권한으로 실행
- WSL 설치 ```wsl --install```
- 설치 확인 ```wsl -v``` 또는 ```wsl -l -v```

### Docker Desktop 설치
- [홈페이지](https://www.docker.com/)에서 설치 파일 다운로드 및 실행
- Windows 재시작
- Docker Desktop 실행
- 동의(Accept) 및 건너뛰기(Skip)

### Docker 동작 테스트
- PowerShell에서 이미지 다운로드 ```docker pull httpd```
- Docker Desktop의 Images에서 다운로드된 이미지 존재 확인
- PowerShell에서 컨테이너 실행 ```docker run -d -p 8080:80 --name httpd-test httpd```
- Docker Desktop의 Containers에 실행중인 컨테이너 존재 확인
- 브라우저에서 [http://localhost:8080/](http://localhost:8080/) 접속 확인
- Docker Desktop의 Containers에서 컨테이너 Stop
- Docker Desktop의 Containers에서 컨테이너 Delete
- Docker Desktop의 Images에서 이미지 Delete

### VcXsrv Windows X Server 설치
- [홈페이지](https://sourceforge.net/projects/vcxsrv/)에서 설치 파일 다운로드 및 실행
- XLaunch 프로그램 실행
  - 주의: Display number 0
  - 주의: Disable access control 활성화

### wsl2 로컬 카메라 장치 연결
- windows에 연결된 USB 장치는 wsl 내에서 바로 인식 불가
- 관리자 권한으로 연 powershell 창에서 명령어 입력 : ```winget install --interactive --exact dorssel.usbipd-win```
- wsl 창에서 USBIP tools 설치 : ```sudo apt install linux-tools-generic hwdata```
```sudo apt install usbutils```
- 관리자 권한 powershell 창에서 명령어 입력 :  
```usbipd list```
```usbipd bind --busid <busid(확인한 busid 입력)>```
```usbipd attach --wsl --busid <busid(확인한 busid 입력)>```
- wsl 창에서 lsusb로 장치 확인 :
```lsusb```
```ls -l /dev/video*```

### Docker image 빌드 방법

**Dockerfile로 빌드**
- PowerShell에서 Dockerfile 실행 `docker build -t skhuwinter .`
- Docker 실행
- Docker Desktop의 Images에서 다운로드된 이미지 존재 확인

### Docker 컨테이너 실행 확인
- PowerShell에서 컨테이너 실행 ```docker run -it --rm --privileged -p 8080:8501 --gpus all --device=/dev/video0:/dev/video0 -e DISPLAY=host.docker.internal:0.0 -e LIBGL_ALWAYS_INDIRECT=0 --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix:ro --hostname $(hostname) --name test skhuwinter bash```
- 진입한 컨테이너에서 환경 설정 `echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && source ~/.bashrc`
- TurtleSim 실행 `ros2 run turtlesim turtlesim_node`
- 다른 PowerShell을 실행하고 컨테이너에 진입 `docker exec -it test /bin/bash`
- TurtleSim Controller 실행 `ros2 run turtlesim turtle_teleop_key`
- 키보드 방향키로 로봇 움직임 확인

### ROS2-AI-PIPELINE 실행
- PowerShell을 실행하고 컨테이너에 진입 `docker exec -it test /bin/bash`
- 다운로드 `git clone https://github.com/SKHU-AI-2024-WINTER/ros2-ai-pipeline.git`
- 진입 `cd ros2-ai-pipeline`
- 빌드
  - 방식 1(전체 빌드): `colcon build`
  - 방식 2(camera_node 부분 빌드): `colcon build --packages-select camera`
- 설치 `source install/setup.bash`
- 실행
  - 방식 1(별도 명시한 설정파일 사용): `ros2 run camera camera_node --ros-args --params-file src/camera/config/settings.yaml`
  - 방식 2(자동 설치된 설정파일 사용): `ros2 launch camera camera.launch.py` 
  - image_publisher node 실행 : `ros2 launch image_publisher image_publisher.launch.py`
  - detector node 실행 :  `ros2 launch detector detector.launch.py`
  - tracker node 실행 : `ros2 launch tracker tracker.launch.py`
  - falldetector node 실행 : `ros2 launch falldetector falldetector.launch.py` 

  - Streamlit node 실행 : `streamlit run app.py --server.address=0.0.0.0` 
- node 실행 시 parameter 설정 방법
  - 방식 1 (실행할 때 직접 지정):`ros2 run tracker tracker_node --ros-args --param tracker_type:='deepsort'`
  - 방식 2 (setting.yaml 수정 후 실행): `ros2 run tracker tracker_node --ros-args --params-file src/tracker/config/settings.yaml`


- 모든 node embedded로 빌드 후 실행
  - `sh launch/build_and_run.sh`
  - `streamlit run app.py`

