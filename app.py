import streamlit as st
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float32
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
import cv2
import numpy as np
import threading
from src.streamlit_node import StreamlitNode
from src.streamlit_node import draw_keypoints


def main():
    rclpy.init()
    node = StreamlitNode()

    executor_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    executor_thread.start()

    st.title("ROS2 Object Detection Stream")
    st.write("실시간 객체 감지 스트리밍...")

    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        image_placeholder = st.empty()
    
    with col2:
        # 서비스 상태 표시 섹션 추가
        st.subheader("서비스 상태")
        status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
        video_status = status_col1.empty()
        detector_status = status_col2.empty()
        camera_status = status_col3.empty()
        falldetector_status = status_col4.empty()
        tracker_status = status_col5.empty()

        st.divider()
        info_placeholder = st.empty()
        latency_placeholder = st.empty()

    log_placeholder = st.empty()

    # Create latency containers
    with latency_placeholder:
        col_inf, col_lat = st.columns(2)
        inf_metric = col_inf.empty()
        lat_metric = col_lat.empty()
    
    while True:
        with node.lock:
            image_data = node.image_data
            detector_data = node.detector_data
            inference_time = node.inference_time
            total_latency = node.total_latency
            keypoints_data = node.keypoints_data
            tracked_objects = node.tracked_objects
            # falldets = node.falldets
            # 서비스 상태
            video_ok = node.video_status
            detector_ok = node.detector_status
            camera_ok = node.camera_status
            fall_ok = node.falldetector_status
            tracker_ok = node.tracker_status

        # 서비스 상태 업데이트
        video_status.metric("video", "working" if video_ok else "error", 
                          delta_color="normal" if video_ok else "inverse")
        detector_status.metric("detector", "working" if detector_ok else "error",
                             delta_color="normal" if detector_ok else "inverse")
        camera_status.metric("camera", "working" if camera_ok else "error",
                           delta_color="normal" if camera_ok else "inverse")
        falldetector_status.metric("falldetector", "working" if fall_ok else "error",
                           delta_color="normal" if fall_ok else "inverse")
        tracker_status.metric("tracker", "working" if tracker_ok else "error",
                           delta_color="normal" if tracker_ok else "inverse")

        if image_data is not None:
            display_image = image_data.copy()
            
            if tracked_objects is not None:
                for obj in tracked_objects:  # 각 obj는 [x1, y1, x2, y2, track_id] 형태
                    try:
                        x1, y1, x2, y2, track_id = obj  # 리스트 언패킹
                        
                        # 좌표를 정수로 변환
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # 트래킹 ID 표시 
                        label = f"ID: {track_id}"
                        label_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
                        cv2.putText(display_image, label, label_position, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    except (ValueError, TypeError) as e:
                        print(f"객체 처리 중 오류 발생: {e}")
                        continue

            
            if keypoints_data is not None:
                display_image = draw_keypoints(display_image, keypoints_data)
                info_placeholder.write(f"감지된 객체 수: {len(keypoints_data)}")
                log_placeholder.write(f"로그: 감지된 객체 수: {len(keypoints_data)}")
            
            else:
                info_placeholder.write("감지된 객체 없음")
            
            # Display latency metrics
            if inference_time is not None:
                inf_metric.metric("추론 시간", f"{inference_time*1000:.1f}ms")
            
            if total_latency is not None:
                lat_metric.metric("전체 지연 시간", f"{total_latency*1000:.1f}ms")
            
            streamlit_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            image_placeholder.image(streamlit_image, channels="RGB", use_container_width=True)
            
        else:
            image_placeholder.write("카메라 스트림 대기 중...")
            info_placeholder.write("이미지 데이터 없음")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.write("애플리케이션 종료")
        rclpy.shutdown()