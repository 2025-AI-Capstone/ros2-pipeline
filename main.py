import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from src.detector.detector.predictor import SelectivePosePredictor
import torch
from src.falldetector.falldetector.model.model import SimpleNN
from src.tracker.tracker.sort import Sort
import time

# cuda device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize YOLO model
model = YOLO('yolov8n-pose.pt')
model.predictor = SelectivePosePredictor()

tracker = Sort()

# Load fall detection model
input_size = 34
fall_model = SimpleNN(input_size)
fall_model.load_state_dict(torch.load('./src/falldetector/falldetector/checkpoints/model.pt', map_location=device))
fall_model.eval()

# COCO keypoint pairs for skeleton visualization
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
    [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
]

def draw_keypoints(frame, keypoints):
    for person_kpts in keypoints:
        if len(person_kpts) < max(SKELETON[0]) or len(person_kpts) < max(SKELETON[1]):  # Keypoint 수가 부족한 경우 스킵
            continue
        
        # Draw keypoints
        for x, y, conf in person_kpts:
            if conf > 0.5:  # Confidence threshold
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)  # Green dots
                
        # Draw skeleton
        for p1_idx, p2_idx in SKELETON:
            # Ensure that both points exist and have sufficient confidence
            if p1_idx-1 < len(person_kpts) and p2_idx-1 < len(person_kpts):
                p1 = person_kpts[p1_idx-1]
                p2 = person_kpts[p2_idx-1]
                
                if p1[2] > 0.5 and p2[2] > 0.5:  # Check confidence for both points
                    x1, y1 = int(p1[0]), int(p1[1])
                    x2, y2 = int(p2[0]), int(p2[1])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines

    return frame

# Start video capture
cap = cv2.VideoCapture(0)  # 0: 기본 카메라 (USB 카메라 또는 내장 카메라)

# Streamlit GUI
def main():
    st.title("Object Detection Stream")
    st.write("실시간 객체 감지 스트리밍...")

    # Layout setup
    col1, col2 = st.columns([2, 1])
    with col1:
        image_placeholder = st.empty()
        popup_placeholder = st.empty()
    with col2:
        # 서비스 상태 아이콘 정의
        def get_status_icon(status):
            if status == "working":
                return "✅", "color:green;"
            else:
                return "✅", "color:green;"
        st.subheader("서비스 상태")
        video_status, detector_status, camera_status, falldetector_status, tracker_status = st.columns(5)
        video_status = st.empty()
        detector_status = st.empty()
        camera_status = st.empty()
        falldetector_status = st.empty()
        tracker_status = st.empty()

        st.divider()
        tab1, tab2, tab3 = st.tabs(["Statics", "Graph", "Summary"])

        with tab1:
            with st.expander("See explanation"):
                st.write('''
                    The chart above shows some numbers I picked for you.
                    I rolled actual dice for these, so they're *guaranteed* to
                    be random.
                ''')
                st.image("https://static.streamlit.io/examples/dice.jpg")
            
        with tab2:
            row1 = st.columns(2)
            row2 = st.columns(2)

            for col in row1 + row2:
                tile = col.container(height=150)
                tile.title(" ")
        with tab3:
            with st.expander("See explanation"):
                st.write('''
                    The chart above shows some numbers I picked for you.
                    I rolled actual dice for these, so they're *guaranteed* to
                    be random.
                ''')
                st.image("https://static.streamlit.io/examples/dice.jpg")


    col3, col4, col5 = st.columns([3, 5, 4])
        
    with col3:

        with st.container(height=300):
            image_placeholder_col3 = st.empty()

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    with col5:
        st.checkbox("Disable text input widget", key="disabled")
        st.radio(
            "Set text input label visibility 👉",
            key="visibility",
            options=["visible", "hidden", "collapsed"],
        )
        st.text_input(
            "Placeholder for the other text input widget",
            "This is a placeholder",
            key="placeholder",
        )   

    with col4:

        st.subheader("ID / Fall Status")
        log_placeholder = st.empty()       
        log_placeholder_2 = st.empty()
    
    while True:

        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            st.write("카메라 스트림 대기 중...")
            continue  # 카메라 프레임을 읽지 못한 경우, 루프를 계속 돌며 대기

        # YOLO Object Detection & Tracking
        results = model(frame)
        detections = []

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints.data

            # Get bounding boxes and keypoints
            for box in boxes:
                if box.cls == 0:  # 사람이 감지된 경우
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])  # Confidence Score
                    detections.append([x1, y1, x2, y2, conf])

        if detections is not None and len(detections) > 0:
            # Apply SORT tracker
            tracked_objects = tracker.update(np.array(detections))

            # Draw bounding boxes and track IDs
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, track)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(0, x2)
                y2 = max(0, y2)    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                roi = frame[y1:y2, x1:x2]
                resized_roi = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2))  # Adjust the scale factor as needed
                resized_streamlit_image = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
                image_placeholder_col3.image(resized_streamlit_image, channels="RGB", use_container_width=False)
                label = f"ID: {track_id}"
                cv2.putText(frame, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                log_placeholder.write(f" {label}")

        # Draw Keypoints if available
        if keypoints is not None and len(keypoints) > 0 and len(detections) > 0:
            print(keypoints.shape)
            print(keypoints[0])
            keypoints_xy = keypoints[..., :2]
            non_zero_count = np.count_nonzero(keypoints_xy)
            if non_zero_count > 13:
                keypoints_resized = keypoints_xy.reshape(keypoints_xy.shape[0], -1)
                print(keypoints_resized.shape)
                keypoints_resized = keypoints_resized.float().to(device)
                keypoints_resized_clone = keypoints_resized.clone().detach()
                falldets = fall_model(keypoints_resized_clone)
            else : falldets = None
            frame = draw_keypoints(frame, keypoints)

            if falldets is not None:
                print(falldets)
                for i, det in enumerate(falldets):
                    # Fall detection 상태를 바운딩 박스 위에 표시
                    if det > 0.5:
                        fall_status = "Fall Detected"
                        color = (255, 0, 255)  # Red for Fall Detected
                        log_placeholder_2.write("Fall Detected")
                        popup_placeholder.warning("FALL DETECTED!", icon="⚠️")
                        
                    else:
                        fall_status = "Normal"
                        color = (0, 255, 0)  # Green for Normal
                        log_placeholder_2.write("Normal")
                        popup_placeholder.empty()

                    text_size = cv2.getTextSize(fall_status, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.putText(frame, fall_status, (int(detections[i][2] - text_size[0]), int(detections[i][1] - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                log_placeholder_2.write("Normal")
                popup_placeholder.empty()
        # Convert frame for Streamlit display
        streamlit_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display video feed in Streamlit
        image_placeholder.image(streamlit_image, channels="RGB", use_container_width=True)
                        
        # Update the status and metrics
        #video_status.metric("video", "✅", delta_color="normal")
        #detector_status.metric("detector", "✅", delta_color="normal")
        #camera_status.metric("camera", "✅", delta_color="normal")
        #falldetector_status.metric("falldetector", "✅", delta_color="normal")
        #tracker_status.metric("tracker", "✅", delta_color="normal")

        # 상태 아이콘 및 색상 업데이트
        video_icon, video_color = get_status_icon("working")
        detector_icon, detector_color = get_status_icon("working")
        camera_icon, camera_color = get_status_icon("working")
        falldetector_icon, falldetector_color = get_status_icon("working")
        tracker_icon, tracker_color = get_status_icon("working")
        
        video_status.markdown(f"<span style='{video_color}'>{video_icon} Video</span>", unsafe_allow_html=True)
        detector_status.markdown(f"<span style='{detector_color}'>{detector_icon} Detector</span>", unsafe_allow_html=True)
        camera_status.markdown(f"<span style='{camera_color}'>{camera_icon} Camera</span>", unsafe_allow_html=True)
        falldetector_status.markdown(f"<span style='{falldetector_color}'>{falldetector_icon} Falldetector</span>", unsafe_allow_html=True)
        tracker_status.markdown(f"<span style='{tracker_color}'>{tracker_icon} Tracker</span>", unsafe_allow_html=True)

    # Release the capture once the loop ends
    cap.release()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.write("애플리케이션 종료")
        cap.release()