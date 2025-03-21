import streamlit as st
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
import cv2
import numpy as np
import threading
import message_filters

class StreamlitNode(Node):

    def __init__(self):
        super().__init__('streamlit_node')

        self.cv_bridge = CvBridge()
        self.lock = threading.Lock()

        # 데이터 저장용 변수
        self.image_data = None
        self.detector_data = None
        self.keypoints_data = None
        # self.falldets = None
        self.tracked_objects = None
        self.inference_time = None
        self.total_latency = None

        # ROS2 Subscriber 설정 (message_filters 활용)
        image_sub = message_filters.Subscriber(self, Image, 'video_publisher/frames')
        bbox_sub = message_filters.Subscriber(self, Detection2DArray, 'detector/bboxes')
        keypoints_sub = message_filters.Subscriber(self, JointState, 'detector/keypoints')
        falldet_sub = message_filters.Subscriber(self, JointState, 'falldetector/falldets')
        tracked_sub = message_filters.Subscriber(self, JointState, 'tracker/tracked_objects') 

        # 동기화 설정: 비슷한 타임스탬프의 메시지만 처리
        sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, bbox_sub, keypoints_sub, tracked_sub],
            queue_size=10, slop=0.5
        )
        sync.registerCallback(self.synced_callback)

        # 개별 데이터 (지연 허용)
        self.subscription_inference = self.create_subscription(
            Float32, 'detector/inference_time', self.inference_callback, 10
        )
        self.subscription_total = self.create_subscription(
            Float32, 'detector/total_latency', self.latency_callback, 10
        )
        # initialize service status
        self.video_status = True
        self.detector_status = True
        self.camera_status = True
        self.falldetector_status = True
        self.tracker_status = True

        # ros2 service client
        self.video_client = self.create_client(SetBool, 'video_publisher/check_video')
        self.detector_client = self.create_client(SetBool, 'detector/check_detector')
        self.camera_client = self.create_client(SetBool, 'camera/check_camera')
        self.falldetector_client = self.create_client(SetBool, 'falldetector/check_fall')
        self.tracker_client = self.create_client(SetBool, 'tracker/check_tracker')
        # initailize data variables
        self.cv_bridge = CvBridge()
        self.image_data = None
        self.detector_data = None
        self.keypoints_data = None
        self.inference_time = None
        self.total_latency = None
        self.tracked_objects = None
        self.falldets = None
        self.lock = threading.Lock()
        self.create_timer(1.0, self.check_services_status)
    
    def check_services_status(self):
        """주기적으로 각 서비스의 상태를 확인"""
        self.check_service_status('video', self.video_client)
        self.check_service_status('detector', self.detector_client)
        self.check_service_status('camera', self.camera_client)
        self.check_service_status('falldetector', self.falldetector_client)
        self.check_service_status('tracker', self.tracker_client)

    def check_service_status(self, name, client):
        """개별 서비스의 상태를 확인하고 업데이트"""
        if not client.wait_for_service(timeout_sec=0.1):
            with self.lock:
                setattr(self, f'{name}_status', False)
            return
            
        request = SetBool.Request()
        request.data = True
        future = client.call_async(request)
        
        # 비동기 응답 처리를 위한 콜백 설정
        future.add_done_callback(
            lambda f: self.service_callback(f, name))

    def service_callback(self, future, name):
        """서비스 응답 처리"""
        try:
            response = future.result()
            setattr(self, f'{name}_status', response.success)
        except Exception as e:
            self.get_logger().error(f'Service call failed for {name}: {str(e)}')
            setattr(self, f'{name}_status', False)

    def synced_callback(self, image_msg, bbox_msg, keypoints_msg, tracked_msg):
        """ 동기화된 데이터를 한 번에 처리하는 콜백 함수 """
        try:
            # ROS2 Image → OpenCV 변환
            image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # 바운딩 박스 데이터 변환
            detections = []
            for detection in bbox_msg.detections:
                x1, y1, x2, y2, conf = detection
                detections.append([x1, y1, x2, y2])

            # 키포인트 데이터 변환
            keypoints = np.array(keypoints_msg.position).reshape(-1, 17, 3) if keypoints_msg.position else None

            '''
            # Fall Detection 결과 변환
            if falldets_msg.name and falldets_msg.position:
                fall_keypoints = np.array(falldets_msg.position).reshape(-1, 17, 3)  # (N, 17, 3) 형태로 변환
                fall_labels = np.array([label for label in falldets_msg.name]) 
                falldets = np.concatenate([fall_keypoints.reshape(-1), fall_labels])
            else:
                falldets = None
            '''
            tracked_objects = []
            for i, track_id in enumerate(tracked_msg.name):  # ID 리스트
                x1, y1, x2, y2 = tracked_msg.position[i * 4:(i + 1) * 4]  # BBox 좌표 4개씩 추출
                tracked_objects.append([x1, y1, x2, y2, int(track_id)])  # ID 포함

            with self.lock:
                self.image_data = image
                self.detector_data = detections
                self.keypoints_data = keypoints
                # self.falldets = falldets
                self.tracked_objects = tracked_objects

        except Exception as e:
            self.get_logger().error(f"Failed to process synced data: {e}")


    def inference_callback(self, msg):
        """ 모델 추론 시간 저장 """
        try:
            with self.lock:
                self.inference_time = msg.data
        except Exception as e:
            self.get_logger().error(f"Failed to process inference time: {e}")

    def latency_callback(self, msg):
        """ 전체 지연 시간 저장 """
        try:
            with self.lock:
                self.total_latency = msg.data
        except Exception as e:
            self.get_logger().error(f"Failed to process latency: {e}")

def main():
    rclpy.init()
    node = StreamlitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

def draw_keypoints(image, keypoints):
    '''
    draw COCO keypoints on image
    '''
    skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    for person_kpts in keypoints:
        # Draw points
        for x, y, conf in person_kpts:
            if conf > 0.5:  # Confidence threshold
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Draw skeleton
        for p1_idx, p2_idx in skeleton:
            p1 = person_kpts[p1_idx-1]
            p2 = person_kpts[p2_idx-1]
            if p1[2] > 0.5 and p2[2] > 0.5:  # Both points confident
                cv2.line(image, 
                        (int(p1[0]), int(p1[1])), 
                        (int(p2[0]), int(p2[1])), 
                        (0, 0, 255), 2)
    return image
'''
def draw_keypoints(image, falldets, threshold=0.5):

    # COCO keypoints connection pairs for visualization
    skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    
    if falldets is None:
        return image 

    try:
        keypoints_data = np.array(falldets[1]).reshape(-1, 17, 3)  # position 데이터
        labels = falldets[0]  # name 데이터
        
        for i, person_kpts in enumerate(keypoints_data):
            fall_status = labels[i] # 0: 정상, 1: 쓰러짐
            keypoint_color = (0, 255, 0) if fall_status == "NORMAL" else (0, 0, 255)
            skeleton_color = (0, 255, 0) if fall_status == "NORMAL" else (0, 0, 255)
            label_color = (0, 255, 0) if fall_status == "NORMAL" else (0, 0, 255)

            label_x, label_y = None, None

        # 키포인트 그리기
        for idx, (x,y, conf) in enumerate(person_kpts):
            if conf > threshold:  # Proceed with comparison
                cv2.circle(image, (int(x), int(y)), 3, keypoint_color, -1)
                if idx == 0:  # 머리 (nose) 위치에 라벨 배치
                    label_x, label_y = int(x), int(y) - 10

            # 스켈레톤 그리기
            for p1_idx, p2_idx in skeleton:
                p1 = person_kpts[p1_idx - 1]
                p2 = person_kpts[p2_idx - 1]

                if p1[2] > threshold and p2[2] > threshold:  # 신뢰도 기준
                    cv2.line(image, (int(p1[0]), int(p1[1])), 
                             (int(p2[0]), int(p2[1])), skeleton_color, 2)

            # 상태 레이블 표시
            if label_x is not None and label_y is not None:
                cv2.putText(image, fall_status, (label_x, label_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2, cv2.LINE_AA)

    except Exception as e:
        print(f"[ERROR] Failed to draw keypoints: {e}")

    return image
'''