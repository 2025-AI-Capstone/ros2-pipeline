import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
from custom_msgs.msg import CustomDetection2D, CustomBoolean
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String, Bool
import message_filters
import base64
import json
import numpy as np
import requests
import time

class Regenerator(Node):
    def __init__(self):
        super().__init__('regenerator_node')

        self.cv_bridge = CvBridge()

        # === 동기화된 데이터 (dashboard 발행용) ===
        image_sub = message_filters.Subscriber(self, Image, 'camera/image_raw')
        bbox_sub = message_filters.Subscriber(self, CustomDetection2D, 'detector/bboxes')
        keypoints_sub = message_filters.Subscriber(self, JointState, 'detector/keypoints')
        fall_sub = message_filters.Subscriber(self, CustomBoolean, 'falldetector/falldets')

        sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, bbox_sub, keypoints_sub, fall_sub],
            queue_size=10, slop=0.2
        )
        sync.registerCallback(self.synced_callback)

        # === 개별 메시지 수신 (상태 추적용) ===
        self.create_subscription(Image, 'camera/image_raw', self.update_camera_time, 10)
        self.create_subscription(CustomDetection2D, 'detector/bboxes', self.update_detector_time, 10)
        self.create_subscription(CustomBoolean, 'falldetector/falldets', self.update_falldetector_time, 10)

        # Dashboard 전송
        self.dashboard_pub = self.create_publisher(String, 'dashboard/data', 10)

        # 마지막 메시지 수신 시간
        self.last_msg_time = {
            'camera': 0,
            'detector': 0,
            'falldetector': 0
        }

        # 상태 전송 설정
        self.api_endpoint = "http://localhost:8000/system-statuses"
        self.event_id = 1

        # 5초마다 상태 판단 및 API 전송
        self.create_timer(5.0, self.check_msg_received_recently)

    def synced_callback(self, image_msg, bbox_msg, keypoint_msg, fall_msg):
        try:
            # 이미지 처리
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # 바운딩 박스 처리
            serialized_bbox = []
            for i in range(0, len(bbox_msg.detections.data), 5):
                x1, y1, x2, y2, conf = bbox_msg.detections.data[i:i+5]
                serialized_bbox.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf})

            # 키포인트 처리
            keypoints_data = np.array(keypoint_msg.position).reshape(-1, 17, 3)

            # 이미지 인코딩
            _, buffer = cv2.imencode('.jpg', cv_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Dashboard JSON 구성
            dashboard_data = {
                'image': image_base64,
                'bboxes': serialized_bbox,
                'keypoints': keypoints_data.tolist(),
                'fall_detection': fall_msg.is_fall.data
            }

            dashboard_msg = String()
            dashboard_msg.data = json.dumps(dashboard_data)
            self.dashboard_pub.publish(dashboard_msg)

        except Exception as e:
            self.get_logger().error(f"[sync_callback] Error: {str(e)}")

    # === 개별 수신 시각 갱신 콜백 ===
    def update_camera_time(self, msg):
        self.last_msg_time['camera'] = time.time()

    def update_detector_time(self, msg):
        self.last_msg_time['detector'] = time.time()

    def update_falldetector_time(self, msg):
        self.last_msg_time['falldetector'] = time.time()

    # === 5초마다 상태 판단 및 전송 ===
    def check_msg_received_recently(self):
        now = time.time()
        for node_name in self.last_msg_time:
            last_time = self.last_msg_time[node_name]
            status = "active" if now - last_time < 5 else "inactive"
            self.send_node_status(node_name, status)

    def send_node_status(self, node_name, status):
        try:
            payload = {
                "event_id": self.event_id,
                "node_name": node_name,
                "status": status
            }
            response = requests.post(self.api_endpoint, json=payload, timeout=5)
            if response.status_code == 200:
                self.get_logger().info(f'Status sent: {node_name} = {status}')
            else:
                self.get_logger().warn(f'Status send failed: {response.status_code} for {node_name}')
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'[HTTP Error] {node_name}: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'[Unexpected Error] {node_name}: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = Regenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('regenerator node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
