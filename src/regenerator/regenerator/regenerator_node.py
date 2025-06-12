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
from collections import deque
import os
from datetime import datetime

class Regenerator(Node):
    def __init__(self):
        super().__init__('regenerator_node')

        self.cv_bridge = CvBridge()

        image_sub = message_filters.Subscriber(self, Image, 'video_publisher/frames')
        bbox_sub = message_filters.Subscriber(self, CustomDetection2D, 'detector/bboxes')
        keypoints_sub = message_filters.Subscriber(self, JointState, 'detector/keypoints')
        fall_sub = message_filters.Subscriber(self, CustomBoolean, 'falldetector/falldets')

        sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, bbox_sub, keypoints_sub, fall_sub],
            queue_size=10, slop=0.2
        )
        sync.registerCallback(self.synced_callback)

        self.dashboard_pub = self.create_publisher(String, 'dashboard/data', 10)

        self.last_msg_time = {'camera': 0, 'detector': 0, 'falldetector': 0}
        self.create_subscription(Image, 'camera/image_raw', self.update_camera_time, 10)
        self.create_subscription(CustomDetection2D, 'detector/bboxes', self.update_detector_time, 10)
        self.create_subscription(CustomBoolean, 'falldetector/falldets', self.update_falldetector_time, 10)

        self.api_endpoint = "http://localhost:8000/system-statuses"
        self.fall_event_api = "http://localhost:8000/fall-video"
        self.event_id = 1
        self.create_timer(5.0, self.check_msg_received_recently)

        self.window = deque(maxlen=150)  # 5초치 데이터 저장

        self.temp_image_dir = "/tmp/fall_images"
        os.makedirs(self.temp_image_dir, exist_ok=True)

    def synced_callback(self, image_msg, bbox_msg, keypoint_msg, fall_msg):
        try:
            # === 이미지 처리 ===
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # === 바운딩 박스 처리 ===
            serialized_bbox = []
            for i in range(0, len(bbox_msg.detections.data), 5):
                x1, y1, x2, y2, conf = bbox_msg.detections.data[i:i + 5]
                serialized_bbox.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf})

            # === 키포인트 처리 ===
            keypoints_data = np.array(keypoint_msg.position).reshape(-1, 17, 3)

            now = time.time()
            now_str = datetime.fromtimestamp(now).isoformat()

            # Dashboard 전송용
            _, buffer = cv2.imencode('.jpg', cv_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            dashboard_data = {
                'timestamp': now_str,
                'image': image_base64,
                'bboxes': serialized_bbox,
                'keypoints': keypoints_data.tolist(),
                'fall_detection': fall_msg.is_fall.data
            }

            self.window.append(dashboard_data)
            self.publish_dashboard(dashboard_data)
            '''
            # === 낙상 발생 시 서버에 POST ===
            if fall_msg.is_fall.data:
                self.get_logger().warn("Fall Detected! Sending event to server...")

                # 이미지 파일로 임시 저장
                dt = datetime.fromtimestamp(now)
                filename = f"fall_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
                temp_path = os.path.join(self.temp_image_dir, filename)
                cv2.imwrite(temp_path, cv_image)

                # 서버에 POST
                files = {
                    'image': open(temp_path, 'rb')
                }
                data = {
                    'timestamp': now_str,
                    'is_fall': 'true',
                    'keypoints': json.dumps(keypoints_data.tolist()),
                    'bboxes': json.dumps(serialized_bbox)
                }

                try:
                    response = requests.post(self.fall_event_api, data=data, files=files, timeout=5)
                    if response.status_code == 200:
                        self.get_logger().info("Fall event successfully sent to server.")
                    else:
                        self.get_logger().error(f"Server responded with {response.status_code}: {response.text}")
                except Exception as e:
                    self.get_logger().error(f"[FallEvent POST Error] {str(e)}")
            '''
        except Exception as e:
            self.get_logger().error(f"[sync_callback] Error: {str(e)}")


    def publish_dashboard(self, data):
        msg = String()
        msg.data = json.dumps(data)
        self.dashboard_pub.publish(msg)

    def update_camera_time(self, msg):
        self.last_msg_time['camera'] = time.time()

    def update_detector_time(self, msg):
        self.last_msg_time['detector'] = time.time()

    def update_falldetector_time(self, msg):
        self.last_msg_time['falldetector'] = time.time()

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
