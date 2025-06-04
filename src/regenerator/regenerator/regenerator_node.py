import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
from custom_msgs.msg import CustomDetection2D, CustomBoolean
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool
import message_filters
import base64
import json
import numpy as np
import requests
import time

class Regenerator(Node):
    def __init__(self):
        super().__init__('regenerator_node')
    
        image_sub = message_filters.Subscriber(self, Image, 'camera/image_raw')
        bbox_sub = message_filters.Subscriber(self, CustomDetection2D, 'detector/bboxes')
        keypoints_sub = message_filters.Subscriber(self, JointState, 'detector/keypoints')
        fall_sub = message_filters.Subscriber(self, CustomBoolean, 'falldetector/falldets')
        
        # data synchronization
        sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, bbox_sub, keypoints_sub, fall_sub],
            queue_size=10, slop=0.2
        )
        sync.registerCallback(self.synced_callback)
        self.cv_bridge = CvBridge()

        # JSON publisher for dashboard
        self.dashboard_pub = self.create_publisher(String, 'dashboard/data', 10)
        
        # Service clients for health checks
        self.camera_client = self.create_client(SetBool, 'camera/check_camera')
        self.detector_client = self.create_client(SetBool, 'detector/check_detector')
        self.falldetector_client = self.create_client(SetBool, 'falldetector/check_fall')
        
        # Timer for periodic service calls (2 seconds)
        self.health_check_timer = self.create_timer(2.0, self.health_check_callback)
        
        # Track last successful service response time for each node
        self.last_service_response = {
            'camera': None,
            'detector': None,
            'falldetector': None
        }
        
        self.api_endpoint = "http://localhost:8000/system-statuses" 
        self.event_id = 1  
    
    def synced_callback(self, image_msg, bbox_msg, keypoint_msg, fall_msg):
        # Convert ROS image to OpenCV
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Parse bbox
        serialized_bbox = []
        for i in range(0, len(bbox_msg.detections.data), 5):
            x1, y1, x2, y2, conf = bbox_msg.detections.data[i:i+5]
            serialized_bbox.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf})

        # Keypoints (17 x 3) for each person
        keypoints_data = np.array(keypoint_msg.position).reshape(-1, 17, 3)

        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', cv_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Dashboard json
        dashboard_data = {
            'image': image_base64,
            'bboxes': serialized_bbox,
            'keypoints': keypoints_data.tolist(),
            'fall_detection': fall_msg.is_fall.data,
        }

        dashboard_msg = String()
        dashboard_msg.data = json.dumps(dashboard_data)
        self.dashboard_pub.publish(dashboard_msg)

    def health_check_callback(self):
        """2초마다 실행되는 헬스체크 콜백"""
        # Camera service check
        if self.camera_client.service_is_ready():
            self.call_service_async(self.camera_client, 'camera')
        else:
            self.get_logger().warn('Camera service not ready')
        
        # Detector service check
        if self.detector_client.service_is_ready():
            self.call_service_async(self.detector_client, 'detector')
        else:
            self.get_logger().warn('Detector service not ready')
        
        # Fall detector service check
        if self.falldetector_client.service_is_ready():
            self.call_service_async(self.falldetector_client, 'falldetector')
        else:
            self.get_logger().warn('Fall detector service not ready')
        
        # 서비스 호출 후 5초 뒤에 상태 보고 스케줄링
        self.create_timer(5.0, self.send_all_node_status)

    def call_service_async(self, client, service_name):
        """서비스를 비동기적으로 호출"""
        request = SetBool.Request()
        request.data = True  # health check request
        
        future = client.call_async(request)
        future.add_done_callback(lambda f: self.service_response_callback(f, service_name))

    def service_response_callback(self, future, service_name):
        """서비스 응답을 처리하는 콜백"""
        try:
            response = future.result()
            if response.success:
                self.last_service_response[service_name] = time.time()
                self.get_logger().info(f'{service_name} service: {response.message}')
            else:
                self.get_logger().warn(f'{service_name} service failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call to {service_name} failed: {str(e)}')

    def send_all_node_status(self):
        """모든 노드의 상태를 확인하고 API로 전송 (일회성 타이머)"""
        current_time = time.time()
        
        for node_name in ['camera', 'detector', 'falldetector']:
            # 마지막 서비스 응답이 7초 이내에 있었는지 확인 (2초 간격 + 5초 여유시간)
            if (self.last_service_response[node_name] is not None and 
                current_time - self.last_service_response[node_name] < 7):
                status = "active"
            else:
                status = "inactive"
            
            self.send_node_status(node_name, status)

    def send_node_status_callback(self):
        """10초마다 실행되는 노드 상태 전송 콜백"""
        current_time = time.time()
        
        for node_name in ['camera', 'detector', 'falldetector']:
            # 마지막 서비스 응답이 15초 이내에 있었는지 확인 (5초 간격 + 여유시간)
            if (self.last_service_response[node_name] is not None and 
                current_time - self.last_service_response[node_name] < 15):
                status = "Active"
            else:
                status = "Inactive"
            
            self.send_node_status(node_name, status)

    def send_node_status(self, node_name, status):
        """API에 노드 상태를 전송"""
        try:
            payload = {
                "event_id": self.event_id,
                "node_name": node_name,
                "status": status
            }
            
            response = requests.post(self.api_endpoint, json=payload, timeout=5)
            if response.status_code == 200:
                self.get_logger().info(f'Node status sent: {node_name} - {status}')
            else:
                self.get_logger().warn(f'Failed to send node status: {response.status_code}')
                
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'Error sending node status: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error in send_node_status: {str(e)}')

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