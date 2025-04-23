import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import os
from pathlib import Path
from custom_msgs import CustomDetection2D, CustomTrackedObjects
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
import message_filters
import base64
import json
import numpy as np

class Regenerator(Node):
    def __init__(self):
        super().__init__('regenerator_node')
    
        image_sub = message_filters.Subscriber(self, Image, 'camera/frames')
        bbox_sub = message_filters.Subscriber(self, CustomDetection2D, 'detector/bboxes')
        # keypoints_sub = message_filters.Subscriber(self, JointState, 'detector/keypoints')
        falldet_sub = message_filters.Subscriber(self, JointState, 'falldetector/falldets')
        tracked_sub = message_filters.Subscriber(self, CustomTrackedObjects, 'tracker/tracked_objects') 
        # data synchronization
        sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, bbox_sub, falldet_sub, tracked_sub],
            queue_size=10, slop=0.5
        )
        sync.registerCallback(self.synced_callback)
        self.cv_bridge = CvBridge()

        # JSON publisher for dashboard
        self.dashboard_pub = self.create_publisher(String, 'dashboard/data', 10)
    
    def synced_callback(self, image_msg, bbox_msg, falldet_msg, tracked_msg):
        # Convert the ROS Image message to OpenCV format
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        _, buffer = cv2.imencode('.jpg', cv_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        bbox_data = bbox_msg.detections.data
        serialized_keypoints = falldet_msg.position
        serialized_falldet = falldet_msg.name
        serialized_bbox = []
        
        for i in range(0, len(bbox_data), 5):
            bbox = {
                'x1': bbox_data[i],
                'y1': bbox_data[i + 1],
                'x2': bbox_data[i + 2],
                'y2': bbox_data[i + 3],
                'conf': bbox_data[i + 4]
                }
            serialized_bbox.append(bbox)

        dashboard_data = {
            'image': image_base64,
            'bboxes': serialized_bbox,
            'keypoints': serialized_keypoints,
            'falldetections': falldet_msg.position,
            'tracked_objects': [
                {'id': obj.id, 'bbox': {'x': obj.bbox.x, 'y': obj.bbox.y,
                                        'width': obj.bbox.width, 'height': obj.bbox.height}}
                for obj in tracked_msg.tracked_objects
            ]
        }

        dashboard_json = String()
        dashboard_json.data = json.dumps(dashboard_data)
        self.dashboard_pub.publish(dashboard_json)

    def synced_callback(self, image_msg, bbox_msg, falldet_msg, tracked_msg):
        # Convert the ROS Image message to OpenCV format
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        bbox_data = bbox_msg.detections.data
        keypoints_data = np.array(keypoints_msg.position).reshape(-1, 17, 3) if keypoints_msg.position else None
        detections = []
        for detection in bbox_msg.detections:
            x1, y1, x2, y2, conf = detection
            detections.append([x1, y1, x2, y2])
        if detections is not None:
                for obj in detections:  # 각 obj는 [x1, y1, x2, y2, track_id] 형태
                    try:
                        x1, y1, x2, y2, track_id = obj  # 리스트 언패킹
                        
                        # 좌표를 정수로 변환
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    except (ValueError, TypeError) as e:
                        print(f"Error in processing Detected objects")
                        continue
                    
        if keypoints_data is not None:
            display_image = draw_keypoints(cv_image, keypoints_data)

        
        _, buffer = cv2.imencode('.jpg', cv_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        dashboard_data = {
            'image': image_base64,
            'bboxes': serialized_bbox,
            'keypoints': serialized_keypoints,
            'falldetections': falldet_msg.position,
            'tracked_objects': [
                {'id': obj.id, 'bbox': {'x': obj.bbox.x, 'y': obj.bbox.y,
                                        'width': obj.bbox.width, 'height': obj.bbox.height}}
                for obj in tracked_msg.tracked_objects
            ]
        }

        dashboard_json = String()
        dashboard_json.data = json.dumps(dashboard_data)
        self.dashboard_pub.publish(dashboard_json)

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