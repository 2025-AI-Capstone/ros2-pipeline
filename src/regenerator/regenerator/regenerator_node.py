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
        # Convert ROS image to OpenCV
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Parse bbox
        serialized_bbox = []
        for i in range(0, len(bbox_msg.detections.data), 5):
            x1, y1, x2, y2, conf = bbox_msg.detections.data[i:i+5]
            serialized_bbox.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf})
            cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Keypoints (17 x 3) for each person
        keypoints_data = np.array(falldet_msg.position).reshape(-1, 17, 3)
        display_image = draw_keypoints(cv_image, keypoints_data)

        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', display_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Tracked objects
        tracked_objs = [
            {'id': obj.id, 'bbox': {
                'x': obj.bbox.x,
                'y': obj.bbox.y,
                'width': obj.bbox.width,
                'height': obj.bbox.height
            }} for obj in tracked_msg.tracked_objects
        ]

        # Dashboard json
        dashboard_data = {
            'image': image_base64,
            'bboxes': serialized_bbox,
            'keypoints': keypoints_data.tolist(),
            'falldetections': falldet_msg.position,
            'tracked_objects': tracked_objs
        }

        dashboard_msg = String()
        dashboard_msg.data = json.dumps(dashboard_data)
        self.dashboard_pub.publish(dashboard_msg)


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