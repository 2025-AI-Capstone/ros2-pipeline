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