import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import os
from pathlib import Path
from custom_msgs import CustomDetection2D, CustomTrackedObjects
from sensor_msgs.msg import Image, JointState
import message_filters


class Regenerator(Node):
    def __init__(self):
        super().__init__('regenerator')
    
        image_sub = message_filters.Subscriber(self, Image, 'camera/frames')
        bbox_sub = message_filters.Subscriber(self, CustomDetection2D, 'detector/bboxes')
        keypoints_sub = message_filters.Subscriber(self, JointState, 'detector/keypoints')
        falldet_sub = message_filters.Subscriber(self, JointState, 'falldetector/falldets')
        tracked_sub = message_filters.Subscriber(self, CustomTrackedObjects, 'tracker/tracked_objects') 
        # data synchronization
        sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, bbox_sub, keypoints_sub, tracked_sub],
            queue_size=10, slop=0.5
        )
        sync.registerCallback(self.synced_callback)
        self.cv_bridge = CvBridge()
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