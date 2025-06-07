import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from vision_msgs.msg import Detection2DArray, Detection2D
from custom_msgs.msg import CustomDetection2D, CustomTrackedObjects
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import time
import torch
import numpy as np  
from std_srvs.srv import SetBool
from tracker.sort import Sort  # SORT
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT

class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        # set tracker_type parameter
        self.declare_parameter('tracker_type', 'sort')  # default : sort
        tracker_type = self.get_parameter('tracker_type').value

        self.get_logger().info(f"Selected Tracker: {tracker_type}")

        self.bridge = CvBridge()
        self.current_frame = None

        # initialize tracker by tracker_type
        if tracker_type == 'sort':
            self.tracker = Sort()

        elif tracker_type == 'deepsort':
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=1.0,
                nn_budget=None,
                max_cosine_distance=0.3,
                override_track_class=None, embedder="mobilenet",
                half=True, bgr=True, embedder_gpu=True,
                embedder_model_name=None, embedder_wts=None, polygon=False, today=None,
            )
        # subscibe bounding box topic
        self.subscription_bboxes = self.create_subscription(
            CustomDetection2D,
            'detector/bboxes',
            self.bbox_callback,
            10
        )
        # subscribe Image topic
        self.subscription_frames = self.create_subscription(
            Image,
            'scheduler/frames',
            self.image_callback,
            10
        )
        # publish tracked objects
        self.publisher = self.create_publisher(
            CustomTrackedObjects,
            'tracker/tracked_objects',
            10
        )

        # Create tracker service
        self.srv = self.create_service(SetBool, 'tracker/check_tracker', self.check_tracker_callback)
        self.create_timer(10, self.printlog)
        self.msg_count = 0
        self.start_time = time.time()

    def image_callback(self, msg):
        """Convert ros2 image to cv2 img"""
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def bbox_callback(self, msg: Detection2DArray):
        """update tracker and publish tracked objects"""
        detections = []
        detection_data = msg.detections.data
        
        if isinstance(self.tracker, Sort):
            # convert ros2 bbox to numpy arrary
            for i in range(0, len(detection_data),5):
                x1 = detection_data[i]
                y1 = detection_data[i+1]
                x2 = detection_data[i+2]
                y2 = detection_data[i+3]
                conf = detection_data[i+4]
                detections.append([x1, y1, x2, y2, conf])
            dets = np.array(detections, dtype=np.float32)
            tracks = self.tracker.update(dets)

        elif isinstance(self.tracker, DeepSort):
            for i, detection in enumerate(msg.detections):
                x1, y1, x2, y2, conf = detection
                detections.append(([x1, y1, x2, y2], conf))
            tracks = self.tracker.update_tracks(detections, frame=self.current_frame) if self.current_frame is not None else []

        # 트래킹 결과 메시지 생성 (JointState 사용)
        tracked_objects_msg = CustomTrackedObjects()
        tracked_objects_msg.header.stamp = msg.header.stamp
        tracked_objects_msg.header.frame_id = msg.header.frame_id

        tracked_ids = []
        tracked_bboxes = []

        for track in tracks:
            track_id = track.track_id if hasattr(track, 'track_id') else track[4]
            ltrb = track.to_ltrb() if hasattr(track, 'to_ltrb') else track[:4]
            
            tracked_ids.append(str(track_id))  # 트래킹 ID를 문자열로 저장
            tracked_bboxes.extend([float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])])  # 바운딩 박스 좌표 저장

        tracked_objects_msg.id = tracked_ids
        tracked_objects_msg.bboxes = tracked_bboxes

        # 트래킹 결과 발행
        if tracked_objects_msg.name:
            self.publisher.publish(tracked_objects_msg)
        self.msg_count += 1

    def check_tracker_callback(self, request, response):
        """Tracker 상태 확인 서비스"""
        if request.data:
            self.get_logger().info("Check tracker service called with True")
            response.success = True
            response.message = "Tracker is running."
        return response

    def printlog(self):
        """메시지 처리 속도 로그 출력"""
        elapsed_time = time.time() - self.start_time
        average_frame_rate = self.msg_count / elapsed_time
        self.get_logger().info(f'''
        Average msg rate: {average_frame_rate:.2f} MPS
        msg count: {self.msg_count}
        ''')
        
        # Reset frame count and start time for the next interval
        self.msg_count = 0
        self.start_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    tracker_node = TrackerNode()
    rclpy.spin(tracker_node)
    tracker_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
