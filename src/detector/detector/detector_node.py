import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from custom_msgs.msg import CustomDetection2D
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
import time
from detector.predictor import SelectivePosePredictor
from ultralytics import YOLO
import logging
from std_srvs.srv import SetBool

class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')
        self.declare_parameter('selection_criteria', 'conf_threshold')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('top_num', 5)

        self.selection_criteria = self.get_parameter('selection_criteria').value
        self.threshold = self.get_parameter('threshold').value
        self.top_num = self.get_parameter('top_num').value

        self.bridge = CvBridge()
        
        # Suppress YOLO model logging
        logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
        
        self.model = YOLO('yolov8n-pose.pt')
        self.model.predictor = SelectivePosePredictor(selection_criteria=self.selection_criteria,
                                                      threshold=self.threshold,
                                                      top_num=self.top_num)

        self.get_logger().info("Detection model loaded")
        
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.cv_bridge = CvBridge()

        # Publisher 설정
        self.publisher = self.create_publisher(CustomDetection2D, 'detector/bboxes', 10)
        self.keypoints_publisher = self.create_publisher(JointState, 'detector/keypoints', 10)
        self.latency_publisher = self.create_publisher(Float32, 'detector/total_latency', 10)
        self.inference_time_publisher = self.create_publisher(Float32, 'detector/inference_time', 10)
        
        # Create CheckDetector service
        self.srv = self.create_service(SetBool, 'detector/check_detector', self.check_detector_callback)
        
        self.inference_time_list = []
        self.total_latency_list = []

        self.create_timer(10, self.printlog)

    def image_callback(self, msg):
        try:
            # 메시지 수신 시간 기록
            receive_time = self.get_clock().now()
            msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
            transmission_latency = (receive_time - msg_time).nanoseconds / 1e9

            # 전체 처리 시작 시간
            process_start = time.time()
            
            # 이미지 변환
            image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 추론 시작
            inference_start = time.time()
            results = self.model(image)
            inference_time = time.time() - inference_start
            
            # Bounding Box 메시지 초기화
            detection_msg = CustomDetection2D()
            detection_msg.header.stamp = msg.header.stamp
            detection_msg.header.frame_id = "camera_frame"

            # Keypoints 메시지 초기화
            keypoints_msg = JointState()
            keypoints_msg.header.stamp = msg.header.stamp
            keypoints_msg.header.frame_id = "camera_frame"

            bounding_boxes = []
            # YOLOv8 결과 처리
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                detections = Float32MultiArray()
                for box, kpts in zip(boxes, keypoints):
                    if box.cls == 0:  # 사람이 감지된 경우
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])  # Confidence Score

                        # Bounding Box 메시지 구성
                        
                        detections.data.extend([x1, y1, x2, y2, conf])

                        # Keypoints 데이터 구성
                        kpts_coords = kpts.data[0].tolist()  # Shape: [17, 3] (COCO 포맷)
                        for point in kpts_coords:
                            keypoints_msg.position.extend([float(point[0]), float(point[1]), float(point[2])])
            
            detection_msg.detections = detections
            # Bounding Box & Keypoints 메시지 전송
            if detection_msg.detections:
                self.publisher.publish(detection_msg)
                self.keypoints_publisher.publish(keypoints_msg)

            # 전체 처리 시간
            processing_time = time.time() - process_start
            total_latency = transmission_latency + processing_time
            
            # 전체 latency publish
            latency_msg = Float32()
            latency_msg.data = float(total_latency)
            self.latency_publisher.publish(latency_msg)
            
            self.inference_time_list.append(inference_time)
            self.total_latency_list.append(total_latency)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")
        
    def printlog(self):
        if self.inference_time_list and self.total_latency_list:
            self.get_logger().info(f"""
            Latency breakdown:
            - Inference time: {sum(self.inference_time_list)/len(self.inference_time_list):.3f}s
            - Total latency: {sum(self.total_latency_list)/len(self.total_latency_list):.3f}s
            """)
            self.inference_time_list.pop(0)
            self.total_latency_list.pop(0)

    def check_detector_callback(self, request, response):
        if request.data:
            self.get_logger().info("Check detector service called with True")
            response.success = True
            response.message = "detector is running."
        return response
    
def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
