#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import os
import time
from std_srvs.srv import SetBool
import base64

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        
        # Declare parameters
        self.declare_parameter('video_path', './src/video_publisher/video/MOT17-08-SDP-raw.webm')
        self.declare_parameter('loop', True)
        self.declare_parameter('fps', 30)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        
        # Get parameters
        self.video_path = self.get_parameter('video_path').value
        self.loop = self.get_parameter('loop').value
        self.fps = self.get_parameter('fps').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        
        # Create publisher
        self.publisher = self.create_publisher(Image, 'video_publisher/frames', 10)
        self.web_publisher = self.create_publisher(String, 'video_publisher/stream', 10)
        # Create CheckVideo service
        self.srv = self.create_service(SetBool, 'video_publisher/check_video', self.check_video_callback)
        
        # Set timer 
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Initialize video capture
        self.init_video_capture()
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        self.get_logger().info(f'Video Publisher started with file: {self.video_path}')
        self.get_logger().info(f'FPS: {self.fps}, Loop: {self.loop}')
        self.get_logger().info(f'Resolution: {self.width}x{self.height}')

        self.create_timer(10, self.printlog)
        
        # Initialize frame count and start time
        self.frame_count = 0
        self.start_time = time.time()

    def init_video_capture(self):
        """비디오 캡처 객체를 초기화하는 메소드"""
        if not os.path.exists(self.video_path):
            self.get_logger().error(f'Video file not found: {self.video_path}')
            return False
            
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open video file')
            return False
            
        # 비디오 정보 출력
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'Video FPS: {actual_fps}')
        self.get_logger().info(f'Total frames: {frame_count}')
        self.get_logger().info(f'Original resolution: {original_width}x{original_height}')
        self.get_logger().info(f'Resizing to: {self.width}x{self.height}')
        
        return True

    def timer_callback(self):
        if not self.cap.isOpened():
            if not self.init_video_capture():
                return
                
        ret, frame = self.cap.read()
        
        if ret:
            # Resize frame to desired resolution
            resized_frame = cv2.resize(frame, (self.width, self.height))
            
            # OpenCV 이미지를 ROS 메시지로 변환
            msg = self.bridge.cv2_to_imgmsg(resized_frame, encoding='bgr8')
            
            # Encode for web streaming
            _, buffer = cv2.imencode('.jpeg', resized_frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_msg = String()
            base64_msg.data = image_base64
            
            # 현재 프레임 번호 추가
            msg.header.stamp = self.get_clock().now().to_msg()
            
            # 프레임 발행
            self.publisher.publish(msg)
            self.web_publisher.publish(base64_msg)
            self.frame_count += 1
            
        else:
            if self.loop:
                self.get_logger().info('Restarting video from beginning')
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오 처음부터 다시 재생
            else:
                self.get_logger().info('End of video reached')
                self.cap.release()
                rclpy.shutdown()

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

    def printlog(self):
        elapsed_time = time.time() - self.start_time
        average_frame_rate = self.frame_count / elapsed_time
        self.get_logger().info(f'''
        Average frame rate: {average_frame_rate:.2f} FPS
        frame count: {self.frame_count}
        ''')
        
        # Reset frame count and start time for the next interval
        self.frame_count = 0
        self.start_time = time.time()
    
    def check_video_callback(self, request, response):
        if request.data:
            self.get_logger().info("Check video service called with True")
            response.success = True
            response.message = "Video is running."

        return response

def main(args=None):
    rclpy.init(args=args)
    
    video_publisher = VideoPublisher()
    
    try:
        rclpy.spin(video_publisher)
    except KeyboardInterrupt:
        video_publisher.get_logger().info('Stopped by keyboard interrupt')
    finally:
        video_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()