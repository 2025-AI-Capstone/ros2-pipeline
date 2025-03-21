import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from pathlib import Path

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        
        self.declare_parameter('image_directory', './src/image_publisher/images')
        self.declare_parameter('publish_rate', 30.0)
        
        self.publisher = self.create_publisher(Image, 'image', 10)
        
        # 이미지 디렉토리 경로 가져오기
        self.image_dir = self.get_parameter('image_directory').value
        rate = self.get_parameter('publish_rate').value
        
        # 이미지 파일 리스트 생성
        self.image_files = []
        self.current_index = 0
        self.cv_bridge = CvBridge()
        
        # 지원하는 이미지 확장자
        self.image_extensions = ['.jpg', '.jpeg', 'png']
        
        # 이미지 파일 찾기
        self.load_image_files()
        
        if not self.image_files:
            self.get_logger().error('No image files found in directory: %s' % self.image_dir)
            return
            
        # 타이머 생성
        self.timer = self.create_timer(1.0/rate, self.timer_callback)
        self.get_logger().info('Started publishing images from: %s' % self.image_dir)
        
    def load_image_files(self):
        if not os.path.exists(self.image_dir):
            self.get_logger().error('Directory does not exist: %s' % self.image_dir)
            return
            
        for file in Path(self.image_dir).iterdir():
            if file.suffix.lower() in self.image_extensions:
                self.image_files.append(str(file))
        self.image_files.sort()
        
    def timer_callback(self):
        # 타이머 콜백 함수: 이미지를 읽어서 발행
        if not self.image_files:
            return
            
        # 현재 이미지 파일 읽기
        image_path = self.image_files[self.current_index]
        
        try:
            # OpenCV로 이미지 읽기
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                self.get_logger().error('Failed to read image: %s' % image_path)
                return
                
            # ROS 메시지로 변환
            ros_image = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'camera_frame'
            
            # 이미지 발행
            self.publisher.publish(ros_image)
            self.get_logger().info('Published image: %s' % image_path)
            
            # 다음 이미지로 인덱스 이동
            self.current_index = (self.current_index + 1) % len(self.image_files)
            
        except Exception as e:
            self.get_logger().error('Error publishing image: %s' % str(e))

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('image_publisher node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()