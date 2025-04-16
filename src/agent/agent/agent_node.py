import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
from std_srvs.srv import SetBool

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        self.declare_parameter('agent_id', 0)
        self.declare_parameter('frame_rate', 30)

        self.agent_id = self.get_parameter('agent_id').value
        self.frame_rate = self.get_parameter('frame_rate').value

        self.publisher = self.create_publisher(Image, 'agent/image_raw', 10)

        # Create Checkagent service
        self.srv = self.create_service(SetBool, 'agent/check_agent', self.check_agent_callback)
        self.bridge = CvBridge()

        self.create_timer(10, self.printlog)
        self.frame_count = 0
        self.start_time = time.time()

        # Open the webcam
        self.cap = cv2.VideoCapture(self.agent_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open agent with ID {self.agent_id}")
            return

        # Timer to periodically publish frames
        self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_frame)
        self.get_logger().info(f"agent node initialized with ID {self.agent_id} at {self.frame_rate} FPS")

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to read frame from agent")
            return

        # Convert frame to ROS2 Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)
        self.frame_count += 1
        
    def destroy_node(self):
        # Release the webcam resource
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

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

    def check_agent_callback(self, request, response):
        if request.data:
            self.get_logger().info("Check agent service called with True")
            response.success = True
            response.message = "agent is running."

        return response

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('agent node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
