import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool

from sklearn.preprocessing import MinMaxScaler
from falldetector.temp import SimpleNN

import torch
import numpy as np
import time
from collections import deque

class FallDetectorNode(Node):
    def __init__(self):
        super().__init__('falldetector_node')

        # Keypoints 구독
        self.subscription = self.create_subscription(
            JointState,
            'detector/keypoints',
            self.keypoints_callback,
            10
        )

        # Bounding Box 구독
        self.bbox_subscription = self.create_subscription(
            Float32MultiArray,
            'detector/bboxes',
            self.bbox_callback,
            10
        )

        # 결과 발행
        self.publisher = self.create_publisher(JointState, 'falldetector/falldets', 10)

        # 서비스 생성
        self.srv = self.create_service(SetBool, 'falldetector/check_fall', self.check_fall_callback)

        # 모델 설정
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_size = 34
        self.model = SimpleNN(self.input_size)
        self.model.load_state_dict(torch.load('./src/falldetector/falldetector/checkpoints/model.pt', map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # 바운딩박스 시계열 큐
        self.bbox_ratio_queue = deque(maxlen=5)
        self.bbox_trigger = False

        # 처리 속도 로그용
        self.create_timer(10, self.printlog)
        self.msg_count = 0
        self.start_time = time.time()

    def bbox_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 5:
            return

        # 단일 객체 기준 (가장 앞 1명)
        x1, y1, x2, y2, conf = data[:5]
        width = x2 - x1
        height = y2 - y1

        if height <= 0:
            return

        ratio = width / height
        self.bbox_ratio_queue.append(ratio)

        if len(self.bbox_ratio_queue) == self.bbox_ratio_queue.maxlen:
            delta = max(self.bbox_ratio_queue) - min(self.bbox_ratio_queue)
            if delta > 0.8:
                self.bbox_trigger = True

    def keypoints_callback(self, msg: JointState):
        try:
            keypoints = np.array(msg.position, dtype=np.float32).reshape(-1, 17, 3)
            keypoints = torch.from_numpy(keypoints).to(self.device)

            num_people = keypoints.shape[0]
            keypoints_xy = keypoints[..., :2]
            non_zero_count = torch.count_nonzero(keypoints_xy).item()

            if non_zero_count > 9 and self.bbox_trigger:
                keypoints_resized = keypoints_xy.reshape(num_people, -1)
                keypoints_resized_clone = keypoints_resized.clone().detach()
                prediction = self.model(keypoints_resized_clone)

                result_msg = JointState()
                result_msg.header.stamp = msg.header.stamp
                result_msg.header.frame_id = msg.header.frame_id

                result_msg.name = ["FALL" if pred == 1 else "NORMAL" for pred in prediction]
                result_msg.position = msg.position

                self.publisher.publish(result_msg)
                self.msg_count += 1

                self.bbox_trigger = False  # 트리거 초기화
            else:
                self.get_logger().info("No valid keypoints or bbox trigger not activated.")

        except Exception as e:
            self.get_logger().error(f"Failed to process fall detection: {e}")

    def min_max_scaling(self, filtered_array):
        scaler = MinMaxScaler()
        coord_values_scaled = scaler.fit_transform(filtered_array[:, :34])
        filtered_array[:, :34] = coord_values_scaled
        return filtered_array

    def check_fall_callback(self, request, response):
        if request.data:
            self.get_logger().info("Check falldetector service called with True")
            response.success = True
            response.message = "falldetector is running."
        return response

    def printlog(self):
        elapsed_time = time.time() - self.start_time
        average_frame_rate = self.msg_count / elapsed_time
        self.get_logger().info(f'''
        Average msg rate: {average_frame_rate:.2f} MPS
        msg count: {self.msg_count}
        ''')
        self.msg_count = 0
        self.start_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = FallDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('falldetector node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
