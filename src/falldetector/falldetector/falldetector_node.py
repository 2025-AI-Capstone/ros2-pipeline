import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sklearn.preprocessing import MinMaxScaler
from falldetector.temp import SimpleNN
import joblib
import numpy as np
import time
from std_srvs.srv import SetBool
import torch

class FallDetectorNode(Node):
    def __init__(self):
        super().__init__('falldetector_node')

        # Keypoints 데이터 구독
        self.subscription = self.create_subscription(
            JointState,
            'detector/keypoints',
            self.keypoints_callback,
            10
        )

        # Fall Detection 결과 발행
        self.publisher = self.create_publisher(JointState, 'falldetector/falldets', 10)

        # CheckFall 서비스 생성
        self.srv = self.create_service(SetBool, 'falldetector/check_fall', self.check_fall_callback)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_size = 34
        # Fall Detection 모델 로드
        self.model = SimpleNN(self.input_size)
        self.model.load_state_dict(torch.load('./src/falldetector/falldetector/checkpoints/model.pt', map_location=self.device))
        self.model = self.model.to(self.device)  # 모델을 GPU로 이동
        self.model.eval()  # 평가 모드로 설정
        self.create_timer(10, self.printlog)
        self.msg_count = 0
        self.start_time = time.time()

    def keypoints_callback(self, msg: JointState):
        try:
            # Keypoints 데이터 변환
            keypoints = np.array(msg.position, dtype=np.float32).reshape(-1, 17, 3)  # (N, 17, 3)
            keypoints = torch.from_numpy(keypoints).to(self.device)
            num_people = keypoints.shape[0]  
            keypoints_xy = keypoints[..., :2]
            non_zero_count = torch.count_nonzero(keypoints_xy).item()
            if non_zero_count > 13:
                keypoints_resized = keypoints_xy.reshape(num_people, -1)
                keypoints_resized_clone = keypoints_resized.clone().detach()
                prediction = self.model(keypoints_resized_clone)


            result_msg = JointState()
            result_msg.header.stamp = msg.header.stamp  
            result_msg.header.frame_id = msg.header.frame_id 

            result_msg.name = ["FALL" if pred == 1 else "NORMAL" for pred in prediction]

            result_msg.position = msg.position

            # 결과 발행
            self.publisher.publish(result_msg)
            self.msg_count += 1

        except Exception as e:
            self.get_logger().error(f"Failed to process fall detection: {e}")


    def min_max_scaling(self, filtered_array):
        """Min-Max Scaling을 적용하여 데이터를 정규화"""
        scaler = MinMaxScaler()
        coord_values_scaled = scaler.fit_transform(filtered_array[:, :34])
        filtered_array[:, :34] = coord_values_scaled
        return filtered_array
    
    def check_fall_callback(self, request, response):
        """Fall Detection 노드 상태 확인"""
        if request.data:
            self.get_logger().info("Check falldetector service called with True")
            response.success = True
            response.message = "falldetector is running."
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