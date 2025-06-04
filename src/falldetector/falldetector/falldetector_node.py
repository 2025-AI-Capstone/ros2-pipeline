import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from custom_msgs.msg import CustomBoolean  # custom message
from std_srvs.srv import SetBool
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import time
from collections import deque
from torch_geometric.data import Data
from falldetector.model import FallDetectionSGAT

class FallDetectorNode(Node):
    def __init__(self):
        super().__init__('falldetector_node')

        self.subscription = self.create_subscription(JointState, 'detector/keypoints', self.keypoints_callback, 10)
        self.bbox_subscription = self.create_subscription(Float32MultiArray, 'detector/bboxes', self.bbox_callback, 10)
        self.publisher = self.create_publisher(CustomBoolean, 'falldetector/falldets', 10)
        self.srv = self.create_service(SetBool, 'falldetector/check_fall', self.check_fall_callback)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = FallDetectionSGAT(in_channels=3).to(self.device)
        checkpoint = torch.load('./src/falldetector/falldetector/checkpoints/stonegat.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.bbox_ratio_queue = deque(maxlen=5)
        self.bbox_trigger = False

        self.create_timer(10, self.printlog)
        self.msg_count = 0
        self.start_time = time.time()

    def bbox_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 5:
            return
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
            keypoints = minmax_scale_keypoints(keypoints)
            keypoints = keypoints[0]  # 단일 인물만 처리
            graph = keypoints_to_graph(keypoints).to(self.device)

            non_zero_count = torch.count_nonzero(graph.x).item()

            result_msg = CustomBoolean()
            result_msg.header.stamp = msg.header.stamp
            result_msg.header.frame_id = msg.header.frame_id

            if non_zero_count > 9 and self.bbox_trigger:
                out = self.model(graph)
                confidence_score = out.squeeze().item()
                result_msg.is_fall = confidence_score > 0.5
                if result_msg.is_fall:
                    self.msg_count += 1
                self.bbox_trigger = False
            else:
                result_msg.is_fall = False

            self.publisher.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process fall detection: {e}")

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

# === 그래프 생성 ===
def keypoints_to_graph(keypoints_np):  # keypoints_np: (18, 3)
    base_edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 7), (6, 8),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12), (5, 11), (6, 12), (17, 5), (17, 6),
        (8, 10), (7, 9), (0, 17)
    ]
    node_features = torch.tensor(keypoints_np, dtype=torch.float32)
    edges = base_edges + [(b, a) for (a, b) in base_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)

# === MinMax 정규화 및 어깨 중간점 추가 ===
def minmax_scale_keypoints(keypoints_np):
    N = keypoints_np.shape[0]
    scaled = np.zeros((N, 18, 3), dtype=np.float32)
    for i in range(N):
        kp = keypoints_np[i]
        mid_shoulder = (kp[5] + kp[6]) / 2.0
        kp_augmented = np.vstack([kp, mid_shoulder])
        xy = kp_augmented[:, :2]
        conf = kp_augmented[:, 2:]
        scaler = MinMaxScaler()
        xy_scaled = scaler.fit_transform(xy)
        scaled[i, :, :2] = xy_scaled
        scaled[i, :, 2:] = conf
    return scaled

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
