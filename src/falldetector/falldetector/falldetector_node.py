import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool
import math
from sklearn.preprocessing import MinMaxScaler
from falldetector.temp import SimpleNN
import json
import torch
import numpy as np
import time
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

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
        self.publisher = self.create_publisher(String, 'falldetector/falldets', 10)

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
            keypoints = minmax_scale_keypoints(keypoints)  # Min-Max Scaling 적용
            keypoints = torch.from_numpy(keypoints).to(self.device)

            # num_people = keypoints.shape[0]
            non_zero_count = torch.count_nonzero(keypoints).item()

            if non_zero_count > 9 and self.bbox_trigger:
                graph = keypoints_to_graph(keypoints).to(self.device)             
                prediction = self.model(graph.x, graph.edge_index, graph.edge_attr)
                confidence_score = prediction.squeeze().tolist()
                if confidence_score > 0.5 :
                    result_msg = String()
                    result_msg.header.stamp = msg.header.stamp
                    result_msg.header.frame_id = msg.header.frame_id
                    result = {
                        "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                        "frame_id": msg.header.frame_id,
                        "confidence_score": confidence_score,
                    }
                    result_msg.data = json.dumps(result)
                    self.publisher.publish(result_msg)
                    self.msg_count += 1

                self.bbox_trigger = False  # 트리거 초기화
            else:
                self.publisher.publish(String(data=0))
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

def keypoints_to_graph(keypoints_np):
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

    edge_attr = []
    for src, dst in edges:
        p1 = keypoints_np[src][:2]
        p2 = keypoints_np[dst][:2]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = (dx**2 + dy**2)**0.5
        angle = math.atan2(dy, dx)
        edge_attr.append([dist, angle])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

def minmax_scale_keypoints(keypoints_np):
    """
    Min-Max Scaling을 x, y 좌표에만 적용하고,
    confidence는 그대로 유지

    Args:
        keypoints_np: (N, 17, 3) numpy array
    Returns:
        scaled_np: (N, 17, 3) numpy array
    """
    N = keypoints_np.shape[0]
    scaled = keypoints_np.copy()

    for i in range(N):
        xy = scaled[i, :, :2]     # (17, 2)
        conf = scaled[i, :, 2:]   # (17, 1)

        scaler = MinMaxScaler()
        xy_scaled = scaler.fit_transform(xy)

        scaled[i, :, :2] = xy_scaled
        scaled[i, :, 2:] = conf   # confidence 그대로 유지

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
