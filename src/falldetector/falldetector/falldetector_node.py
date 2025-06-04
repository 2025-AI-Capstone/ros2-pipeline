import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from custom_msgs.msg import CustomBoolean
from std_srvs.srv import SetBool
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import time
from torch_geometric.data import Data
from falldetector.model import FallDetectionSGAT

class FallDetectorNode(Node):
    def __init__(self):
        super().__init__('falldetector_node')

        self.subscription = self.create_subscription(JointState, 'detector/keypoints', self.keypoints_callback, 10)
        self.publisher = self.create_publisher(CustomBoolean, 'falldetector/falldets', 10)
        self.srv = self.create_service(SetBool, 'falldetector/check_fall', self.check_fall_callback)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = FallDetectionSGAT(in_channels=3).to(self.device)
        checkpoint = torch.load('./src/falldetector/falldetector/checkpoints/stonegat.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.create_timer(10, self.printlog)
        self.msg_count = 0
        self.start_time = time.time()

    def keypoints_callback(self, msg: JointState):
        try:
            people = np.array(msg.position, dtype=np.float32).reshape(-1, 17, 3)
            people = minmax_scale_keypoints(people)  # (N, 18, 3)

            result_msg = CustomBoolean()
            result_msg.header.stamp = msg.header.stamp
            result_msg.header.frame_id = msg.header.frame_id
            result_msg.is_fall = Bool(data=False)

            for person in people:
                if np.count_nonzero(person[:, :2]) < 10:
                    continue
                graph = keypoints_to_graph(person).to(self.device)
                out = self.model(graph.x, graph.edge_index, graph.edge_attr)
                confidence = out.squeeze().item()
                if confidence > 0.5:
                    result_msg.is_fall = Bool(data=True)
                    self.msg_count += 1
                    break  # 첫 낙상 감지자만 처리

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

def keypoints_to_graph(keypoints_np):  # keypoints_np: (18, 3)
    base_edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 7), (6, 8),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12), (5, 11), (6, 12), (17, 5), (17, 6),
        (8, 10), (7, 9), (0, 17)
    ]
    
    # 양방향으로 만듦
    edges = base_edges + [(b, a) for (a, b) in base_edges]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(keypoints_np, dtype=torch.float32)  # (18, 3)

    edge_attr = []
    for src, dst in edges:
        p1 = keypoints_np[src]
        p2 = keypoints_np[dst]

        # 거리
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = (dx**2 + dy**2)**0.5

        # 각도 (라디안)
        angle = np.arctan2(dy, dx)

        # confidence 평균
        conf_avg = (p1[2] + p2[2]) / 2.0

        edge_attr.append([distance, angle, conf_avg])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


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
