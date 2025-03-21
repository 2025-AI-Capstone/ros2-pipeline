import torch.nn as nn

# 신경망 모델 클래스 정의
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 첫 번째 은닉층
        self.fc2 = nn.Linear(128, 64)          # 두 번째 은닉층
        self.fc3 = nn.Linear(64, 1)            # 출력층
        self.relu = nn.ReLU()                  # ReLU 활성화 함수
        self.sigmoid = nn.Sigmoid()            # Sigmoid 출력층

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 첫 번째 은닉층 + ReLU
        x = self.relu(self.fc2(x))  # 두 번째 은닉층 + ReLU
        x = self.sigmoid(self.fc3(x))  # 출력층 + Sigmoid
        return x