import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATv2Conv, GCNConv

# edge atrribute 설정 조심
class FallDetectionSGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=1, edge_attr_dim=None):
        super(FallDetectionSGAT, self).__init__()
        
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=2, concat=False, edge_dim=edge_attr_dim)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.gat2 = GATv2Conv(hidden_channels, 64, heads=2, concat=False, edge_dim=edge_attr_dim)
        self.bn2 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, out_channels)
    
    def forward(self, x, edge_index, edge_attr, batch=None):        
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.gat2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = global_mean_pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
        
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.elu(x)
        
        x = self.fc2(x)
        return torch.sigmoid(x)

