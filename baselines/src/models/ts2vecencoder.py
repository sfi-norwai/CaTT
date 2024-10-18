import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TSEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(TSEncoder, self).__init__()
        
        # 1D Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=output_size, kernel_size=1)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_size)
        
    def forward(self, x):  # x: B x T x input_dims
        x = x.float()
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        
        
        mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        mask &= nan_mask
        x[~mask] = 0
        
        # Permute to match Conv1D input: (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)
        
        # First convolutional layer
        x = self.conv1(x)  # Shape: (batch_size, 128, sequence_length)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second convolutional layer
        x = self.conv2(x)  # Shape: (batch_size, 64, sequence_length)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Third convolutional layer
        x = self.conv3(x)  # Shape: (batch_size, output_size, sequence_length)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Permute back to original order: (batch_size, sequence_length, output_size)
        x = x.permute(0, 2, 1)
        
        return x