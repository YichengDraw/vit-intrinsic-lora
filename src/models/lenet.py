"""
LeNet Architecture for intrinsic dimension experiments.

Standard LeNet-5 style architecture:
- Conv(6, 5x5) -> MaxPool(2x2) -> Conv(16, 5x5) -> MaxPool(2x2)
- FC(120) -> FC(84) -> FC(10)

Paper reports:
- MNIST LeNet: 44,426 parameters, d_int90 ≈ 275
- CIFAR-10 LeNet: 62,006 parameters, d_int90 ≈ 2,900
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LeNet(nn.Module):
    """
    LeNet-5 style convolutional network.
    
    Args:
        input_channels: Number of input channels (1 for MNIST, 3 for CIFAR)
        num_classes: Number of output classes
        input_size: Spatial size of input (28 for MNIST, 32 for CIFAR)
        dropout: Dropout rate for FC layers
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout_rate = dropout
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate FC input size based on input spatial dimension
        # Formula: ((input_size - 4) / 2 - 4) / 2
        # MNIST (28): (28-4)/2 = 12, (12-4)/2 = 4 -> 16 * 4 * 4 = 256
        # CIFAR (32): (32-4)/2 = 14, (14-4)/2 = 5 -> 16 * 5 * 5 = 400
        conv_output_size = ((input_size - 4) // 2 - 4) // 2
        self.fc_input_dim = 16 * conv_output_size * conv_output_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers with optional dropout
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions

def lenet_mnist(**kwargs) -> LeNet:
    """LeNet for MNIST (1 channel, 28x28)."""
    return LeNet(input_channels=1, num_classes=10, input_size=28, **kwargs)


def lenet_cifar(**kwargs) -> LeNet:
    """LeNet for CIFAR-10 (3 channels, 32x32)."""
    return LeNet(input_channels=3, num_classes=10, input_size=32, **kwargs)
