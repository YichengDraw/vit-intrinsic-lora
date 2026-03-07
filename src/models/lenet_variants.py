"""
LeNet Variants for ConvNet architecture study (Supplementary S10).

Four variants to study the contribution of local receptive fields and weight-tying:
1. Standard LeNet: Local connections + tied weights (Conv2d)
2. UntiedLeNet: Local connections, untied weights (LocallyConnected2D equivalent)
3. FCTiedLeNet: Global connections + tied weights (full-image convolutions)
4. FCLeNet: Global connections, untied weights (equivalent to FC with same units)

Paper findings:
- Both local connections AND weight-tying contribute to lower d_int90
- Standard LeNet: d_int90 ≈ 290 (MNIST), 1000 (CIFAR)
- Untied LeNet: d_int90 ≈ 600 (MNIST), 2750 (CIFAR)
- FCTied LeNet: d_int90 ≈ 425 (MNIST), 2500 (CIFAR)
- FC LeNet: d_int90 ≈ 2000 (MNIST), 35000 (CIFAR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class UntiedLeNet(nn.Module):
    """
    LeNet with untied weights (LocallyConnected2D equivalent).
    
    Each spatial position has its own set of filter weights.
    This significantly increases parameter count.
    
    For MNIST (28x28):
    - Conv1 equivalent: 6 filters, 5x5, applied at 24x24 positions = 6 * 25 * 24 * 24
    - Conv2 equivalent: 16 filters, 5x5, applied at 8x8 positions
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Layer 1: Untied convolution (locally connected)
        # Output size: input_size - 4 = 24 (for MNIST)
        self.out_size1 = input_size - 4
        # Each position (i, j) has its own 5x5 filter for each of 6 output channels
        # Total: 6 * (input_channels * 5 * 5) * out_size1 * out_size1
        self.local_weights1 = nn.Parameter(
            torch.randn(6, input_channels * 25, self.out_size1, self.out_size1) * 0.1
        )
        self.local_bias1 = nn.Parameter(torch.zeros(6, self.out_size1, self.out_size1))
        
        # Pooling: 24 -> 12
        self.pool = nn.MaxPool2d(2, 2)
        self.pooled_size1 = self.out_size1 // 2
        
        # Layer 2: Untied convolution
        self.out_size2 = self.pooled_size1 - 4
        self.local_weights2 = nn.Parameter(
            torch.randn(16, 6 * 25, self.out_size2, self.out_size2) * 0.1
        )
        self.local_bias2 = nn.Parameter(torch.zeros(16, self.out_size2, self.out_size2))
        
        # Pooling
        self.pooled_size2 = self.out_size2 // 2
        
        # FC layers (same as standard LeNet)
        fc_input = 16 * self.pooled_size2 * self.pooled_size2
        self.fc1 = nn.Linear(fc_input, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def _locally_connected_forward(
        self, 
        x: torch.Tensor, 
        weights: torch.Tensor, 
        bias: torch.Tensor,
        kernel_size: int = 5
    ) -> torch.Tensor:
        """Apply locally connected layer (untied convolution)."""
        batch_size = x.size(0)
        in_channels = x.size(1)
        out_channels = weights.size(0)
        out_h, out_w = weights.size(2), weights.size(3)
        
        # Extract patches using unfold
        # x: (batch, in_channels, H, W)
        # patches: (batch, in_channels * kernel^2, out_h * out_w)
        patches = F.unfold(x, kernel_size=kernel_size)
        patches = patches.view(batch_size, in_channels * kernel_size * kernel_size, out_h, out_w)
        
        # Apply position-specific weights
        # weights: (out_channels, in_channels * kernel^2, out_h, out_w)
        # Result: (batch, out_channels, out_h, out_w)
        output = (patches.unsqueeze(1) * weights.unsqueeze(0)).sum(dim=2)
        output = output + bias.unsqueeze(0)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self._locally_connected_forward(x, self.local_weights1, self.local_bias1)
        x = F.relu(x)
        x = self.pool(x)
        
        # Layer 2
        x = self._locally_connected_forward(x, self.local_weights2, self.local_bias2)
        x = F.relu(x)
        x = self.pool(x)
        
        # FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class FCTiedLeNet(nn.Module):
    """
    LeNet with global convolutions (filters cover entire input) but tied weights.
    
    Uses very large kernels that span the full image, but weights are still shared
    across positions (with 'same' padding).
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28
    ):
        super().__init__()
        
        # Global convolution: kernel covers (2*input_size - 1) with same padding
        # to mimic looking at the entire image
        kernel1 = 2 * input_size - 1
        pad1 = (kernel1 - 1) // 2
        
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=kernel1, padding=pad1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After first pool
        size1 = input_size // 2
        kernel2 = size1  # Cover the full feature map
        pad2 = (kernel2 - 1) // 2
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=kernel2, padding=pad2)
        
        # After second pool
        size2 = size1 // 2
        
        # FC layers
        self.fc1 = nn.Linear(16 * size2 * size2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FCLeNet(nn.Module):
    """
    Fully connected LeNet (no local connections, no weight tying).
    
    Uses the same number of hidden units as LeNet but with FC layers.
    This is essentially an FC network with LeNet's layer sizes.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28
    ):
        super().__init__()
        
        input_dim = input_channels * input_size * input_size
        
        # Approximate LeNet hidden sizes
        # Conv1 output: 6 * 24 * 24 = 3456 (MNIST)
        # Pool1 output: 6 * 12 * 12 = 864
        # Conv2 output: 16 * 8 * 8 = 1024
        # Pool2 output: 16 * 4 * 4 = 256
        
        conv1_out = 6 * (input_size - 4) * (input_size - 4)
        pool1_out = conv1_out // 4
        conv2_out = 16 * ((input_size - 4) // 2 - 4) * ((input_size - 4) // 2 - 4)
        pool2_out = conv2_out // 4
        
        self.fc1 = nn.Linear(input_dim, conv1_out)
        self.fc2 = nn.Linear(conv1_out, pool1_out)  # Simulates pooling
        self.fc3 = nn.Linear(pool1_out, conv2_out)
        self.fc4 = nn.Linear(conv2_out, pool2_out)  # Simulates pooling
        self.fc5 = nn.Linear(pool2_out, 120)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x
