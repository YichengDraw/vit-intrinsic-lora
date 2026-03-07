"""
ResNet for CIFAR-10 intrinsic dimension experiments.

CIFAR-10 ResNet follows He et al. architecture:
- Initial 3x3 conv
- 3 stages with 2n blocks each (n=3 for ResNet-20, n=5 for ResNet-32)
- Each stage: 16, 32, 64 filters
- Downsample by stride-2 at stage transitions
- Global average pooling + FC

Paper uses ResNet for CIFAR-10 with d_int90 between 20k-50k.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BasicBlock(nn.Module):
    """
    Basic residual block for CIFAR ResNets.
    
    Two 3x3 conv layers with skip connection.
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet for CIFAR-10/100.
    
    Args:
        num_blocks: Number of blocks per stage [n, n, n]
        num_classes: Number of output classes
        in_channels: Input image channels
    """
    
    def __init__(
        self,
        num_blocks: List[int],
        num_classes: int = 10,
        in_channels: int = 3
    ):
        super().__init__()
        
        self.in_planes = 16
        
        # Initial convolution
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        
        # Three stages
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        """Create a residual layer with multiple blocks."""
        downsample = None
        
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * BasicBlock.expansion
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights following He et al."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions for standard configurations

def ResNet20(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-20 for CIFAR (3 blocks per stage, total 20 layers)."""
    return ResNetCIFAR([3, 3, 3], num_classes=num_classes)


def ResNet32(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-32 for CIFAR (5 blocks per stage, total 32 layers)."""
    return ResNetCIFAR([5, 5, 5], num_classes=num_classes)


def ResNet44(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-44 for CIFAR (7 blocks per stage, total 44 layers)."""
    return ResNetCIFAR([7, 7, 7], num_classes=num_classes)


def ResNet56(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-56 for CIFAR (9 blocks per stage, total 56 layers)."""
    return ResNetCIFAR([9, 9, 9], num_classes=num_classes)
