"""
SqueezeNet for ImageNet intrinsic dimension experiments.

SqueezeNet achieves AlexNet-level accuracy with 50x fewer parameters.
Used in the paper for ImageNet experiments (Supplementary S9).

Architecture:
- Fire modules with squeeze (1x1) and expand (1x1 + 3x3) layers
- ~1.24M parameters
- Paper found d_int90 > 500k for ImageNet

Reference: Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fire(nn.Module):
    """
    Fire module: squeeze layer (1x1 conv) + expand layer (1x1 + 3x3 conv).
    """
    
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int
    ):
        super().__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.squeeze(x))
        return torch.cat([
            F.relu(self.expand1x1(x)),
            F.relu(self.expand3x3(x))
        ], dim=1)


class SqueezeNet(nn.Module):
    """
    SqueezeNet v1.1 (more efficient variant).
    
    Args:
        num_classes: Number of output classes (1000 for ImageNet)
        in_channels: Number of input channels (3 for RGB)
    """
    
    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # Fire2, Fire3
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # Fire4, Fire5
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # Fire6, Fire7, Fire8, Fire9
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        
        # Final convolution
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
