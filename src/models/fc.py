"""
Fully Connected Networks for intrinsic dimension experiments.

Paper configurations:
- MNIST: FC(W=200, L=2) has ~199,210 parameters
- CIFAR-10: FC(W=200, L=2) has ~1,055,610 parameters (larger input)
- Shuffled label: FC-5 (5 layers) has ~959,610 parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class FCNetwork(nn.Module):
    """
    Flexible fully connected network.
    
    Args:
        input_dim: Input feature dimension (784 for MNIST, 3072 for CIFAR)
        hidden_dims: List of hidden layer dimensions (default: [200, 200])
        output_dim: Number of output classes (default: 10)
        activation: Activation function ('relu', 'tanh', 'sigmoid')
        dropout: Dropout rate (0 = no dropout)
        batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [200, 200],
        output_dim: int = 10,
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, will be flattened if needed
            
        Returns:
            Logits of shape (batch, output_dim)
        """
        # Flatten input if needed (e.g., from images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions for paper configurations

def fc_mnist(width: int = 200, depth: int = 2, **kwargs) -> FCNetwork:
    """FC network for MNIST (input: 28x28 = 784)."""
    return FCNetwork(
        input_dim=784,
        hidden_dims=[width] * depth,
        output_dim=10,
        **kwargs
    )


def fc_cifar(width: int = 200, depth: int = 2, **kwargs) -> FCNetwork:
    """FC network for CIFAR-10 (input: 32x32x3 = 3072)."""
    return FCNetwork(
        input_dim=3072,
        hidden_dims=[width] * depth,
        output_dim=10,
        **kwargs
    )


def fc_shuffled_label(width: int = 200, **kwargs) -> FCNetwork:
    """FC-5 network for shuffled label experiment (5 hidden layers)."""
    return FCNetwork(
        input_dim=784,
        hidden_dims=[width] * 5,
        output_dim=10,
        **kwargs
    )
