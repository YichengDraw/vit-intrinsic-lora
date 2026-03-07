"""
Dense Random Projection

Projects from low-dimensional subspace (d) to high-dimensional parameter space (D).
P_ij ~ N(0, 1/sqrt(d)) to preserve distances approximately (Johnson-Lindenstrauss).

For subspace training: θ = θ_0 + P @ d
where d is the trainable low-dimensional vector.
"""

import torch
import torch.nn as nn
import math


class DenseProjection(nn.Module):
    """
    Dense random projection matrix P: R^d -> R^D
    
    Memory: O(d * D) - can be expensive for large D
    Computation: O(d * D) per forward pass
    
    Args:
        input_dim (int): Subspace dimension d
        output_dim (int): Parameter space dimension D
        seed (int): Random seed for reproducibility
        scale (float): Standard deviation of projection entries (default: 1/sqrt(d))
    """
    
    def __init__(self, input_dim: int, output_dim: int, seed: int = 42, scale: float = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Scale to preserve variance: Var(P @ x) ≈ Var(x) when P_ij ~ N(0, 1/d)
        # Paper uses 1/sqrt(d) for standard deviation
        if scale is None:
            scale = 1.0 / math.sqrt(input_dim)
        
        # Generate random projection matrix (frozen, not trainable)
        generator = torch.Generator().manual_seed(seed)
        projection_matrix = torch.randn(output_dim, input_dim, generator=generator) * scale
        
        # Register as buffer (not a parameter, but moves with device)
        self.register_buffer('P', projection_matrix)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project low-dimensional vector to high-dimensional space.
        
        Args:
            x: Tensor of shape (d,) or (batch, d)
            
        Returns:
            Tensor of shape (D,) or (batch, D)
        """
        # Handle both single vector and batched input
        if x.dim() == 1:
            return torch.mv(self.P, x)
        else:
            return torch.mm(x, self.P.t())
    
    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}'
