"""
Fastfood Transform for Structured Random Projections

Replaces dense D×d matrix with structured transform: P = S H G Π H B
where:
  - B: Random binary diagonal (±1)
  - H: Walsh-Hadamard transform
  - Π: Random permutation
  - G: Random Gaussian diagonal
  - S: Scaling diagonal

Complexity: O(D log D) instead of O(Dd) for dense projection
Memory: O(D) instead of O(Dd)

Reference: Le, Q., Sarlós, T., & Smola, A. (2013). Fastfood - Approximating Kernel Expansions.
"""

import torch
import torch.nn as nn
import math


def next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def fast_walsh_hadamard_transform(x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform (FWHT) - Vectorized Implementation.
    
    Computes H @ x where H is the Hadamard matrix.
    Uses vectorized butterfly algorithm with O(N log N) complexity.
    Memory-efficient: no per-element clones, all operations are batched.
    
    Args:
        x: Input tensor of shape (..., N) where N must be power of 2
        normalize: If True, divide by sqrt(N) for orthonormal transform
        
    Returns:
        Transformed tensor of same shape
    """
    orig_shape = x.shape
    n = x.size(-1)
    
    # Verify power of 2
    if n & (n - 1) != 0:
        raise ValueError(f"Input size {n} must be a power of 2")
    
    # Reshape for batch processing: (batch, n)
    x = x.reshape(-1, n)
    batch_size = x.size(0)
    
    # Vectorized butterfly algorithm
    # Process all pairs at each level simultaneously
    h = 1
    while h < n:
        # Reshape to (batch, n//(2*h), 2, h) to process pairs
        x = x.view(batch_size, n // (2 * h), 2, h)
        
        # a = first elements of pairs, b = second elements
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        
        # Hadamard butterfly: [a+b, a-b]
        x = torch.stack([a + b, a - b], dim=2)
        
        # Reshape back to (batch, n)
        x = x.view(batch_size, n)
        h *= 2
    
    if normalize:
        x = x / math.sqrt(n)
    
    return x.reshape(orig_shape)


class FastfoodProjection(nn.Module):
    """
    Fastfood structured random projection P: R^d -> R^D
    
    Uses structured matrices instead of dense random matrix.
    
    Memory: O(D) instead of O(d * D)
    Computation: O(D log D) instead of O(d * D)
    
    Args:
        input_dim (int): Subspace dimension d
        output_dim (int): Parameter space dimension D
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Pad to next power of 2 for Hadamard transform
        self.padded_dim = next_power_of_2(max(input_dim, output_dim))
        
        # Set random seed
        generator = torch.Generator().manual_seed(seed)
        
        # B: Binary diagonal (±1)
        B = torch.bernoulli(
            torch.ones(self.padded_dim) * 0.5, 
            generator=generator
        ) * 2 - 1
        self.register_buffer('B', B)
        
        # G: Gaussian diagonal ~ N(0, 1)
        G = torch.randn(self.padded_dim, generator=generator)
        self.register_buffer('G', G)
        
        # Π: Random permutation (store as indices)
        Pi = torch.randperm(self.padded_dim, generator=generator)
        self.register_buffer('Pi', Pi)
        
        # S: Scaling diagonal (for variance normalization)
        # Unnormalized FWHT scales variance by N per application
        # Two FWHT applications: variance scales by N^2
        # To normalize back to unit variance: divide by N
        # This matches the behavior of dense projection for subspace training
        S = torch.ones(self.padded_dim) / self.padded_dim
        self.register_buffer('S', S)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fastfood transform: P @ x = S * H * G * Π * H * B * x
        
        Args:
            x: Tensor of shape (d,) or (batch, d)
            
        Returns:
            Tensor of shape (D,) or (batch, D)
        """
        is_1d = x.dim() == 1
        if is_1d:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        
        # Pad input to padded_dim
        if x.size(-1) < self.padded_dim:
            padded = torch.zeros(batch_size, self.padded_dim, device=x.device, dtype=x.dtype)
            padded[:, :x.size(-1)] = x
            x = padded
        
        # Step 1: Multiply by B (element-wise)
        x = x * self.B
        
        # Step 2: First Hadamard transform
        x = fast_walsh_hadamard_transform(x)
        
        # Step 3: Permute by Π
        x = x[:, self.Pi]
        
        # Step 4: Multiply by G (element-wise)
        x = x * self.G
        
        # Step 5: Second Hadamard transform
        x = fast_walsh_hadamard_transform(x)
        
        # Step 6: Scale by S
        x = x * self.S
        
        # Slice to output dimension
        x = x[:, :self.output_dim]
        
        if is_1d:
            x = x.squeeze(0)
        
        return x
    
    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, padded_dim={self.padded_dim}'
