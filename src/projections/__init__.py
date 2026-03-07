"""
Random projection methods for subspace training.

Three projection types:
1. Dense: P_ij ~ N(0, 1/sqrt(d)) - Standard random projection
2. Sparse: P_ij ~ {-1, 0, +1} with prob {1/6, 2/3, 1/6} - Memory efficient
3. Fastfood: P = S H G Π H B - Structured, O(D log D) complexity
"""

from .dense import DenseProjection
from .sparse import SparseProjection
from .fastfood import FastfoodProjection

__all__ = ["DenseProjection", "SparseProjection", "FastfoodProjection"]
