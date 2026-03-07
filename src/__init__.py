"""
Intrinsic Dimension of Objective Landscapes
Reimplementation of Li et al., ICLR 2018
"""

from .models import SubspaceModel
from .projections import DenseProjection, SparseProjection, FastfoodProjection

__version__ = "1.0.0"
__all__ = ["SubspaceModel", "DenseProjection", "SparseProjection", "FastfoodProjection"]
