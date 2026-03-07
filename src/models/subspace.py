"""
SubspaceModel: Core wrapper for training in random subspaces.

Instead of training all D parameters θ, we train a d-dimensional vector
in a random subspace: θ = θ₀ + P @ d

where:
  - θ₀: Initial (frozen) parameters
  - P: Random projection matrix (D × d)
  - d: Trainable low-dimensional vector

This allows measuring the intrinsic dimension of objective landscapes.

Compatible with PyTorch 1.8.0+
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple

from ..projections import DenseProjection, SparseProjection, FastfoodProjection


# Check PyTorch version for optimal implementation
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
_HAS_FUNCTIONAL_CALL = _TORCH_VERSION >= (2, 0)

# Import functional_call if available (PyTorch 2.0+)
functional_call = None
if _HAS_FUNCTIONAL_CALL:
    try:
        from torch.func import functional_call
    except ImportError:
        _HAS_FUNCTIONAL_CALL = False


def _get_module_by_name(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """
    Get the module and parameter name for a dotted parameter path.
    
    For example, 'layer1.conv.weight' returns (model.layer1.conv, 'weight')
    """
    parts = name.split('.')
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


class SubspaceModel(nn.Module):
    """
    Wraps any PyTorch model to train in a low-dimensional random subspace.
    
    Args:
        model: Base model whose parameters will be projected
        subspace_dim: Dimension d of the trainable subspace
        projection_type: Type of random projection ('dense', 'sparse', 'fastfood')
        seed: Random seed for reproducible projections
        init_scale: Scale of initial theta (default 0 = start at θ₀)
        trainable_param_names: Optional explicit parameter-name allowlist for projection.
            If None, project all parameters (original behavior).
    """
    
    def __init__(
        self,
        model: nn.Module,
        subspace_dim: int,
        projection_type: str = 'dense',  # 'dense', 'sparse', 'fastfood'
        seed: int = 42,
        init_scale: float = 0.0,
        trainable_param_names: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.subspace_dim = subspace_dim
        self.projection_type = projection_type
        
        # Store the base model (frozen)
        self.model = model
        
        # Extract and store initial parameters θ₀
        self._param_shapes: Dict[str, torch.Size] = {}
        self._param_numels: Dict[str, int] = {}
        self._params_0: Dict[str, torch.Tensor] = {}
        self._buffers_0: Dict[str, torch.Tensor] = {}
        
        # Store parameter names in order for consistent iteration
        self._param_names: List[str] = []
        
        full_total_params = 0
        for name, param in model.named_parameters():
            self._param_names.append(name)
            self._param_shapes[name] = param.shape
            self._param_numels[name] = param.numel()
            self._params_0[name] = param.detach().clone()
            full_total_params += param.numel()
        
        for name, buffer in model.named_buffers():
            self._buffers_0[name] = buffer.detach().clone()
        
        # Determine which parameters are projected by P @ theta.
        if trainable_param_names is None:
            self._projected_param_names = list(self._param_names)
        else:
            requested = set(trainable_param_names)
            unknown = sorted(requested.difference(self._param_names))
            if unknown:
                raise ValueError(f"Unknown parameter names in trainable_param_names: {unknown[:5]}")
            self._projected_param_names = [name for name in self._param_names if name in requested]
            if not self._projected_param_names:
                raise ValueError("No parameters selected for projection")
        
        self._projected_param_set = set(self._projected_param_names)
        self.base_total_params = full_total_params
        self.total_params = sum(self._param_numels[name] for name in self._projected_param_names)
        
        self._projected_offsets: Dict[str, int] = {}
        running_offset = 0
        for name in self._projected_param_names:
            self._projected_offsets[name] = running_offset
            running_offset += self._param_numels[name]
        
        # Freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Initialize trainable subspace vector θ (d-dimensional)
        # Initialize to 0 so we start exactly at θ₀
        self.theta = nn.Parameter(torch.randn(subspace_dim) * init_scale)
        
        # Create projection P: R^d -> R^D
        self.projection = self._create_projection(
            subspace_dim, self.total_params, projection_type, seed
        )
        
        # Print model info
        compression_ratio = self.total_params / subspace_dim
        print(f"SubspaceModel created:")
        print(f"  Base model params (D_full): {self.base_total_params:,}")
        print(f"  Projected params (D_proj): {self.total_params:,}")
        print(f"  Subspace dim (d): {subspace_dim:,}")
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        print(f"  Projection type: {projection_type}")
        print(f"  Projected parameter tensors: {len(self._projected_param_names)}")
        print(f"  PyTorch version: {torch.__version__} ({'functional_call' if _HAS_FUNCTIONAL_CALL else 'param swap'})")
    
    def _create_projection(
        self, 
        input_dim: int, 
        output_dim: int, 
        projection_type: str, 
        seed: int
    ) -> nn.Module:
        """Create the appropriate projection module."""
        if projection_type == 'dense':
            return DenseProjection(input_dim, output_dim, seed=seed)
        elif projection_type == 'sparse':
            return SparseProjection(input_dim, output_dim, seed=seed)
        elif projection_type == 'fastfood':
            return FastfoodProjection(input_dim, output_dim, seed=seed)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")
    
    def _get_full_params(self) -> Dict[str, torch.Tensor]:
        """
        Compute full parameter dict: θ = θ₀ + P @ d
        
        Returns:
            Dictionary of parameter name -> tensor (with gradient tracking)
        """
        # Project subspace vector to full parameter space
        delta_flat = self.projection(self.theta)
        
        # Reconstruct parameter dictionary
        state_dict = {}
        
        # Add buffers (unchanged)
        for name, buffer in self._buffers_0.items():
            state_dict[name] = buffer.to(delta_flat.device)
        
        # Add parameters (θ₀ + delta) - these tensors maintain gradient tracking!
        for name in self._param_names:
            param_0 = self._params_0[name].to(delta_flat.device)
            if name in self._projected_param_set:
                numel = self._param_numels[name]
                shape = self._param_shapes[name]
                offset = self._projected_offsets[name]
                delta = delta_flat[offset:offset + numel].view(shape)
                # Add to initial parameters - result has requires_grad=True
                state_dict[name] = param_0 + delta
            else:
                # Keep frozen parameters fixed at θ₀.
                state_dict[name] = param_0
        
        return state_dict
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass using subspace-projected parameters.
        
        For PyTorch 2.0+: Uses torch.func.functional_call for efficiency.
        For PyTorch < 2.0: Temporarily swaps parameters in model's _parameters dict.
        """
        params = self._get_full_params()
        
        if _HAS_FUNCTIONAL_CALL and functional_call is not None:
            # PyTorch 2.0+: Use functional_call (most efficient)
            return functional_call(self.model, params, (x,) + args, kwargs)
        else:
            # PyTorch < 2.0: Swap parameters, run forward, restore
            # Key insight: We directly replace tensors in _parameters dict,
            # NOT using .data.copy_() which breaks gradient tracking.
            return self._forward_with_param_swap(x, params, *args, **kwargs)
    
    def _forward_with_param_swap(self, x: torch.Tensor, params: Dict[str, torch.Tensor], 
                                  *args, **kwargs) -> torch.Tensor:
        """
        Forward pass by temporarily swapping model parameters.
        
        This maintains gradient tracking because we directly put our computed
        tensors (which have grad_fn from the projection) into the model's
        _parameters dict.
        """
        # Store original parameters
        original_params = {}
        
        # Replace each parameter with our computed value
        for name in self._param_names:
            module, param_name = _get_module_by_name(self.model, name)
            
            # Store original
            original_params[name] = module._parameters[param_name]
            
            # Replace with computed parameter (maintains gradient tracking!)
            # Note: _parameters dict accepts Tensor, not just Parameter
            module._parameters[param_name] = params[name]  # type: ignore
        
        try:
            # Run forward pass - gradients will flow through params[name] -> theta
            output = self.model(x, *args, **kwargs)
        finally:
            # Restore original parameters (important for model state consistency)
            for name in self._param_names:
                module, param_name = _get_module_by_name(self.model, name)
                module._parameters[param_name] = original_params[name]
        
        return output
    
    def get_effective_params(self) -> torch.Tensor:
        """Return the current full parameter vector θ = θ₀ + P @ d."""
        delta_flat = self.projection(self.theta)
        
        params_flat = []
        for name in self._param_names:
            numel = self._param_numels[name]
            param_0 = self._params_0[name].flatten()
            if name in self._projected_param_set:
                offset = self._projected_offsets[name]
                delta = delta_flat[offset:offset + numel]
                params_flat.append(param_0.to(delta.device) + delta)
            else:
                params_flat.append(param_0.to(delta_flat.device))
        
        return torch.cat(params_flat)
    
    def to(self, *args, **kwargs):
        """Move model and stored parameters to device."""
        # Call parent's to() method
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        self.projection.to(*args, **kwargs)
        
        # Determine device from args/kwargs
        device = None
        if args:
            device = args[0]
        elif 'device' in kwargs:
            device = kwargs['device']
        
        if device is not None:
            # Move stored initial parameters
            for name in self._params_0:
                self._params_0[name] = self._params_0[name].to(device)
            for name in self._buffers_0:
                self._buffers_0[name] = self._buffers_0[name].to(device)
        
        return self
    
    def get_compression_ratio(self) -> float:
        """Return the parameter compression ratio D/d."""
        return self.total_params / self.subspace_dim
    
    def get_projected_param_names(self) -> List[str]:
        """Return parameter names that are projected by the random subspace."""
        return list(self._projected_param_names)
    
    def __repr__(self) -> str:
        return (f"SubspaceModel(\n"
                f"  total_params={self.total_params:,},\n"
                f"  base_total_params={self.base_total_params:,},\n"
                f"  subspace_dim={self.subspace_dim:,},\n"
                f"  projection={self.projection_type}\n"
                f")")
