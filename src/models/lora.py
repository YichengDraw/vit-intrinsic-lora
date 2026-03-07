"""
LoRA utilities for timm Vision Transformer models.

This file provides:
1) Lightweight LoRA wrappers for nn.Linear and ViT qkv projection.
2) Utility functions to attach LoRA modules to timm ViTs.
3) Helper functions for trainable-parameter accounting and ID matching.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _freeze_module_params(module: nn.Module) -> None:
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a standard nn.Linear layer.

    Effective weight: W + (alpha / r) * B @ A
    where A in R^(r x in), B in R^(out x r).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.base = base_linear
        _freeze_module_params(self.base)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, self.base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.base.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out


class LoRAQKVLinear(nn.Module):
    """
    LoRA wrapper for timm ViT qkv projection (single Linear with 3*dim outputs).

    This allows targeting q/k/v independently while keeping base qkv frozen.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        enable_q: bool = True,
        enable_k: bool = True,
        enable_v: bool = True
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if not (enable_q or enable_k or enable_v):
            raise ValueError("At least one of q/k/v must be enabled")

        self.base = base_linear
        _freeze_module_params(self.base)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.base.out_features % 3 != 0:
            raise ValueError(
                f"Expected qkv out_features multiple of 3, got {self.base.out_features}"
            )

        self.embed_dim = self.base.out_features // 3
        self.in_features = self.base.in_features

        self.enable_map: Dict[str, bool] = {
            "q": enable_q,
            "k": enable_k,
            "v": enable_v,
        }
        self.slice_map: Dict[str, Tuple[int, int]] = {
            "q": (0, self.embed_dim),
            "k": (self.embed_dim, 2 * self.embed_dim),
            "v": (2 * self.embed_dim, 3 * self.embed_dim),
        }

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name, enabled in self.enable_map.items():
            if enabled:
                self.lora_A[name] = nn.Parameter(torch.empty(rank, self.in_features))
                self.lora_B[name] = nn.Parameter(torch.zeros(self.embed_dim, rank))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, enabled in self.enable_map.items():
            if enabled:
                nn.init.kaiming_uniform_(self.lora_A[name], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[name])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta_full = torch.zeros_like(base_out)
        dropped = self.dropout(x)
        for name, enabled in self.enable_map.items():
            if not enabled:
                continue
            start, end = self.slice_map[name]
            delta = F.linear(F.linear(dropped, self.lora_A[name]), self.lora_B[name])
            delta_full[..., start:end] = delta
        return base_out + self.scaling * delta_full


def apply_lora_to_timm_vit(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    scope: str = "qkvo",
    train_head: bool = True
) -> nn.Module:
    """
    Attach LoRA modules to a timm ViT model.

    Supported scopes:
    - qv
    - qkvo
    - qkvo_mlp
    """
    if scope not in {"qv", "qkvo", "qkvo_mlp"}:
        raise ValueError(f"Unsupported LoRA scope: {scope}")

    # Freeze everything first, then explicitly enable LoRA + optional head.
    _freeze_module_params(model)

    enable_q = "q" in scope
    enable_k = "k" in scope
    enable_v = "v" in scope
    enable_o = "o" in scope
    enable_mlp = "mlp" in scope

    if not hasattr(model, "blocks"):
        raise ValueError("Expected timm ViT model with .blocks")

    for block in model.blocks:
        block.attn.qkv = LoRAQKVLinear(
            block.attn.qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            enable_q=enable_q,
            enable_k=enable_k,
            enable_v=enable_v,
        )
        if enable_o:
            block.attn.proj = LoRALinear(
                block.attn.proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
        if enable_mlp:
            block.mlp.fc1 = LoRALinear(
                block.mlp.fc1,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            block.mlp.fc2 = LoRALinear(
                block.mlp.fc2,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    if train_head and hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def get_timm_vit_id_param_names(
    model: nn.Module,
    scope: str = "qkvo",
    include_head: bool = True
) -> List[str]:
    """
    Parameter-name selection for module-matched ID.

    These names define which native weight tensors are projected by SubspaceModel.
    """
    if scope not in {"qv", "qkvo", "qkvo_mlp"}:
        raise ValueError(f"Unsupported scope: {scope}")

    names: List[str] = []
    for name, _ in model.named_parameters():
        # QKV weights and biases
        if name.endswith("attn.qkv.weight"):
            # qkv share one matrix in timm; selecting qv still maps to the same tensor.
            names.append(name)
        elif name.endswith("attn.qkv.bias"):
            names.append(name)
        # Projection weights and biases (for qkvo and qkvo_mlp scopes)
        elif scope in {"qkvo", "qkvo_mlp"} and name.endswith("attn.proj.weight"):
            names.append(name)
        elif scope in {"qkvo", "qkvo_mlp"} and name.endswith("attn.proj.bias"):
            names.append(name)
        # MLP weights and biases (for qkvo_mlp scope only)
        elif scope == "qkvo_mlp" and (name.endswith("mlp.fc1.weight") or name.endswith("mlp.fc2.weight")):
            names.append(name)
        elif scope == "qkvo_mlp" and (name.endswith("mlp.fc1.bias") or name.endswith("mlp.fc2.bias")):
            names.append(name)
        # Head weights and biases
        elif include_head and (name == "head.weight" or name == "head.bias"):
            names.append(name)
    return names


def estimate_lora_trainable_params(
    model: nn.Module,
    rank: int,
    scope: str = "qkvo",
    train_head: bool = True
) -> int:
    """
    Estimate LoRA trainable-parameter count for a timm ViT model.
    """
    if scope not in {"qv", "qkvo", "qkvo_mlp"}:
        raise ValueError(f"Unsupported scope: {scope}")
    if rank <= 0:
        raise ValueError(f"rank must be > 0, got {rank}")

    total = 0
    num_qkv_targets = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module_name.endswith("attn.qkv"):
            # Separate LoRA adapters for q/k/v slices.
            embed_dim = module.out_features // 3
            num_active = 0
            if "q" in scope:
                num_active += 1
            if "k" in scope:
                num_active += 1
            if "v" in scope:
                num_active += 1
            total += num_active * rank * (module.in_features + embed_dim)
            num_qkv_targets += 1
        elif scope in {"qkvo", "qkvo_mlp"} and module_name.endswith("attn.proj"):
            total += rank * (module.in_features + module.out_features)
        elif scope == "qkvo_mlp" and (module_name.endswith("mlp.fc1") or module_name.endswith("mlp.fc2")):
            total += rank * (module.in_features + module.out_features)

    if num_qkv_targets == 0:
        raise ValueError("No qkv linear modules found. Is this a timm ViT model?")

    if train_head and hasattr(model, "head"):
        total += sum(param.numel() for param in model.head.parameters())

    return total
