"""
Neural network models for intrinsic dimension experiments.
"""

from .subspace import SubspaceModel
from .fc import FCNetwork
from .lenet import LeNet
from .lenet_variants import UntiedLeNet, FCTiedLeNet, FCLeNet
from .resnet import ResNet20, ResNet32, ResNet44
from .squeezenet import SqueezeNet
from .policy import FCPolicy, ConvPolicy
from .lora import (
    LoRALinear,
    LoRAQKVLinear,
    apply_lora_to_timm_vit,
    count_trainable_parameters,
    get_timm_vit_id_param_names,
    estimate_lora_trainable_params,
)

__all__ = [
    "SubspaceModel",
    "FCNetwork",
    "LeNet",
    "UntiedLeNet",
    "FCTiedLeNet",
    "FCLeNet",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "SqueezeNet",
    "FCPolicy",
    "ConvPolicy",
    "LoRALinear",
    "LoRAQKVLinear",
    "apply_lora_to_timm_vit",
    "count_trainable_parameters",
    "get_timm_vit_id_param_names",
    "estimate_lora_trainable_params",
]
