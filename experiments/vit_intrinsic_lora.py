"""
ViT transfer experiment runner for:
1) Full fine-tuning baseline
2) Linear-probe baseline
3) LoRA fine-tuning
4) Intrinsic-dimension (ID) subspace training (module-matched or full-parameter)

Designed for the Winter2 ViT + LoRA + Intrinsic Dimension plan on:
- CIFAR-100
- Flowers-102
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (  # noqa: E402
    SubspaceModel,
    apply_lora_to_timm_vit,
    count_trainable_parameters,
    get_timm_vit_id_param_names,
)
from src.utils.data import get_cifar100_loaders, get_flowers102_loaders  # noqa: E402


_HAS_TORCH_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")


def _amp_autocast(device: torch.device, enabled: bool):
    """Compatibility wrapper for torch.amp.autocast vs torch.cuda.amp.autocast."""
    if _HAS_TORCH_AMP:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _amp_scaler(enabled: bool, device: torch.device):
    """Compatibility wrapper for torch.amp.GradScaler vs torch.cuda.amp.GradScaler."""
    if _HAS_TORCH_AMP and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device=device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _enable_timm_grad_checkpointing(model: nn.Module) -> bool:
    """
    Enable timm gradient checkpointing when supported by the underlying model.

    Returns:
        True if checkpointing was enabled, otherwise False.
    """
    target = model.model if isinstance(model, SubspaceModel) else model
    setter = getattr(target, "set_grad_checkpointing", None)
    if not callable(setter):
        return False
    try:
        setter(True)
        return True
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Warning: failed to enable gradient checkpointing: {exc}")
        return False


@dataclass
class TrainResult:
    best_test_acc: float
    final_test_acc: float
    final_train_acc: float
    total_time_sec: float
    epochs_trained: int
    trainable_params: int
    total_params: int
    train_acc_history: List[float]
    test_acc_history: List[float]
    train_loss_history: List[float]
    test_loss_history: List[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_vit_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    try:
        import timm
    except Exception as exc:
        raise ImportError(
            "timm is required for ViT experiments. Install with: pip install timm>=0.6.7"
        ) from exc

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

        # Common prefix cleanup for DDP checkpoints.
        cleaned = {}
        for key, value in checkpoint.items():
            new_key = key[7:] if key.startswith("module.") else key
            cleaned[new_key] = value

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

    return model


def get_dataset_config(dataset: str) -> Tuple[int, int]:
    """Return (num_classes, default_batch_size)."""
    if dataset == "cifar100":
        return 100, 128
    if dataset == "flowers102":
        return 102, 64
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_loaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    augment: bool,
    image_size: int
):
    if dataset == "cifar100":
        return get_cifar100_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
            image_size=image_size
        )
    if dataset == "flowers102":
        return get_flowers102_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
            image_size=image_size,
            combine_train_val=True
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_head(model: nn.Module) -> None:
    if not hasattr(model, "head"):
        raise ValueError("Model does not expose .head for linear probing")
    for param in model.head.parameters():
        param.requires_grad = True


def setup_experiment_model(args: argparse.Namespace, num_classes: int) -> Tuple[nn.Module, Dict]:
    """
    Build and configure a model according to --mode.
    """
    base_model = build_vit_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        checkpoint_path=args.checkpoint_path,
    )
    model_info: Dict[str, object] = {
        "mode": args.mode,
        "model_name": args.model_name,
        "pretrained": not args.no_pretrained,
        "checkpoint_path": args.checkpoint_path,
    }

    if args.mode == "full":
        # Full fine-tune baseline.
        for param in base_model.parameters():
            param.requires_grad = True
        model = base_model

    elif args.mode == "linear":
        # Linear probe baseline.
        freeze_all(base_model)
        unfreeze_head(base_model)
        model = base_model

    elif args.mode == "lora":
        model = apply_lora_to_timm_vit(
            base_model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            scope=args.lora_scope,
            train_head=True,
        )
        model_info.update(
            {
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_scope": args.lora_scope,
            }
        )

    elif args.mode == "id_module":
        if args.id_scope == "qv":
            print(
                "Warning: timm uses a shared qkv weight matrix; "
                "ID scope 'qv' currently maps to the full qkv weight tensor."
            )
        selected_names = get_timm_vit_id_param_names(
            base_model,
            scope=args.id_scope,
            include_head=True,
        )
        model = SubspaceModel(
            base_model,
            subspace_dim=args.subspace_dim,
            projection_type=args.projection,
            seed=args.seed,
            trainable_param_names=selected_names,
        )
        model_info.update(
            {
                "projection": args.projection,
                "subspace_dim": args.subspace_dim,
                "id_scope": args.id_scope,
                "id_projected_param_count": len(selected_names),
            }
        )

    elif args.mode == "id_full":
        model = SubspaceModel(
            base_model,
            subspace_dim=args.subspace_dim,
            projection_type=args.projection,
            seed=args.seed,
            trainable_param_names=None,  # all parameters
        )
        model_info.update(
            {
                "projection": args.projection,
                "subspace_dim": args.subspace_dim,
                "id_scope": "full",
            }
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if getattr(args, "grad_checkpoint", False):
        # For frozen-backbone modes (linear/lora/id_*), timm's default reentrant
        # checkpoint path can emit "None of the inputs have requires_grad=True"
        # and may drop gradients in checkpointed segments.
        if args.mode != "full":
            model_info["grad_checkpoint"] = False
            model_info["grad_checkpoint_skipped_reason"] = (
                "disabled_for_frozen_backbone_mode_due_to_reentrant_checkpoint_warning"
            )
            print(
                "[opt] Skipped gradient checkpointing for mode={} to avoid potential "
                "gradient loss with frozen backbones.".format(args.mode)
            )
        else:
            enabled = _enable_timm_grad_checkpointing(model)
            model_info["grad_checkpoint"] = enabled
            if enabled:
                print("[opt] Enabled gradient checkpointing.")
            else:
                print("[opt] Gradient checkpointing requested but not supported by this model.")

    return model, model_info


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    use_amp: bool,
    grad_accum_steps: int = 1,
    max_batches: int = 0,
) -> Tuple[float, float]:
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _amp_autocast(device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        reached_batch_cap = max_batches > 0 and (batch_idx + 1) >= max_batches
        is_last_loader_batch = (batch_idx + 1) >= num_batches
        should_step = ((batch_idx + 1) % grad_accum_steps == 0) or reached_batch_cap or is_last_loader_batch
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += logits.argmax(dim=1).eq(targets).sum().item()
        running_total += batch_size

    avg_loss = running_loss / max(running_total, 1)
    avg_acc = 100.0 * running_correct / max(running_total, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 0,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _amp_autocast(device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += logits.argmax(dim=1).eq(targets).sum().item()
        running_total += batch_size

    avg_loss = running_loss / max(running_total, 1)
    avg_acc = 100.0 * running_correct / max(running_total, 1)
    return avg_loss, avg_acc


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optimizer_name: str
) -> torch.optim.Optimizer:
    if hasattr(model, "theta"):
        params = [model.theta]
    else:
        params = [param for param in model.parameters() if param.requires_grad]

    if not params:
        raise RuntimeError("No trainable parameters found.")

    if optimizer_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        # For ViT transfer, SGD is usually only sensible for linear probing.
        momentum = 0.9
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_train_result(
    model: nn.Module,
    best_test_acc: float,
    train_acc_history: List[float],
    test_acc_history: List[float],
    train_loss_history: List[float],
    test_loss_history: List[float],
    total_time_sec: float,
) -> TrainResult:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = count_trainable_parameters(model)
    final_train_acc = train_acc_history[-1] if train_acc_history else 0.0
    final_test_acc = test_acc_history[-1] if test_acc_history else 0.0
    return TrainResult(
        best_test_acc=best_test_acc if test_acc_history else final_test_acc,
        final_test_acc=final_test_acc,
        final_train_acc=final_train_acc,
        total_time_sec=total_time_sec,
        epochs_trained=len(train_acc_history),
        trainable_params=trainable_params,
        total_params=total_params,
        train_acc_history=train_acc_history,
        test_acc_history=test_acc_history,
        train_loss_history=train_loss_history,
        test_loss_history=test_loss_history,
    )


def run_training(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    use_amp: bool,
    grad_accum_steps: int = 1,
    log_interval: int = 1,
    checkpoint_path: str = "",
    resume_from_checkpoint: bool = False,
    save_every_epoch: int = 1,
    max_train_batches: int = 0,
    max_test_batches: int = 0,
) -> TrainResult:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay, optimizer_name=optimizer_name)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = _amp_scaler(enabled=use_amp, device=device)

    train_loss_history: List[float] = []
    train_acc_history: List[float] = []
    test_loss_history: List[float] = []
    test_acc_history: List[float] = []

    start_epoch = 1
    best_test_acc = 0.0
    previous_elapsed = 0.0

    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state"])
            except Exception:
                pass

        history = checkpoint.get("history", {})
        train_loss_history = list(history.get("train_loss", []))
        train_acc_history = list(history.get("train_acc", []))
        test_loss_history = list(history.get("test_loss", []))
        test_acc_history = list(history.get("test_acc", []))
        best_test_acc = float(checkpoint.get("best_test_acc", 0.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        previous_elapsed = float(checkpoint.get("total_time_sec", 0.0))
        print(
            f"[resume] Loaded checkpoint: {checkpoint_path} "
            f"(epoch={start_epoch - 1}, best={best_test_acc:.2f}%)"
        )

    if start_epoch > epochs:
        return _build_train_result(
            model=model,
            best_test_acc=best_test_acc,
            train_acc_history=train_acc_history,
            test_acc_history=test_acc_history,
            train_loss_history=train_loss_history,
            test_loss_history=test_loss_history,
            total_time_sec=previous_elapsed,
        )

    start = time.time()
    progress = tqdm(range(start_epoch, epochs + 1), desc="Training", dynamic_ncols=True)
    for epoch in progress:
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            grad_accum_steps=grad_accum_steps,
            max_batches=max_train_batches,
        )
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
            use_amp,
            max_batches=max_test_batches,
        )
        scheduler.step()

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        best_test_acc = max(best_test_acc, test_acc)

        if epoch % log_interval == 0:
            progress.set_postfix(
                {
                    "train_acc": f"{train_acc:.2f}",
                    "test_acc": f"{test_acc:.2f}",
                    "best": f"{best_test_acc:.2f}",
                }
            )

        if checkpoint_path and save_every_epoch > 0 and (epoch % save_every_epoch == 0 or epoch == epochs):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            ckpt_payload = {
                "epoch": epoch,
                "best_test_acc": best_test_acc,
                "total_time_sec": previous_elapsed + (time.time() - start),
                "history": {
                    "train_loss": train_loss_history,
                    "train_acc": train_acc_history,
                    "test_loss": test_loss_history,
                    "test_acc": test_acc_history,
                },
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
            }
            tmp_path = checkpoint_path + ".tmp"
            torch.save(ckpt_payload, tmp_path)
            os.replace(tmp_path, checkpoint_path)

    elapsed = previous_elapsed + (time.time() - start)
    return _build_train_result(
        model=model,
        best_test_acc=best_test_acc,
        train_acc_history=train_acc_history,
        test_acc_history=test_acc_history,
        train_loss_history=train_loss_history,
        test_loss_history=test_loss_history,
        total_time_sec=elapsed,
    )


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name

    base = [args.dataset, args.mode, args.model_name.replace("/", "_")]
    if args.mode == "lora":
        base.append(f"scope-{args.lora_scope}")
        base.append(f"r-{args.lora_rank}")
    if args.mode in {"id_module", "id_full"}:
        base.append(f"d-{args.subspace_dim}")
        base.append(f"proj-{args.projection}")
        if args.mode == "id_module":
            base.append(f"idscope-{args.id_scope}")
    if args.grad_accum_steps > 1:
        base.append(f"ga-{args.grad_accum_steps}")
    if args.grad_checkpoint:
        base.append("gckpt")
    base.append(f"seed-{args.seed}")
    return "__".join(base)


def save_result(
    output_dir: str,
    run_name: str,
    args: argparse.Namespace,
    model_info: Dict,
    train_result: TrainResult
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{run_name}.json")
    payload = {
        "run_name": run_name,
        "args": vars(args),
        "model_info": model_info,
        "metrics": asdict(train_result),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT + LoRA + Intrinsic Dimension experiment runner")

    # Task setup
    parser.add_argument("--mode", type=str, required=True, choices=["full", "linear", "lora", "id_module", "id_full"])
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "flowers102"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="results/vit_intrinsic")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If final json exists, skip. If checkpoint exists, resume from checkpoint.",
    )

    # Model
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--image_size", type=int, default=224)

    # Train setup
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=0, help="0 -> dataset default")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).",
    )
    parser.add_argument(
        "--grad_checkpoint",
        action="store_true",
        help="Enable timm gradient checkpointing when supported to reduce VRAM usage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (0 disables periodic checkpoints).",
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="Limit train batches per epoch for quick tuning (0 means full epoch).",
    )
    parser.add_argument(
        "--max_test_batches",
        type=int,
        default=0,
        help="Limit eval batches per epoch for quick tuning (0 means full eval).",
    )
    parser.add_argument(
        "--cleanup_checkpoint",
        action="store_true",
        help="Delete run checkpoint after successful completion.",
    )

    # LoRA params
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_scope", type=str, default="qkvo", choices=["qv", "qkvo", "qkvo_mlp"])

    # ID params
    parser.add_argument("--subspace_dim", type=int, default=200000)
    parser.add_argument("--projection", type=str, default="fastfood", choices=["dense", "sparse", "fastfood"])
    parser.add_argument("--id_scope", type=str, default="qkvo", choices=["qv", "qkvo", "qkvo_mlp"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad_accum_steps must be >= 1, got {args.grad_accum_steps}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    num_classes, default_batch_size = get_dataset_config(args.dataset)
    batch_size = args.batch_size if args.batch_size > 0 else default_batch_size
    effective_batch_size = batch_size * args.grad_accum_steps

    run_name = build_run_name(args)
    run_out_dir = os.path.join(args.output_dir, args.dataset, args.mode)
    os.makedirs(run_out_dir, exist_ok=True)
    result_path = os.path.join(run_out_dir, f"{run_name}.json")
    checkpoint_path = os.path.join(run_out_dir, f"{run_name}.ckpt.pt")
    if args.resume and os.path.exists(result_path):
        print(f"[resume] Skip existing run: {result_path}")
        return

    print("=" * 80)
    print("ViT Intrinsic/LoRA Experiment")
    print("=" * 80)
    print(f"mode={args.mode} dataset={args.dataset} model={args.model_name}")
    print(
        "device={} amp={} batch_size={} grad_accum_steps={} effective_batch={} epochs={}".format(
            device,
            use_amp,
            batch_size,
            args.grad_accum_steps,
            effective_batch_size,
            args.epochs,
        )
    )
    print(f"output={result_path}")
    print(f"checkpoint={checkpoint_path}")

    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        augment=not args.no_aug,
        image_size=args.image_size,
    )

    model, model_info = setup_experiment_model(args, num_classes=num_classes)
    train_result = run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        use_amp=use_amp,
        grad_accum_steps=args.grad_accum_steps,
        checkpoint_path=checkpoint_path,
        resume_from_checkpoint=args.resume,
        save_every_epoch=args.checkpoint_every,
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches,
    )

    saved = save_result(
        output_dir=run_out_dir,
        run_name=run_name,
        args=args,
        model_info=model_info,
        train_result=train_result,
    )

    print("\n" + "=" * 80)
    print(f"Saved result: {saved}")
    print(
        "best_test_acc={:.2f}% final_test_acc={:.2f}% trainable_params={:,} total_params={:,}".format(
            train_result.best_test_acc,
            train_result.final_test_acc,
            train_result.trainable_params,
            train_result.total_params,
        )
    )
    if args.cleanup_checkpoint and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
