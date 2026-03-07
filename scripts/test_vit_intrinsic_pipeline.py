"""
Smoke tests for ViT + LoRA + Intrinsic-Dimension pipeline.

This script intentionally uses tiny random datasets to validate:
1) model construction for each mode
2) forward/backward correctness
3) one-epoch training loop execution
4) result metadata sanity (trainable params, histories)

It does NOT aim to validate final accuracy.
"""

import argparse
import os
import sys
import traceback
from argparse import Namespace
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.vit_intrinsic_lora import run_training, set_seed, setup_experiment_model  # noqa: E402


def make_fake_loaders(
    num_classes: int,
    image_size: int,
    batch_size: int,
    num_batches: int,
) -> DataLoader:
    total = batch_size * num_batches
    images = torch.randn(total, 3, image_size, image_size)
    targets = torch.randint(0, num_classes, (total,))
    dataset = TensorDataset(images, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def build_args(
    mode: str,
    model_name: str,
    seed: int,
    subspace_dim: int,
    projection: str,
    id_scope: str,
    lora_scope: str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
) -> Namespace:
    # Keep fields aligned with setup_experiment_model requirements.
    return Namespace(
        mode=mode,
        model_name=model_name,
        no_pretrained=True,  # avoid checkpoint downloads in smoke test
        checkpoint_path="",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_scope=lora_scope,
        id_scope=id_scope,
        subspace_dim=subspace_dim,
        projection=projection,
        seed=seed,
    )


def run_mode(
    mode: str,
    model_name: str,
    num_classes: int,
    image_size: int,
    batch_size: int,
    num_batches: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer: str,
    device: torch.device,
    seed: int,
    subspace_dim: int,
    projection: str,
    id_scope: str,
    lora_scope: str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
) -> Dict:
    set_seed(seed)

    train_loader = make_fake_loaders(
        num_classes=num_classes,
        image_size=image_size,
        batch_size=batch_size,
        num_batches=num_batches,
    )
    test_loader = make_fake_loaders(
        num_classes=num_classes,
        image_size=image_size,
        batch_size=batch_size,
        num_batches=max(1, num_batches // 2),
    )

    args = build_args(
        mode=mode,
        model_name=model_name,
        seed=seed,
        subspace_dim=subspace_dim,
        projection=projection,
        id_scope=id_scope,
        lora_scope=lora_scope,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    model, _ = setup_experiment_model(args, num_classes=num_classes)
    result = run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_name=optimizer,
        use_amp=False,
    )

    if result.epochs_trained != epochs:
        raise RuntimeError(f"{mode}: expected epochs_trained={epochs}, got {result.epochs_trained}")
    if len(result.train_acc_history) != epochs or len(result.test_acc_history) != epochs:
        raise RuntimeError(f"{mode}: history length mismatch")
    if result.trainable_params <= 0:
        raise RuntimeError(f"{mode}: non-positive trainable params")
    if not (0.0 <= result.best_test_acc <= 100.0):
        raise RuntimeError(f"{mode}: invalid best_test_acc={result.best_test_acc}")

    return {
        "mode": mode,
        "trainable_params": result.trainable_params,
        "total_params": result.total_params,
        "best_test_acc": result.best_test_acc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test ViT intrinsic/LoRA pipeline")
    parser.add_argument("--model_name", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--subspace_dim", type=int, default=512)
    parser.add_argument("--projection", type=str, default="fastfood", choices=["dense", "sparse", "fastfood"])
    parser.add_argument("--id_scope", type=str, default="qkvo", choices=["qv", "qkvo", "qkvo_mlp"])

    parser.add_argument("--lora_scope", type=str, default="qkvo", choices=["qv", "qkvo", "qkvo_mlp"])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])

    parser.add_argument(
        "--modes",
        type=str,
        default="full,linear,lora,id_module,id_full",
        help="Comma-separated modes to test",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print("ViT Intrinsic/LoRA Smoke Test")
    print("=" * 80)
    print(f"device={device} model={args.model_name} modes={modes}")

    summaries: List[Dict] = []
    failures: List[str] = []

    for mode in modes:
        print(f"\n[mode={mode}] starting...")
        try:
            summary = run_mode(
                mode=mode,
                model_name=args.model_name,
                num_classes=args.num_classes,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                optimizer=args.optimizer,
                device=device,
                seed=args.seed,
                subspace_dim=args.subspace_dim,
                projection=args.projection,
                id_scope=args.id_scope,
                lora_scope=args.lora_scope,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            summaries.append(summary)
            print(
                "[mode={}] ok | trainable={:,} total={:,} best_acc={:.2f}".format(
                    mode,
                    summary["trainable_params"],
                    summary["total_params"],
                    summary["best_test_acc"],
                )
            )
        except Exception as exc:
            failures.append(mode)
            print(f"[mode={mode}] FAILED: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Smoke Test Summary")
    print("=" * 80)
    for item in summaries:
        print(
            "PASS {:>9s} | trainable={:>10,d} | total={:>10,d} | best_acc={:>6.2f}".format(
                item["mode"],
                item["trainable_params"],
                item["total_params"],
                item["best_test_acc"],
            )
        )
    if failures:
        print(f"FAILED modes: {failures}")
        raise SystemExit(1)
    print("All requested modes passed.")


if __name__ == "__main__":
    main()
