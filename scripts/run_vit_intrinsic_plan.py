"""
Unattended runner for the ViT + LoRA + Intrinsic Dimension plan.

This script launches many short runs (one config each) and relies on
--resume in experiments/vit_intrinsic_lora.py to safely continue after
interruptions.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from glob import glob
from typing import Dict, List

import timm

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import estimate_lora_trainable_params  # noqa: E402


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_cmd(cmd: List[str], cwd: str, dry_run: bool = False) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[run] {pretty}")
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {pretty}")


def latest_lora_best_rank(
    output_dir: str,
    dataset: str,
    model_name: str,
    seed: int,
    lora_scope: str,
) -> int:
    """
    Pick best rank from existing LoRA json files for a given seed.
    Returns -1 if no matching file exists.
    """
    lora_dir = os.path.join(output_dir, dataset, "lora")
    if not os.path.exists(lora_dir):
        return -1

    model_tag = model_name.replace("/", "_")
    pattern = os.path.join(lora_dir, f"{dataset}__lora__{model_tag}__scope-{lora_scope}__r-*__seed-{seed}.json")
    files = glob(pattern)
    if not files:
        return -1

    best_rank = -1
    best_acc = -1.0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            rank = int(data["args"]["lora_rank"])
            acc = float(data["metrics"]["best_test_acc"])
        except Exception:
            continue
        if acc > best_acc:
            best_acc = acc
            best_rank = rank
    return best_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViT intrinsic-dimension plan")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--phase", type=str, default="all", choices=["all", "baselines", "module_sweep", "aux", "full_id"])

    parser.add_argument("--dataset_main", type=str, default="cifar100", choices=["cifar100", "flowers102"])
    parser.add_argument("--dataset_aux", type=str, default="flowers102", choices=["cifar100", "flowers102"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="results/vit_intrinsic")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable timm pretrained download.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Optional local torch checkpoint path passed to vit_intrinsic_lora.py.",
    )
    parser.add_argument("--lora_scope", type=str, default="qkvo", choices=["qv", "qkvo", "qkvo_mlp"])
    parser.add_argument("--lora_alpha_ratio", type=float, default=2.0, help="LoRA alpha = ratio * rank")

    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--lora_ranks", type=str, default="2,4,8,16,32")
    parser.add_argument("--full_id_dims", type=str, default="100000,300000,1000000,3000000")

    parser.add_argument("--epochs_full", type=int, default=20)
    parser.add_argument("--epochs_linear", type=int, default=15)
    parser.add_argument("--epochs_sweep", type=int, default=15)
    parser.add_argument("--epochs_aux", type=int, default=12)
    parser.add_argument("--epochs_full_id", type=int, default=10)

    # Tunable optimization hyperparameters (used by all runs in each family).
    parser.add_argument("--lr_full", type=float, default=5e-5)
    parser.add_argument("--wd_full", type=float, default=0.05)
    parser.add_argument("--lr_linear", type=float, default=1e-3)
    parser.add_argument("--wd_linear", type=float, default=0.0)
    parser.add_argument("--lr_lora", type=float, default=3e-4)
    parser.add_argument("--wd_lora", type=float, default=0.0)
    parser.add_argument("--lr_id", type=float, default=1e-2)
    parser.add_argument("--wd_id", type=float, default=0.0)

    parser.add_argument("--batch_size_main", type=int, default=128)
    parser.add_argument("--batch_size_aux", type=int, default=64)
    parser.add_argument(
        "--batch_size_full_id",
        type=int,
        default=0,
        help="Batch size for id_full sweep (0 -> use batch_size_main).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for full/linear/lora/id_module.",
    )
    parser.add_argument(
        "--grad_accum_steps_full_id",
        type=int,
        default=0,
        help="Gradient accumulation steps for id_full (0 -> use grad_accum_steps).",
    )
    parser.add_argument(
        "--grad_checkpoint",
        action="store_true",
        help="Enable model gradient checkpointing for all runs when supported.",
    )
    parser.add_argument(
        "--gpu_profile",
        type=str,
        default="none",
        choices=["none", "rtx4070_8g"],
        help="Optional preset that adjusts memory-sensitive defaults.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print all commands without executing.")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Pass-through for epoch checkpoint interval.")
    parser.add_argument("--cleanup_checkpoint", action="store_true", help="Remove checkpoints after successful runs.")

    args = parser.parse_args()

    if args.gpu_profile == "rtx4070_8g":
        # Apply only when user did not already override these values.
        if args.batch_size_main == 128:
            args.batch_size_main = 16
        if args.batch_size_aux == 64:
            args.batch_size_aux = 16
        if args.batch_size_full_id == 0:
            args.batch_size_full_id = 8
        if args.grad_accum_steps == 1:
            args.grad_accum_steps = 4
        if args.grad_accum_steps_full_id == 0:
            args.grad_accum_steps_full_id = 8
        args.grad_checkpoint = True
        print(
            "[preset] rtx4070_8g -> batch_size_main={} batch_size_aux={} "
            "batch_size_full_id={} grad_accum_steps={} grad_accum_steps_full_id={} grad_checkpoint={}".format(
                args.batch_size_main,
                args.batch_size_aux,
                args.batch_size_full_id,
                args.grad_accum_steps,
                args.grad_accum_steps_full_id,
                args.grad_checkpoint,
            )
        )

    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad_accum_steps must be >= 1, got {args.grad_accum_steps}")
    if args.grad_accum_steps_full_id < 0:
        raise ValueError(f"--grad_accum_steps_full_id must be >= 0, got {args.grad_accum_steps_full_id}")
    if args.batch_size_full_id < 0:
        raise ValueError(f"--batch_size_full_id must be >= 0, got {args.batch_size_full_id}")

    batch_size_full_id = args.batch_size_full_id if args.batch_size_full_id > 0 else args.batch_size_main
    grad_accum_steps_full_id = (
        args.grad_accum_steps_full_id if args.grad_accum_steps_full_id > 0 else args.grad_accum_steps
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiment_script = os.path.join(project_root, "experiments", "vit_intrinsic_lora.py")

    seeds = parse_csv_ints(args.seeds)
    lora_ranks = parse_csv_ints(args.lora_ranks)
    full_id_dims = parse_csv_ints(args.full_id_dims)

    # Estimate module-matched ID dimensions from LoRA trainable-parameter counts.
    num_classes_main = 100 if args.dataset_main == "cifar100" else 102
    tmp_model = timm.create_model(args.model_name, pretrained=False, num_classes=num_classes_main)
    matched_id_dim: Dict[int, int] = {
        rank: estimate_lora_trainable_params(tmp_model, rank=rank, scope=args.lora_scope, train_head=True)
        for rank in lora_ranks
    }
    print("Matched ID dims from LoRA trainable params:")
    for rank in lora_ranks:
        print(f"  rank={rank:<3d} -> d={matched_id_dim[rank]:,}")

    amp_flag = [] if not args.no_amp else ["--no_amp"]
    resume_flag = ["--resume"] if args.resume else []
    cleanup_flag = ["--cleanup_checkpoint"] if args.cleanup_checkpoint else []
    grad_checkpoint_flag = ["--grad_checkpoint"] if args.grad_checkpoint else []

    def dispatch(cmd: List[str]) -> None:
        run_cmd(cmd, cwd=project_root, dry_run=args.dry_run)

    def base_cmd() -> List[str]:
        cmd = [
            args.python_bin,
            experiment_script,
            "--model_name", args.model_name,
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--num_workers", str(args.num_workers),
            "--checkpoint_every", str(args.checkpoint_every),
            "--grad_accum_steps", str(args.grad_accum_steps),
            *grad_checkpoint_flag,
            *amp_flag,
            *resume_flag,
            *cleanup_flag,
        ]
        if args.no_pretrained:
            cmd.append("--no_pretrained")
        if args.checkpoint_path:
            cmd.extend(["--checkpoint_path", args.checkpoint_path])
        return cmd

    def run_main_baselines() -> None:
        for seed in seeds:
            cmd_full = [
                *base_cmd(),
                "--mode", "full",
                "--dataset", args.dataset_main,
                "--seed", str(seed),
                "--epochs", str(args.epochs_full),
                "--batch_size", str(args.batch_size_main),
                "--lr", str(args.lr_full),
                "--weight_decay", str(args.wd_full),
                "--optimizer", "adamw",
            ]
            dispatch(cmd_full)

            cmd_linear = [
                *base_cmd(),
                "--mode", "linear",
                "--dataset", args.dataset_main,
                "--seed", str(seed),
                "--epochs", str(args.epochs_linear),
                "--batch_size", str(args.batch_size_main),
                "--lr", str(args.lr_linear),
                "--weight_decay", str(args.wd_linear),
                "--optimizer", "adamw",
            ]
            dispatch(cmd_linear)

    def run_module_sweep() -> None:
        for seed in seeds:
            for rank in lora_ranks:
                cmd_lora = [
                    *base_cmd(),
                    "--mode", "lora",
                    "--dataset", args.dataset_main,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs_sweep),
                    "--batch_size", str(args.batch_size_main),
                    "--lr", str(args.lr_lora),
                    "--weight_decay", str(args.wd_lora),
                    "--optimizer", "adamw",
                    "--lora_scope", args.lora_scope,
                    "--lora_rank", str(rank),
                    "--lora_alpha", str(args.lora_alpha_ratio * rank),
                ]
                dispatch(cmd_lora)

                cmd_id = [
                    *base_cmd(),
                    "--mode", "id_module",
                    "--dataset", args.dataset_main,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs_sweep),
                    "--batch_size", str(args.batch_size_main),
                    "--lr", str(args.lr_id),
                    "--weight_decay", str(args.wd_id),
                    "--optimizer", "adamw",
                    "--projection", "fastfood",
                    "--id_scope", args.lora_scope,
                    "--subspace_dim", str(matched_id_dim[rank]),
                ]
                dispatch(cmd_id)

    def run_aux_validation() -> None:
        seed = seeds[0]
        best_rank = latest_lora_best_rank(
            output_dir=args.output_dir,
            dataset=args.dataset_main,
            model_name=args.model_name,
            seed=seed,
            lora_scope=args.lora_scope,
        )
        if best_rank not in lora_ranks:
            best_rank = lora_ranks[min(2, len(lora_ranks) - 1)]
        aux_dim = matched_id_dim[best_rank]
        print(f"Aux dataset rank choice: best_rank={best_rank}, matched_d={aux_dim}")

        cmd_full = [
            *base_cmd(),
            "--mode", "full",
            "--dataset", args.dataset_aux,
            "--seed", str(seed),
            "--epochs", str(args.epochs_aux),
            "--batch_size", str(args.batch_size_aux),
            "--lr", str(args.lr_full),
            "--weight_decay", str(args.wd_full),
            "--optimizer", "adamw",
        ]
        dispatch(cmd_full)

        cmd_lora = [
            *base_cmd(),
            "--mode", "lora",
            "--dataset", args.dataset_aux,
            "--seed", str(seed),
            "--epochs", str(args.epochs_aux),
            "--batch_size", str(args.batch_size_aux),
            "--lr", str(args.lr_lora),
            "--weight_decay", str(args.wd_lora),
            "--optimizer", "adamw",
            "--lora_scope", args.lora_scope,
            "--lora_rank", str(best_rank),
            "--lora_alpha", str(args.lora_alpha_ratio * best_rank),
        ]
        dispatch(cmd_lora)

        cmd_id = [
            *base_cmd(),
            "--mode", "id_module",
            "--dataset", args.dataset_aux,
            "--seed", str(seed),
            "--epochs", str(args.epochs_aux),
            "--batch_size", str(args.batch_size_aux),
            "--lr", str(args.lr_id),
            "--weight_decay", str(args.wd_id),
            "--optimizer", "adamw",
            "--projection", "fastfood",
            "--id_scope", args.lora_scope,
            "--subspace_dim", str(aux_dim),
        ]
        dispatch(cmd_id)

    def run_full_id_sweep() -> None:
        seed = seeds[0]
        for dim in full_id_dims:
            cmd = [
                *base_cmd(),
                "--mode", "id_full",
                "--dataset", args.dataset_main,
                "--seed", str(seed),
                "--epochs", str(args.epochs_full_id),
                "--batch_size", str(batch_size_full_id),
                "--grad_accum_steps", str(grad_accum_steps_full_id),
                "--lr", str(args.lr_id),
                "--weight_decay", str(args.wd_id),
                "--optimizer", "adamw",
                "--projection", "fastfood",
                "--subspace_dim", str(dim),
            ]
            dispatch(cmd)

    if args.phase in {"all", "baselines"}:
        run_main_baselines()
    if args.phase in {"all", "module_sweep"}:
        run_module_sweep()
    if args.phase in {"all", "aux"}:
        run_aux_validation()
    if args.phase in {"all", "full_id"}:
        run_full_id_sweep()

    if args.dry_run:
        print("\nDry run completed (no commands executed).")
    else:
        print("\nAll requested phases completed.")


if __name__ == "__main__":
    main()
