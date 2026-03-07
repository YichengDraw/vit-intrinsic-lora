"""
Automatic hyperparameter tuning for ViT transfer experiments.

Method:
- Successive Halving over small candidate grids.
- Stage budgets are defined by epoch counts (e.g., 3,6,12).
- Optional per-epoch batch caps keep early tuning cheap.

This script launches experiments/vit_intrinsic_lora.py as subprocesses
and reads their JSON outputs.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ConfigResult:
    config_id: str
    mode: str
    params: Dict[str, object]
    stage: int
    epochs: int
    run_name: str
    result_path: str
    best_test_acc: float
    final_test_acc: float


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def run_cmd(cmd: List[str], cwd: str, dry_run: bool = False) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[run] {pretty}")
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {pretty}")


def load_metrics(result_path: str) -> Tuple[float, float]:
    with open(result_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload["metrics"]
    return float(metrics["best_test_acc"]), float(metrics["final_test_acc"])


def build_search_space(args: argparse.Namespace, mode: str) -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []
    if mode == "full":
        for lr in parse_csv_floats(args.full_lrs):
            for wd in parse_csv_floats(args.full_wds):
                configs.append({"lr": lr, "weight_decay": wd})
    elif mode == "linear":
        for lr in parse_csv_floats(args.linear_lrs):
            for wd in parse_csv_floats(args.linear_wds):
                configs.append({"lr": lr, "weight_decay": wd})
    elif mode == "lora":
        rank = args.lora_rank_for_tuning
        for lr in parse_csv_floats(args.lora_lrs):
            for wd in parse_csv_floats(args.lora_wds):
                for mult in parse_csv_floats(args.lora_alpha_multipliers):
                    alpha = float(rank * mult)
                    configs.append({"lr": lr, "weight_decay": wd, "lora_rank": rank, "lora_alpha": alpha})
    elif mode == "id_module":
        for lr in parse_csv_floats(args.id_lrs):
            for wd in parse_csv_floats(args.id_wds):
                configs.append(
                    {
                        "lr": lr,
                        "weight_decay": wd,
                        "subspace_dim": args.id_dim_for_tuning,
                        "projection": args.projection,
                        "id_scope": args.id_scope,
                    }
                )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if not configs:
        raise ValueError(f"Empty search space for mode={mode}")
    return configs


def launch_config(
    args: argparse.Namespace,
    mode: str,
    config_id: str,
    params: Dict[str, object],
    stage_idx: int,
    epochs: int,
) -> ConfigResult:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_script = os.path.join(project_root, "experiments", "vit_intrinsic_lora.py")

    run_name = f"tune__{mode}__{config_id}__stage{stage_idx}__seed{args.seed}"

    cmd = [
        args.python_bin,
        exp_script,
        "--mode", mode,
        "--dataset", args.dataset,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--run_name", run_name,
        "--model_name", args.model_name,
        "--seed", str(args.seed),
        "--epochs", str(epochs),
        "--num_workers", str(args.num_workers),
        "--batch_size", str(args.batch_size),
        "--optimizer", "adamw",
        "--checkpoint_every", "0",
        "--grad_accum_steps", str(args.grad_accum_steps),
        "--max_train_batches", str(args.max_train_batches),
        "--max_test_batches", str(args.max_test_batches),
    ]
    if args.no_amp:
        cmd.append("--no_amp")
    if args.no_aug:
        cmd.append("--no_aug")
    if args.no_pretrained:
        cmd.append("--no_pretrained")
    if args.checkpoint_path:
        cmd.extend(["--checkpoint_path", args.checkpoint_path])
    if args.grad_checkpoint:
        cmd.append("--grad_checkpoint")

    cmd.extend(["--lr", str(params["lr"])])
    cmd.extend(["--weight_decay", str(params["weight_decay"])])

    if mode == "lora":
        cmd.extend(["--lora_scope", args.lora_scope])
        cmd.extend(["--lora_rank", str(params["lora_rank"])])
        cmd.extend(["--lora_alpha", str(params["lora_alpha"])])
        cmd.extend(["--lora_dropout", str(args.lora_dropout)])
    elif mode == "id_module":
        cmd.extend(["--projection", str(params["projection"])])
        cmd.extend(["--id_scope", str(params["id_scope"])])
        cmd.extend(["--subspace_dim", str(params["subspace_dim"])])

    run_cmd(cmd, cwd=project_root, dry_run=args.dry_run)

    result_path = os.path.join(args.output_dir, args.dataset, mode, f"{run_name}.json")
    if args.dry_run:
        return ConfigResult(
            config_id=config_id,
            mode=mode,
            params=params,
            stage=stage_idx,
            epochs=epochs,
            run_name=run_name,
            result_path=result_path,
            best_test_acc=0.0,
            final_test_acc=0.0,
        )

    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Missing result json: {result_path}")
    best_acc, final_acc = load_metrics(result_path)
    return ConfigResult(
        config_id=config_id,
        mode=mode,
        params=params,
        stage=stage_idx,
        epochs=epochs,
        run_name=run_name,
        result_path=result_path,
        best_test_acc=best_acc,
        final_test_acc=final_acc,
    )


def successive_halving(args: argparse.Namespace, mode: str, configs: List[Dict[str, object]]) -> Dict[str, object]:
    stage_epochs = parse_csv_ints(args.stage_epochs)
    survivors: List[Tuple[str, Dict[str, object]]] = [
        (f"c{idx:03d}", cfg) for idx, cfg in enumerate(configs)
    ]

    summary: Dict[str, object] = {
        "mode": mode,
        "stage_epochs": stage_epochs,
        "initial_candidates": len(configs),
        "stages": [],
        "best": None,
    }

    for stage_idx, epochs in enumerate(stage_epochs, start=1):
        print(f"\n{'=' * 80}\n[mode={mode}] stage={stage_idx} epochs={epochs} candidates={len(survivors)}\n{'=' * 80}")
        stage_results: List[ConfigResult] = []
        for config_id, params in survivors:
            result = launch_config(
                args=args,
                mode=mode,
                config_id=config_id,
                params=params,
                stage_idx=stage_idx,
                epochs=epochs,
            )
            stage_results.append(result)
            print(f"  {config_id} -> best={result.best_test_acc:.3f} final={result.final_test_acc:.3f}")

        if args.dry_run:
            keep_count = max(args.min_keep, math.ceil(len(survivors) * args.keep_ratio))
            if stage_idx == len(stage_epochs):
                keep_count = 1
            survivors = survivors[:keep_count]
            summary["stages"].append(
                {
                    "stage": stage_idx,
                    "epochs": epochs,
                    "results": [r.__dict__ for r in stage_results],
                    "survivors": [cid for cid, _ in survivors],
                }
            )
            continue

        stage_results.sort(key=lambda x: x.best_test_acc, reverse=True)
        keep_count = max(args.min_keep, math.ceil(len(stage_results) * args.keep_ratio))
        if stage_idx == len(stage_epochs):
            keep_count = 1
        keep_count = min(keep_count, len(stage_results))

        kept_ids = {r.config_id for r in stage_results[:keep_count]}
        survivors = [(cid, cfg) for cid, cfg in survivors if cid in kept_ids]

        summary["stages"].append(
            {
                "stage": stage_idx,
                "epochs": epochs,
                "results": [r.__dict__ for r in stage_results],
                "survivors": [cid for cid, _ in survivors],
            }
        )

    if not args.dry_run:
        last_stage = summary["stages"][-1]
        best_result = last_stage["results"][0]
        summary["best"] = best_result
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic hyperparameter tuning for ViT experiments")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--modes", type=str, default="full,linear,lora,id_module")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "flowers102"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="results/vit_intrinsic")
    parser.add_argument("--tune_output_dir", type=str, default="results/vit_intrinsic_tuning")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable timm pretrained download.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Optional local torch checkpoint path passed to vit_intrinsic_lora.py.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for each tuning run.",
    )
    parser.add_argument(
        "--grad_checkpoint",
        action="store_true",
        help="Enable model gradient checkpointing in tuning runs when supported.",
    )
    parser.add_argument("--dry_run", action="store_true")

    # Multi-fidelity schedule.
    parser.add_argument("--stage_epochs", type=str, default="3,6,12")
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--min_keep", type=int, default=2)
    parser.add_argument("--max_train_batches", type=int, default=64)
    parser.add_argument("--max_test_batches", type=int, default=32)

    # Search spaces.
    parser.add_argument("--full_lrs", type=str, default="2e-5,5e-5,1e-4")
    parser.add_argument("--full_wds", type=str, default="0.01,0.05")
    parser.add_argument("--linear_lrs", type=str, default="3e-4,1e-3,3e-3")
    parser.add_argument("--linear_wds", type=str, default="0.0,1e-4")
    parser.add_argument("--lora_lrs", type=str, default="1e-4,3e-4,1e-3")
    parser.add_argument("--lora_wds", type=str, default="0.0,1e-4")
    parser.add_argument("--lora_alpha_multipliers", type=str, default="1,2,4")
    parser.add_argument("--lora_rank_for_tuning", type=int, default=8)
    parser.add_argument("--lora_scope", type=str, default="qkvo", choices=["qv", "qkvo", "qkvo_mlp"])
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--id_lrs", type=str, default="1e-4,3e-4,1e-3,3e-3")
    parser.add_argument("--id_wds", type=str, default="0.0,1e-4")
    parser.add_argument("--id_dim_for_tuning", type=int, default=371812)
    parser.add_argument("--id_scope", type=str, default="qkvo", choices=["qv", "qkvo", "qkvo_mlp"])
    parser.add_argument("--projection", type=str, default="fastfood", choices=["dense", "sparse", "fastfood"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad_accum_steps must be >= 1, got {args.grad_accum_steps}")
    modes = parse_csv_strs(args.modes)
    os.makedirs(args.tune_output_dir, exist_ok=True)

    print("=" * 80)
    print("ViT Hyperparameter Tuning (Successive Halving)")
    print("=" * 80)
    print(f"modes={modes} dataset={args.dataset} model={args.model_name} dry_run={args.dry_run}")
    print(f"stage_epochs={args.stage_epochs} keep_ratio={args.keep_ratio} min_keep={args.min_keep}")
    print(f"max_train_batches={args.max_train_batches} max_test_batches={args.max_test_batches}")

    all_summary: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "modes": {},
    }

    for mode in modes:
        configs = build_search_space(args, mode)
        print(f"\n[mode={mode}] search space size = {len(configs)}")
        summary = successive_halving(args, mode, configs)
        all_summary["modes"][mode] = summary

    out_path = os.path.join(
        args.tune_output_dir,
        f"tuning_summary__{args.dataset}__{args.model_name.replace('/', '_')}__seed{args.seed}.json",
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(all_summary, handle, indent=2)
    print(f"\nSaved tuning summary: {out_path}")


if __name__ == "__main__":
    main()
