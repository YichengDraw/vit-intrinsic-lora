"""
One-click unattended pipeline:
1) smoke test
2) automatic hyperparameter tuning
3) full experiment plan

All outputs stay under results/ by default.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List


def run_cmd(cmd: List[str], cwd: str, dry_run: bool = False) -> int:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[run] {pretty}")
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=cwd)
    return completed.returncode


def read_tuning_best(summary_path: str) -> Dict[str, float]:
    """
    Parse tuning summary and return best lr/wd (+ lora alpha ratio) per mode.
    """
    defaults = {
        "lr_full": 5e-5,
        "wd_full": 0.05,
        "lr_linear": 1e-3,
        "wd_linear": 0.0,
        "lr_lora": 3e-4,
        "wd_lora": 0.0,
        "lr_id": 1e-2,
        "wd_id": 0.0,
        "lora_alpha_ratio": 2.0,
    }
    if not os.path.exists(summary_path):
        return defaults

    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    modes = payload.get("modes", {})
    full_best = (modes.get("full", {}) or {}).get("best")
    linear_best = (modes.get("linear", {}) or {}).get("best")
    lora_best = (modes.get("lora", {}) or {}).get("best")
    id_best = (modes.get("id_module", {}) or {}).get("best")

    if full_best and "params" in full_best:
        defaults["lr_full"] = float(full_best["params"]["lr"])
        defaults["wd_full"] = float(full_best["params"]["weight_decay"])
    if linear_best and "params" in linear_best:
        defaults["lr_linear"] = float(linear_best["params"]["lr"])
        defaults["wd_linear"] = float(linear_best["params"]["weight_decay"])
    if lora_best and "params" in lora_best:
        defaults["lr_lora"] = float(lora_best["params"]["lr"])
        defaults["wd_lora"] = float(lora_best["params"]["weight_decay"])
        rank = float(lora_best["params"].get("lora_rank", 0.0))
        alpha = float(lora_best["params"].get("lora_alpha", 0.0))
        if rank > 0:
            defaults["lora_alpha_ratio"] = alpha / rank
    if id_best and "params" in id_best:
        defaults["lr_id"] = float(id_best["params"]["lr"])
        defaults["wd_id"] = float(id_best["params"]["weight_decay"])

    return defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click unattended ViT pipeline")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="results/vit_intrinsic")
    parser.add_argument("--tune_output_dir", type=str, default="results/vit_intrinsic_tuning")
    parser.add_argument("--auto_output_dir", type=str, default="results/auto_runs")

    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable timm pretrained download in all stages.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Optional local torch checkpoint path passed to tuning/full stages.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selector for smoke stage only.",
    )
    parser.add_argument("--dataset_main", type=str, default="cifar100", choices=["cifar100", "flowers102"])
    parser.add_argument("--dataset_aux", type=str, default="flowers102", choices=["cifar100", "flowers102"])
    parser.add_argument("--seed", type=int, default=42, help="Seed for tuning summary naming.")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Seeds for final plan runs.")
    parser.add_argument("--lora_scope", type=str, default="qkvo", choices=["qv", "qkvo", "qkvo_mlp"])
    parser.add_argument("--lora_ranks", type=str, default="2,4,8,16,32")
    parser.add_argument("--full_id_dims", type=str, default="100000,300000,1000000,3000000")
    parser.add_argument("--gpu_profile", type=str, default="none", choices=["none", "rtx4070_8g"])
    parser.add_argument("--batch_size_main", type=int, default=128)
    parser.add_argument("--batch_size_aux", type=int, default=64)
    parser.add_argument("--batch_size_full_id", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_accum_steps_full_id", type=int, default=0)
    parser.add_argument("--grad_checkpoint", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--cleanup_checkpoint", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--skip_tuning", action="store_true")
    parser.add_argument("--skip_full_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue even if a stage fails.")

    # Smoke settings
    parser.add_argument("--smoke_model", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--smoke_modes", type=str, default="full,linear,lora,id_module,id_full")
    parser.add_argument("--smoke_epochs", type=int, default=1)
    parser.add_argument("--smoke_num_batches", type=int, default=2)
    parser.add_argument("--smoke_batch_size", type=int, default=2)

    # Tuning settings
    parser.add_argument("--tune_modes", type=str, default="full,linear,lora,id_module")
    parser.add_argument("--tune_stage_epochs", type=str, default="3,6,12")
    parser.add_argument("--tune_keep_ratio", type=float, default=0.5)
    parser.add_argument("--tune_min_keep", type=int, default=2)
    parser.add_argument("--tune_batch_size", type=int, default=128)
    parser.add_argument("--tune_grad_accum_steps", type=int, default=1)
    parser.add_argument("--tune_max_train_batches", type=int, default=64)
    parser.add_argument("--tune_max_test_batches", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu_profile == "rtx4070_8g":
        # Apply only if user kept defaults.
        if args.tune_batch_size == 128:
            args.tune_batch_size = 16
        if args.tune_grad_accum_steps == 1:
            args.tune_grad_accum_steps = 4
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
            "[preset] rtx4070_8g -> tune_batch_size={} tune_grad_accum_steps={} "
            "batch_size_main={} batch_size_aux={} batch_size_full_id={} grad_accum_steps={} "
            "grad_accum_steps_full_id={} grad_checkpoint={}".format(
                args.tune_batch_size,
                args.tune_grad_accum_steps,
                args.batch_size_main,
                args.batch_size_aux,
                args.batch_size_full_id,
                args.grad_accum_steps,
                args.grad_accum_steps_full_id,
                args.grad_checkpoint,
            )
        )
    if args.tune_grad_accum_steps < 1:
        raise ValueError(f"--tune_grad_accum_steps must be >= 1, got {args.tune_grad_accum_steps}")
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad_accum_steps must be >= 1, got {args.grad_accum_steps}")
    if args.grad_accum_steps_full_id < 0:
        raise ValueError(f"--grad_accum_steps_full_id must be >= 0, got {args.grad_accum_steps_full_id}")
    if args.batch_size_full_id < 0:
        raise ValueError(f"--batch_size_full_id must be >= 0, got {args.batch_size_full_id}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(project_root, "scripts")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.auto_output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    manifest_path = os.path.join(run_dir, "manifest.json")

    manifest: Dict[str, object] = {
        "timestamp": ts,
        "args": vars(args),
        "steps": [],
    }

    def record_step(name: str, cmd: List[str], rc: int) -> None:
        manifest["steps"].append(
            {
                "name": name,
                "command": cmd,
                "return_code": rc,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    print("=" * 80)
    print("One-Click ViT Pipeline")
    print("=" * 80)
    print(f"run_dir={run_dir}")
    print(f"dry_run={args.dry_run}")

    # Step 1: smoke test
    if not args.skip_smoke:
        smoke_cmd = [
            args.python_bin,
            os.path.join(scripts_dir, "test_vit_intrinsic_pipeline.py"),
            "--model_name", args.smoke_model,
            "--modes", args.smoke_modes,
            "--epochs", str(args.smoke_epochs),
            "--num_batches", str(args.smoke_num_batches),
            "--batch_size", str(args.smoke_batch_size),
            "--device", args.device,
        ]
        rc = run_cmd(smoke_cmd, cwd=project_root, dry_run=args.dry_run)
        record_step("smoke_test", smoke_cmd, rc)
        if rc != 0 and not args.continue_on_error:
            raise SystemExit(rc)

    # Step 2: tuning
    if not args.skip_tuning:
        tune_cmd = [
            args.python_bin,
            os.path.join(scripts_dir, "tune_vit_hparams.py"),
            "--python_bin", args.python_bin,
            "--modes", args.tune_modes,
            "--dataset", args.dataset_main,
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--tune_output_dir", args.tune_output_dir,
            "--model_name", args.model_name,
            "--seed", str(args.seed),
            "--batch_size", str(args.tune_batch_size),
            "--grad_accum_steps", str(args.tune_grad_accum_steps),
            "--num_workers", str(args.num_workers),
            "--stage_epochs", args.tune_stage_epochs,
            "--keep_ratio", str(args.tune_keep_ratio),
            "--min_keep", str(args.tune_min_keep),
            "--max_train_batches", str(args.tune_max_train_batches),
            "--max_test_batches", str(args.tune_max_test_batches),
            "--lora_scope", args.lora_scope,
        ]
        if args.no_pretrained:
            tune_cmd.append("--no_pretrained")
        if args.checkpoint_path:
            tune_cmd.extend(["--checkpoint_path", args.checkpoint_path])
        if args.no_amp:
            tune_cmd.append("--no_amp")
        if args.grad_checkpoint:
            tune_cmd.append("--grad_checkpoint")
        if args.dry_run:
            tune_cmd.append("--dry_run")
        rc = run_cmd(tune_cmd, cwd=project_root, dry_run=args.dry_run)
        record_step("auto_tuning", tune_cmd, rc)
        if rc != 0 and not args.continue_on_error:
            raise SystemExit(rc)

    # Read tuned hyperparameters (if summary exists).
    model_tag = args.model_name.replace("/", "_")
    summary_path = os.path.join(
        args.tune_output_dir,
        f"tuning_summary__{args.dataset_main}__{model_tag}__seed{args.seed}.json",
    )
    tuned = read_tuning_best(summary_path)
    manifest["tuned"] = tuned
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    # Step 3: full run
    if not args.skip_full_run:
        full_cmd = [
            args.python_bin,
            os.path.join(scripts_dir, "run_vit_intrinsic_plan.py"),
            "--phase", "all",
            "--dataset_main", args.dataset_main,
            "--dataset_aux", args.dataset_aux,
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--model_name", args.model_name,
            "--lora_scope", args.lora_scope,
            "--seeds", args.seeds,
            "--lora_ranks", args.lora_ranks,
            "--full_id_dims", args.full_id_dims,
            "--gpu_profile", args.gpu_profile,
            "--batch_size_main", str(args.batch_size_main),
            "--batch_size_aux", str(args.batch_size_aux),
            "--batch_size_full_id", str(args.batch_size_full_id),
            "--grad_accum_steps", str(args.grad_accum_steps),
            "--grad_accum_steps_full_id", str(args.grad_accum_steps_full_id),
            "--num_workers", str(args.num_workers),
            "--checkpoint_every", str(args.checkpoint_every),
            "--lr_full", str(tuned["lr_full"]),
            "--wd_full", str(tuned["wd_full"]),
            "--lr_linear", str(tuned["lr_linear"]),
            "--wd_linear", str(tuned["wd_linear"]),
            "--lr_lora", str(tuned["lr_lora"]),
            "--wd_lora", str(tuned["wd_lora"]),
            "--lr_id", str(tuned["lr_id"]),
            "--wd_id", str(tuned["wd_id"]),
            "--lora_alpha_ratio", str(tuned["lora_alpha_ratio"]),
        ]
        if args.no_pretrained:
            full_cmd.append("--no_pretrained")
        if args.checkpoint_path:
            full_cmd.extend(["--checkpoint_path", args.checkpoint_path])
        if args.no_amp:
            full_cmd.append("--no_amp")
        if args.grad_checkpoint:
            full_cmd.append("--grad_checkpoint")
        if args.resume:
            full_cmd.append("--resume")
        if args.cleanup_checkpoint:
            full_cmd.append("--cleanup_checkpoint")
        if args.dry_run:
            full_cmd.append("--dry_run")

        rc = run_cmd(full_cmd, cwd=project_root, dry_run=args.dry_run)
        record_step("full_plan", full_cmd, rc)
        if rc != 0 and not args.continue_on_error:
            raise SystemExit(rc)

    print("\nPipeline completed.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
