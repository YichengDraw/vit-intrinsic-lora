# ViT + LoRA + Intrinsic Dimension

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

PyTorch project for transfer-learning experiments that compare full fine-tuning, linear probing, LoRA, and intrinsic-dimension subspace tuning on Vision Transformers.

This repository focuses on a practical question: if LoRA only trains a small set of parameters, can random low-dimensional subspace tuning reach similar performance under a matched trainable-parameter budget?

## Highlights

- ViT transfer experiments on `CIFAR-100` and `Flowers-102`
- Five training modes in one experiment runner:
  - `full`
  - `linear`
  - `lora`
  - `id_module`
  - `id_full`
- LoRA wrappers for timm Vision Transformers
- Module-matched intrinsic-dimension tuning via `SubspaceModel`
- Dense, sparse, and Fastfood projection backends
- Unattended sweep runner, tuning script, and smoke test
- RTX 4070 8GB-friendly preset with gradient accumulation and optional gradient checkpointing

## Repository Structure

```text
vit_intrinsic_lora/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── requirements.vit_py37.txt
├── test_compatibility.py
├── docs/
│   └── vit_id_lora_execution_plan.md
├── experiments/
│   └── vit_intrinsic_lora.py
├── scripts/
│   ├── run_vit_intrinsic_plan.py
│   ├── run_vit_all_in_one.py
│   ├── tune_vit_hparams.py
│   └── test_vit_intrinsic_pipeline.py
└── src/
    ├── models/
    ├── projections/
    └── utils/
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

For older environments such as Python 3.7 / older PyTorch stacks:

```bash
pip install -r requirements.vit_py37.txt
```

## Quick Start

### 1. Compatibility Check

```bash
python test_compatibility.py
```

### 2. Smoke Test The Whole ViT Pipeline

```bash
python scripts/test_vit_intrinsic_pipeline.py \
  --model_name vit_tiny_patch16_224 \
  --modes full,linear,lora,id_module,id_full \
  --epochs 1 \
  --num_batches 2
```

### 3. Preview The Full Experiment Plan

```bash
python scripts/run_vit_intrinsic_plan.py --phase all --dry_run
```

### 4. Run A Single Experiment

Full fine-tuning:

```bash
python experiments/vit_intrinsic_lora.py \
  --mode full \
  --dataset cifar100 \
  --model_name vit_base_patch16_224 \
  --epochs 20 \
  --batch_size 128 \
  --lr 5e-5 \
  --weight_decay 0.05 \
  --optimizer adamw
```

LoRA:

```bash
python experiments/vit_intrinsic_lora.py \
  --mode lora \
  --dataset cifar100 \
  --model_name vit_base_patch16_224 \
  --lora_scope qkvo \
  --lora_rank 8 \
  --lora_alpha 16 \
  --epochs 15 \
  --batch_size 128 \
  --lr 3e-4 \
  --weight_decay 0.0 \
  --optimizer adamw
```

Module-matched intrinsic dimension:

```bash
python experiments/vit_intrinsic_lora.py \
  --mode id_module \
  --dataset cifar100 \
  --model_name vit_base_patch16_224 \
  --id_scope qkvo \
  --subspace_dim 666724 \
  --projection fastfood \
  --epochs 15 \
  --batch_size 128 \
  --lr 1e-2 \
  --weight_decay 0.0 \
  --optimizer adamw
```

## Main Components

### `experiments/vit_intrinsic_lora.py`

This is the main single-run entrypoint. It supports:

- `full`: full fine-tuning baseline
- `linear`: linear probe baseline
- `lora`: LoRA fine-tuning
- `id_module`: intrinsic-dimension tuning on a selected parameter subset
- `id_full`: intrinsic-dimension tuning over the full parameter space

It handles:

- timm model creation and optional checkpoint loading
- dataset loading for CIFAR-100 and Flowers-102
- optimizer and scheduler setup
- AMP support
- gradient accumulation
- per-run JSON result saving
- checkpoint save/resume behavior

### `src/models/lora.py`

This file implements the LoRA layer logic used by the ViT experiments.

Important pieces:

- `LoRALinear`: wraps a standard `nn.Linear`
- `LoRAQKVLinear`: targets timm ViT `qkv` projection while allowing selective q/k/v adaptation
- `apply_lora_to_timm_vit(...)`: injects LoRA modules into the attention stack
- trainable-parameter counting helpers for fair LoRA vs ID comparison

### `src/models/subspace.py`

This file provides the `SubspaceModel` wrapper used for intrinsic-dimension training.

Core idea:

```text
theta = theta_0 + P @ d
```

where:

- `theta_0` is the frozen base parameter vector
- `d` is the trainable low-dimensional vector
- `P` is a random projection

Important implementation detail:

- PyTorch 2.x uses `torch.func.functional_call` when available
- older versions fall back to parameter swapping while preserving gradients
- `trainable_param_names` allows module-matched projection instead of always projecting every parameter

### `src/projections/`

Three projection backends are included:

- `dense`: straightforward random projection
- `sparse`: lower-memory sparse projection
- `fastfood`: structured `O(D log D)` projection for large parameter spaces

For large ViT experiments, Fastfood is typically the practical choice.

## Practical Training Features

The repository is more than a minimal research prototype. It includes several quality-of-life and cost-control features:

- resume-safe run orchestration
- separate result JSONs per configuration
- periodic checkpoint saving
- automatic hyperparameter tuning script
- dry-run support before expensive experiments
- RTX 4070 8GB preset via `--gpu_profile rtx4070_8g`
- gradient accumulation through `--grad_accum_steps`
- optional gradient checkpointing when supported by timm models

### RTX 4070 8GB Preset

```bash
python scripts/run_vit_intrinsic_plan.py \
  --phase all \
  --dataset_main cifar100 \
  --dataset_aux flowers102 \
  --model_name vit_base_patch16_224 \
  --gpu_profile rtx4070_8g \
  --resume \
  --checkpoint_every 1
```

This preset lowers batch sizes and increases gradient accumulation so the project is more usable on consumer GPUs.

## Data And Outputs

Datasets are stored under `./data` by default.

Common output locations:

- run results: `results/vit_intrinsic/...`
- tuning summaries: `results/vit_intrinsic_tuning/...`
- auto-run manifests: `results/auto_runs/...`
- intermediate checkpoints: `*.ckpt.pt`

These runtime artifacts are intentionally ignored by Git in this repository.

## Offline Checkpoint Usage

If automatic pretrained download is blocked, you can provide a local timm-compatible checkpoint:

```bash
python scripts/run_vit_all_in_one.py \
  --dataset_main cifar100 \
  --dataset_aux cifar100 \
  --model_name vit_base_patch16_224 \
  --no_pretrained \
  --checkpoint_path /path/to/local_vit_checkpoint.pth \
  --resume
```

## Documentation

Detailed execution notes are available in:

- `docs/vit_id_lora_execution_plan.md`

## License

This project is released under the MIT License. See `LICENSE` for details.
