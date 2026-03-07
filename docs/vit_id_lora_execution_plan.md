# ViT + LoRA + Intrinsic Dimension Execution Plan

## 1. Final Decisions

- Main dataset: `CIFAR-100`
- Secondary validation dataset: `Flowers-102`
- Backbone: `ViT-Base/16` (`vit_base_patch16_224`)
- Pretraining for mainline: standard supervised ImageNet-pretrained ViT
- Comparison rule: match by **trainable parameter count**
- Runtime budget target: ~`30 GPU-hours` on `V100 32GB`
- Full-parameter ID: keep as optional coarse sweep (stop early if clearly infeasible)
- Example target environment: a cloud GPU machine with modern PyTorch

## 1.1 Resource Note (Cloud GPU Machines)

If the instance card shows `8vCPUs | 64GiB`, the `64GiB` is typically **host RAM**, not GPU VRAM.
Check real GPU memory inside the job:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

No code change is required just because host RAM is 64GiB.

## 2. What Was Implemented

### 2.1 Core capability changes

- Added partial-parameter ID projection in `SubspaceModel`
  - File: `src/models/subspace.py`
  - New argument: `trainable_param_names`
  - Supports both:
    - full-parameter ID (`trainable_param_names=None`)
    - module-matched ID (explicit parameter allowlist)

- Added LoRA implementation for timm ViT
  - File: `src/models/lora.py`
  - Includes:
    - `LoRALinear`
    - `LoRAQKVLinear` (q/k/v-selective for timm `attn.qkv`)
    - `apply_lora_to_timm_vit(...)`
    - parameter-count helpers for matched-ID design

- Added ViT transfer data loaders
  - File: `src/utils/data.py`
  - New loaders:
    - `get_cifar100_loaders(...)`
    - `get_flowers102_loaders(...)`
  - Added ImageNet-style normalization/224 preprocessing for ViT transfer

### 2.2 New experiment scripts

- Single-run experiment driver:
  - `experiments/vit_intrinsic_lora.py`
  - Modes:
    - `full` (full fine-tuning baseline)
    - `linear` (linear probe baseline)
    - `lora` (LoRA fine-tuning)
    - `id_module` (module-matched ID)
    - `id_full` (full-parameter ID)
  - Stores per-run JSON logs
  - Supports `--resume` skip behavior

- Unattended plan runner:
  - `scripts/run_vit_intrinsic_plan.py`
  - Launches baseline, LoRA-vs-ID sweeps, Flowers validation, full-ID coarse sweep
  - Computes module-matched ID dimensions from LoRA trainable params
  - Supports `--resume` for crash-safe continuation

### 2.3 Dependency updates

- Updated `requirements.txt`:
  - `timm>=0.6.7`

## 3. Recommended File/Folder Layout

All additions are in:

- `experiments/vit_intrinsic_lora.py`
- `scripts/run_vit_intrinsic_plan.py`
- `src/models/lora.py`
- `src/models/subspace.py` (extended)
- `src/utils/data.py` (extended)
- `docs/vit_id_lora_execution_plan.md` (this file)

## 4. Operating Procedure

## 4.0 What to Upload to Another Machine

Upload the whole folder:

- this project folder

This is sufficient because it already includes:

- `src/` core library code
- `experiments/` training entrypoints
- `scripts/` runner/tuner/test tools
- `requirements.txt`
- `docs/vit_id_lora_execution_plan.md`

You do not need external paper PDFs for execution.

## 4.1 Environment setup

```bash
cd vit_intrinsic_lora
# For Python 3.7 / torch 1.10 notebook environments:
pip install -r requirements.vit_py37.txt
```

Optional quick sanity:

```bash
python -c "import timm, torch; print(timm.__version__, torch.__version__)"
```

Verify GPU and CUDA visibility:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### 4.1.1 Offline pretrained checkpoint (avoid timm URL 403)

If cloud egress to Google Storage is blocked, run with local checkpoint and disable online pretrained download:

```bash
python scripts/run_vit_all_in_one.py \
  --dataset_main cifar100 \
  --dataset_aux cifar100 \
  --model_name vit_base_patch16_224 \
  --seeds 42 \
  --skip_smoke \
  --no_pretrained \
  --checkpoint_path /path/to/local_vit_checkpoint.pth \
  --resume \
  > logs/run_all_in_one.log 2>&1 &
```

Notes:

- `--no_pretrained` prevents timm from fetching remote `.npz`.
- `--checkpoint_path` loads your uploaded local ViT checkpoint.
- Keep `--dataset_aux cifar100` on torch/torchvision stacks without `Flowers102` support.

## 4.2 Single-run examples

Full fine-tuning baseline on CIFAR-100:

```bash
python experiments/vit_intrinsic_lora.py \
  --mode full \
  --dataset cifar100 \
  --epochs 20 \
  --batch_size 128 \
  --lr 5e-5 \
  --weight_decay 0.05 \
  --optimizer adamw \
  --resume
```

Linear probe baseline:

```bash
python experiments/vit_intrinsic_lora.py \
  --mode linear \
  --dataset cifar100 \
  --epochs 15 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 0.0 \
  --optimizer adamw \
  --resume
```

LoRA (QKVO, rank=8):

```bash
python experiments/vit_intrinsic_lora.py \
  --mode lora \
  --dataset cifar100 \
  --lora_scope qkvo \
  --lora_rank 8 \
  --lora_alpha 16 \
  --epochs 15 \
  --batch_size 128 \
  --lr 3e-4 \
  --weight_decay 0.0 \
  --optimizer adamw \
  --resume
```

Module-matched ID (example `d=666724`):

```bash
python experiments/vit_intrinsic_lora.py \
  --mode id_module \
  --dataset cifar100 \
  --id_scope qkvo \
  --subspace_dim 666724 \
  --projection fastfood \
  --epochs 15 \
  --batch_size 128 \
  --lr 1e-2 \
  --weight_decay 0.0 \
  --optimizer adamw \
  --resume
```

Full-parameter ID (coarse test point):

```bash
python experiments/vit_intrinsic_lora.py \
  --mode id_full \
  --dataset cifar100 \
  --subspace_dim 1000000 \
  --projection fastfood \
  --epochs 10 \
  --batch_size 128 \
  --lr 1e-2 \
  --weight_decay 0.0 \
  --optimizer adamw \
  --resume
```

## 4.3 Unattended full plan (recommended)

```bash
python scripts/run_vit_intrinsic_plan.py \
  --phase all \
  --dataset_main cifar100 \
  --dataset_aux flowers102 \
  --model_name vit_base_patch16_224 \
  --seeds 42,43,44 \
  --lora_ranks 2,4,8,16,32 \
  --full_id_dims 100000,300000,1000000,3000000 \
  --resume
```

This script runs one configuration per process, so interruptions waste less compute and recovery is simple.

Validate command groups before burning GPU:

```bash
python scripts/run_vit_intrinsic_plan.py \
  --phase all \
  --dry_run
```

Long-run reliability flags (recommended for overnight):

```bash
python scripts/run_vit_intrinsic_plan.py \
  --phase all \
  --resume \
  --checkpoint_every 1
```

## 4.4 Smoke test before cloud run

Use this to validate the whole code path without dataset downloads:

```bash
python scripts/test_vit_intrinsic_pipeline.py \
  --model_name vit_tiny_patch16_224 \
  --modes full,linear,lora,id_module,id_full \
  --epochs 1 \
  --num_batches 2
```

For a closer architecture check (slower), switch to:

```bash
python scripts/test_vit_intrinsic_pipeline.py \
  --model_name vit_base_patch16_224 \
  --modes lora,id_module,id_full \
  --epochs 1 \
  --num_batches 1
```

## 4.5 Automatic Hyperparameter Tuning (Recommended)

Use the built-in successive-halving tuner before full runs:

```bash
python scripts/tune_vit_hparams.py \
  --dataset cifar100 \
  --model_name vit_base_patch16_224 \
  --modes full,linear,lora,id_module \
  --stage_epochs 3,6,12 \
  --keep_ratio 0.5 \
  --max_train_batches 64 \
  --max_test_batches 32
```

Why this is cost-effective:

- Stage 1 quickly rejects weak configs with low batch/epoch budget
- Stage 2/3 only spend compute on top candidates
- Outputs one JSON summary with best config per mode

Safety checks before actual tuning:

```bash
python scripts/tune_vit_hparams.py --dry_run
```

Recommended real tuning command (cost-aware):

```bash
python scripts/tune_vit_hparams.py \
  --dataset cifar100 \
  --model_name vit_base_patch16_224 \
  --modes full,linear,lora,id_module \
  --stage_epochs 3,6,12 \
  --keep_ratio 0.5 \
  --max_train_batches 64 \
  --max_test_batches 32
```

Tuning output:

- `results/vit_intrinsic_tuning/tuning_summary__*.json`

Use best configs from that summary for the final 30h sweep.

## 4.6 Full Run (After Tuning)

Single-command full pipeline:

```bash
python scripts/run_vit_intrinsic_plan.py \
  --phase all \
  --resume \
  --checkpoint_every 1
```

If needed, print command groups first:

```bash
python scripts/run_vit_intrinsic_plan.py --phase all --dry_run
```

## 4.7 One-Click Unattended Run

Use the all-in-one driver to execute:

1. smoke test
2. automatic tuning
3. full plan run (with tuned hyperparameters injected)

```bash
python scripts/run_vit_all_in_one.py \
  --no_pretrained \
  --checkpoint_path /path/to/local_vit_checkpoint.pth \
  --resume \
  --checkpoint_every 1
```

Dry-run preview:

```bash
python scripts/run_vit_all_in_one.py --dry_run
```

All-in-one manifest is saved to:

- `results/auto_runs/run_YYYYMMDD_HHMMSS/manifest.json`

## 5. Budgeted 30h Schedule (Practical)

1. `0-4h`: CIFAR-100 baselines (`full`, `linear`), 2-3 seeds  
2. `4-14h`: module-matched LoRA-vs-ID sweeps, 1 seed full grid  
3. `14-20h`: add seeds near Pareto frontier / threshold crossing  
4. `20-24h`: Flowers-102 key-point validation (full + best LoRA + matched ID)  
5. `24-30h`: full-parameter ID coarse points (early-stop if trend is poor)

## 6. Early-stop Rules (to avoid wasted GPU)

- For full-parameter ID line:
  - Stop after 2-3 coarse points if accuracy is far below `0.9 * full-FT baseline`
  - Report as lower-bound style result, e.g. `d_int90 > largest_tested_d`

- For module-matched line:
  - Prioritize extra seeds only around best LoRA rank and closest ID dimensions

## 7. Does This Need Constant Manual Intervention?

Short answer: **No**, if you use the runner and `--resume`.

- Why it is sleep-friendly:
- Each configuration is a separate run
- Completed runs are persisted as JSON
- Relaunching with `--resume` skips finished items automatically
- In-progress runs save `*.ckpt.pt` every epoch (`--checkpoint_every`)
- If a run is interrupted mid-way, relaunching the same command with `--resume`
  continues from the last saved checkpoint

Checkpoint/result behavior:

- Completed run:
  - `.../<run_name>.json` is written (final metrics)
- In-progress run:
  - `.../<run_name>.ckpt.pt` stores epoch/history/optimizer/scheduler/scaler/model states
- Interrupted run:
  - Final JSON may not exist yet, but checkpoint is still usable for continuation

## 7.1 Single-GPU vs Multi-GPU Utilization

Current code path is **single-process, single-GPU per run**.

- One run does not automatically consume 8 GPUs.
- To utilize 8 GPUs efficiently, run multiple independent jobs in parallel (seed/rank sharding).

### Practical parallel strategy (recommended)

1. Run tuning on 1 GPU first (avoid multiplying bad configs).
2. Split final sweeps across GPUs by seed and/or rank groups.

Linux examples:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_vit_intrinsic_plan.py --phase baselines --seeds 42 --resume
CUDA_VISIBLE_DEVICES=1 python scripts/run_vit_intrinsic_plan.py --phase baselines --seeds 43 --resume
CUDA_VISIBLE_DEVICES=2 python scripts/run_vit_intrinsic_plan.py --phase baselines --seeds 44 --resume
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_vit_intrinsic_plan.py --phase module_sweep --seeds 42 --lora_ranks 2,4 --resume
CUDA_VISIBLE_DEVICES=1 python scripts/run_vit_intrinsic_plan.py --phase module_sweep --seeds 42 --lora_ranks 8,16 --resume
CUDA_VISIBLE_DEVICES=2 python scripts/run_vit_intrinsic_plan.py --phase module_sweep --seeds 42 --lora_ranks 32 --resume
```

This is usually better than one giant distributed job for this project, because experiments are naturally independent.

## 11. Remote Storage Workflow

Recommended practical flow:

1. Upload project to cloud workspace:
2. Run experiments on training instance.
3. Sync results back to your remote storage:
   - `results/vit_intrinsic/`
   - `results/vit_intrinsic_tuning/`
   - `results/auto_runs/`
4. Verify files in remote storage.
5. Recycle/release compute resources.

As long as the `results/` directories are synced before recycle, outputs are preserved.

- What still benefits from daytime check-ins:
  - After first few runs, verify learning rates are not clearly unstable
  - Confirm no dataset download/auth issues in cloud environment

So you can safely run overnight; you are not required to manually insert jobs mid-run.

## 8. Paper Alignment and Source Mapping

## 8.1 Intrinsic Dimension (ICLR 2018)

Paper: `Intrinsic Dimension 1804.08838v1.pdf`

- Core idea used exactly: train in random subspace and measure threshold dimension
  - Found text: page 1-2 mentions training in “smaller, randomly oriented subspace”
- Threshold metric in this project:
  - page 4 defines `d_int90` as intrinsic dimension at 90% baseline

How we map it:

- ID experiment modes (`id_module`, `id_full`) implement `theta = theta0 + P @ d`
- `d_int90` is computed against full fine-tuning baseline per dataset

## 8.2 ViT transfer setup

Paper: `ViT.pdf`

- page 1-2: ViT transferred to small/mid benchmarks including `CIFAR-100`, `VTAB`
- page 4: explicitly lists `Oxford Flowers-102` among transfer benchmarks
- page 5: describes fine-tuning setup

How we map it:

- Datasets selected: CIFAR-100 + Flowers-102
- Backbone: ViT-Base/16
- Transfer/fine-tuning protocol with ImageNet-normalized 224-pixel inputs

## 8.3 LoRA low-rank adaptation

Paper: `LoRA.pdf`

- page 1-2: LoRA freezes pretrained weights and inserts low-rank trainable factors
- page 2/4: adaptation updates are hypothesized low intrinsic rank

How we map it:

- `lora` mode injects low-rank adapters on ViT attention stack (QKVO scope)
- Uses trainable-param-matching against ID for fair comparison

## 8.4 DINOv3 / iBOT relation (context, not mainline baseline)

- `iBOT 2111.07832v3.pdf` page 6 mentions transfer results including `CIFAR100`, `Flowers`
- `DINO v3.pdf` contains Flowers evaluation tables

How we map it:

- Mainline control variable remains supervised ViT pretraining
- DINO/iBOT checkpoints can be added later as controlled ablations

## 9. What Is “Same as Original” vs “New”

Same-as-paper logic:

- Random-subspace optimization and `d_int90` measurement (ID paper)
- ViT transfer tasks on CIFAR-100 / Flowers style datasets (ViT lineage)
- LoRA low-rank adaptation mechanism (LoRA paper)

New contribution in this project:

- Direct head-to-head comparison: **module-matched ID vs LoRA** under equal trainable-parameter budget
- Added practical “full-ID coarse feasibility line” for modern ViT transfer

## 10. Optional Reductions if Time Is Tight

If runtime is tighter than expected, drop in this order:

1. Reduce full-ID coarse points first  
2. Reduce second dataset seeds (keep 1 seed key-point validation)  
3. Keep module-matched LoRA-vs-ID mainline intact

This preserves the most publishable conclusion with minimal sacrifice.


