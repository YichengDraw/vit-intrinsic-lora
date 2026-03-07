"""
MNIST FC Architecture Sweep (Figure 3)

Measure d_int90 across 20 FC architectures:
widths = [50, 100, 200, 400], depths = [1..5]
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.fc import fc_mnist
from src.models import SubspaceModel
from src.utils.data import get_mnist_loaders
from src.utils.training import train_subspace_model, train_epoch, evaluate
from src.utils.metrics import measure_dint90_with_bootstrap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_baseline_accuracy(
    width: int,
    depth: int,
    device: torch.device,
    epochs: int = 20,
    seed: int = 42
) -> float:
    """Train full FC model to get baseline accuracy (single seed)."""
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders()
    model = fc_mnist(width=width, depth=depth)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)
        best_acc = max(best_acc, test_acc)

    return best_acc


def train_subspace(
    width: int,
    depth: int,
    subspace_dim: int,
    device: torch.device,
    projection_type: str = "dense",
    epochs: int = 30,
    lr: float = 0.01,
    seed: int = 42
) -> float:
    """Train FC model in subspace and return best test accuracy."""
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders()
    base_model = fc_mnist(width=width, depth=depth)
    model = SubspaceModel(base_model, subspace_dim, projection_type=projection_type, seed=seed)

    results = train_subspace_model(
        model, train_loader, test_loader, device,
        epochs=epochs, lr=lr, verbose=False
    )
    return results["best_test_acc"]


def _load_existing_results(path: str) -> Dict:
    if not os.path.exists(path):
        return {"configs": []}
    with open(path, "r") as f:
        return json.load(f)


def _save_results(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_mnist_fc_variants(
    device: torch.device,
    widths: List[int],
    depths: List[int],
    dimensions: List[int],
    epochs: int = 30,
    n_seeds: int = 3,
    n_bootstrap: int = 500,
    save_dir: str = "results",
    force: bool = False
) -> Dict:
    """
    Run MNIST FC architecture sweep and compute d_int90 with bootstrap.
    """
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, "mnist_fc_variants.json")
    results = _load_existing_results(result_path)
    existing = {(c["width"], c["depth"]): c for c in results.get("configs", [])}

    for width in widths:
        for depth in depths:
            key = (width, depth)
            if key in existing and not force:
                print(f"Skipping width={width}, depth={depth} (cached)")
                continue

            base_model = fc_mnist(width=width, depth=depth)
            total_params = base_model.count_parameters()
            dims = [d for d in dimensions if d <= total_params]
            if not dims:
                dims = [min(50, total_params)]

            print(f"\nConfig width={width}, depth={depth} (D={total_params:,})")

            seeds = [42 + i for i in range(n_seeds)]
            baseline_accs = [
                get_baseline_accuracy(width, depth, device, epochs=20, seed=s)
                for s in seeds
            ]

            def train_fn(dim: int, seed_val: int) -> float:
                return train_subspace(
                    width, depth, dim, device,
                    epochs=epochs, lr=0.01, seed=seed_val
                )

            stats = measure_dint90_with_bootstrap(
                train_fn,
                dims,
                baseline_accs,
                n_seeds=n_seeds,
                n_bootstrap=n_bootstrap,
                seed=42
            )

            config_result = {
                "width": width,
                "depth": depth,
                "total_params": total_params,
                "dimensions": stats["dimensions"],
                "accuracies_mean": stats["accuracies_mean"],
                "accuracies_std": stats["accuracies_std"],
                "baseline_mean": stats["baseline_mean"],
                "baseline_std": stats["baseline_std"],
                "threshold": stats["threshold"],
                "d_int90_mean": stats["d_int90_mean"],
                "d_int90_std": stats["d_int90_std"],
                "n_seeds": stats["n_seeds"],
                "n_bootstrap": stats["n_bootstrap"],
            }

            existing[key] = config_result
            results["configs"] = list(existing.values())
            _save_results(result_path, results)

    # Plot
    configs = results.get("configs", [])
    if not configs:
        print("No results to plot.")
        return results

    xs = [c["total_params"] for c in configs]
    ys = [c["d_int90_mean"] for c in configs]
    yerrs = [c["d_int90_std"] for c in configs]
    sizes = [c["width"] * 2 for c in configs]
    depths_vals = [c["depth"] for c in configs]

    plt.figure(figsize=(10, 6))
    for x, y, yerr in zip(xs, ys, yerrs):
        plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="gray", alpha=0.6)
    sc = plt.scatter(xs, ys, s=sizes, c=depths_vals, cmap="viridis", alpha=0.8, edgecolors="black")
    cbar = plt.colorbar(sc)
    cbar.set_label("Depth")

    plt.xscale("log")
    plt.xlabel("Number of parameters D", fontsize=12)
    plt.ylabel("Intrinsic dimension $d_{int90}$", fontsize=12)
    plt.title("MNIST FC Variants: $d_{int90}$ vs Parameters", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "mnist_fc_variants.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST FC Architecture Sweep")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_bootstrap", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--force", action="store_true", help="Re-run even if cached")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    widths = [50, 100, 200, 400]
    depths = [1, 2, 3, 4, 5]
    dimensions = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]

    run_mnist_fc_variants(
        device=device,
        widths=widths,
        depths=depths,
        dimensions=dimensions,
        epochs=args.epochs,
        n_seeds=args.n_seeds,
        n_bootstrap=args.n_bootstrap,
        save_dir=args.save_dir,
        force=args.force
    )
