"""
Projection Comparison Experiment

Compare dense, sparse, and fastfood projections on MNIST FC.
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List

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


def get_baseline_accuracy(device: torch.device, epochs: int = 20, seed: int = 42) -> float:
    """Train full FC model for baseline accuracy."""
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders()
    model = fc_mnist(width=200, depth=2)
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
    subspace_dim: int,
    device: torch.device,
    projection_type: str,
    epochs: int = 30,
    lr: float = 0.01,
    seed: int = 42
) -> float:
    """Train FC model in a random subspace."""
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders()
    base_model = fc_mnist(width=200, depth=2)
    model = SubspaceModel(base_model, subspace_dim, projection_type=projection_type, seed=seed)

    results = train_subspace_model(
        model, train_loader, test_loader, device,
        epochs=epochs, lr=lr, verbose=False
    )
    return results["best_test_acc"]


def _load_cached(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_cached(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_projection_comparison(
    device: torch.device,
    dimensions: List[int],
    epochs: int = 30,
    n_seeds: int = 3,
    n_bootstrap: int = 300,
    save_dir: str = "results",
    force: bool = False
) -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, "projection_comparison.json")
    results = _load_cached(result_path) if not force else {}

    projections = ["dense", "sparse", "fastfood"]
    seeds = [42 + i for i in range(n_seeds)]
    baseline_accs = [get_baseline_accuracy(device, epochs=20, seed=s) for s in seeds]

    for proj in projections:
        if proj in results and not force:
            print(f"Skipping {proj} (cached)")
            continue

        def train_fn(dim: int, seed_val: int) -> float:
            return train_subspace(dim, device, proj, epochs=epochs, lr=0.01, seed=seed_val)

        stats = measure_dint90_with_bootstrap(
            train_fn,
            dimensions,
            baseline_accs,
            n_seeds=n_seeds,
            n_bootstrap=n_bootstrap,
            seed=42
        )

        results[proj] = stats
        _save_cached(result_path, results)

    # Plot
    plt.figure(figsize=(10, 6))
    for proj in projections:
        if proj not in results:
            continue
        stats = results[proj]
        dims = stats["dimensions"]
        accs = stats["accuracies_mean"]
        accs_std = stats["accuracies_std"]
        plt.errorbar(dims, accs, yerr=accs_std, marker="o", linewidth=2, label=proj)

    plt.xscale("log")
    plt.xlabel("Subspace dimension (d)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Projection Method Comparison (MNIST FC)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "projection_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projection Comparison")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_bootstrap", type=int, default=300)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dimensions = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
    run_projection_comparison(
        device=device,
        dimensions=dimensions,
        epochs=args.epochs,
        n_seeds=args.n_seeds,
        n_bootstrap=args.n_bootstrap,
        save_dir=args.save_dir,
        force=args.force
    )
