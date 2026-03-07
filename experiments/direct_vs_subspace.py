"""
Direct vs Subspace Training Comparison (Figure 4)

Gray points: direct training with small networks
Blue points: subspace training results
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.fc import fc_mnist
from src.utils.data import get_mnist_loaders
from src.utils.training import train_epoch, evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_direct_model(
    model: nn.Module,
    device: torch.device,
    epochs: int = 20,
    seed: int = 42
) -> float:
    """Train a model directly and return best test accuracy."""
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)
        best_acc = max(best_acc, test_acc)
    return best_acc


def load_subspace_results(path: str) -> Optional[Dict[str, List[float]]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)

    dims = data.get("dimensions_tested") or data.get("dimensions")
    accs = data.get("accuracies") or data.get("accuracies_mean")
    if dims is None or accs is None:
        return None
    return {"dims": dims, "accs": accs}


class ScaledLeNet(nn.Module):
    """LeNet-style model with scaled channel/FC widths."""

    def __init__(self, scale: float = 1.0, input_size: int = 28):
        super().__init__()
        c1 = max(1, int(round(6 * scale)))
        c2 = max(1, int(round(16 * scale)))
        f1 = max(1, int(round(120 * scale)))
        f2 = max(1, int(round(84 * scale)))

        self.conv1 = nn.Conv2d(1, c1, kernel_size=5)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

        conv_output_size = ((input_size - 4) // 2 - 4) // 2
        fc_input_dim = c2 * conv_output_size * conv_output_size

        self.fc1 = nn.Linear(fc_input_dim, f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _load_cached(path: str) -> Dict:
    if not os.path.exists(path):
        return {"models": []}
    with open(path, "r") as f:
        return json.load(f)


def _save_cached(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_direct_vs_subspace(
    device: torch.device,
    epochs: int = 20,
    save_dir: str = "results",
    force: bool = False,
    subspace_fc_path: str = "results/mnist_fc_results.json",
    subspace_lenet_path: str = "results/mnist_lenet_results.json"
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # FC direct models
    fc_cache_path = os.path.join(save_dir, "mnist_direct_fc.json")
    fc_cache = _load_cached(fc_cache_path)
    fc_existing = {m["width"]: m for m in fc_cache.get("models", [])}

    fc_widths = [10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400]
    fc_models = []
    for width in fc_widths:
        if width in fc_existing and not force:
            fc_models.append(fc_existing[width])
            continue
        model = fc_mnist(width=width, depth=2)
        acc = train_direct_model(model, device, epochs=epochs, seed=42)
        fc_models.append({
            "width": width,
            "depth": 2,
            "total_params": model.count_parameters(),
            "acc": acc
        })
        fc_existing[width] = fc_models[-1]
        fc_cache["models"] = list(fc_existing.values())
        _save_cached(fc_cache_path, fc_cache)

    # Conv direct models (scaled LeNet)
    conv_cache_path = os.path.join(save_dir, "mnist_direct_lenet.json")
    conv_cache = _load_cached(conv_cache_path)
    conv_existing = {m["scale"]: m for m in conv_cache.get("models", [])}

    scales = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    conv_models = []
    for scale in scales:
        if scale in conv_existing and not force:
            conv_models.append(conv_existing[scale])
            continue
        model = ScaledLeNet(scale=scale)
        acc = train_direct_model(model, device, epochs=epochs, seed=42)
        conv_models.append({
            "scale": scale,
            "total_params": model.count_parameters(),
            "acc": acc
        })
        conv_existing[scale] = conv_models[-1]
        conv_cache["models"] = list(conv_existing.values())
        _save_cached(conv_cache_path, conv_cache)

    # Load subspace results
    sub_fc = load_subspace_results(subspace_fc_path)
    sub_lenet = load_subspace_results(subspace_lenet_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # FC plot
    ax1 = axes[0]
    ax1.scatter(
        [m["total_params"] for m in fc_models],
        [m["acc"] for m in fc_models],
        c="gray", alpha=0.6, label="Direct"
    )
    if sub_fc is not None:
        ax1.scatter(sub_fc["dims"], sub_fc["accs"], c="#4C72B0", label="Subspace")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of trainable parameters")
    ax1.set_ylabel("Validation accuracy")
    ax1.set_title("FC Networks (MNIST)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Conv plot
    ax2 = axes[1]
    ax2.scatter(
        [m["total_params"] for m in conv_models],
        [m["acc"] for m in conv_models],
        c="gray", alpha=0.6, label="Direct"
    )
    if sub_lenet is not None:
        ax2.scatter(sub_lenet["dims"], sub_lenet["accs"], c="#4C72B0", label="Subspace")
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of trainable parameters")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_title("Conv Networks (MNIST)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "mnist_direct_vs_subspace.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct vs Subspace Comparison (MNIST)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--force", action="store_true", help="Re-run even if cached")
    parser.add_argument("--subspace_fc", type=str, default="results/mnist_fc_results.json")
    parser.add_argument("--subspace_lenet", type=str, default="results/mnist_lenet_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_direct_vs_subspace(
        device=device,
        epochs=args.epochs,
        save_dir=args.save_dir,
        force=args.force,
        subspace_fc_path=args.subspace_fc,
        subspace_lenet_path=args.subspace_lenet
    )
