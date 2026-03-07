"""
MNIST Experiment with Fully Connected Networks (Section 3.1)

Train FC networks on MNIST in random subspaces to measure d_int90.

Paper results:
- FC(W=200, L=2): D = 199,210 parameters, d_int90 ≈ 750
- FC networks of varying sizes have similar d_int90 for the same dataset
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import json
import random
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SubspaceModel
from src.models.fc import fc_mnist
from src.utils.data import get_mnist_loaders
from src.utils.training import train_subspace_model, train_epoch, evaluate
from src.utils.metrics import find_dint90, IntrinsicDimResult, measure_dint90_with_bootstrap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_baseline_accuracy(device: torch.device, epochs: int = 20, seed: int = 42) -> float:
    """Train full FC model to get baseline accuracy."""
    print("Training baseline (full parameter space)...")
    
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders()
    model = fc_mnist(width=200, depth=2)
    model.to(device)
    
    print(f"Baseline model parameters: {model.count_parameters():,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, device)
        _, acc = evaluate(model, test_loader, criterion, device)
        best_acc = max(best_acc, acc)
        print(f"  Epoch {epoch}: {acc:.2f}%")
    
    print(f"Baseline accuracy: {best_acc:.2f}%")
    return best_acc


def train_subspace(
    subspace_dim: int,
    device: torch.device,
    projection_type: str = 'dense',
    epochs: int = 30,
    lr: float = 0.01,
    seed: int = 42
) -> float:
    """Train FC model in subspace and return best test accuracy."""
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders()
    base_model = fc_mnist(width=200, depth=2)
    
    model = SubspaceModel(base_model, subspace_dim, projection_type=projection_type, seed=seed)
    
    results = train_subspace_model(
        model, train_loader, test_loader, device,
        epochs=epochs, lr=lr, verbose=False
    )
    
    return results['best_test_acc']


def run_mnist_fc_experiment(
    device: torch.device,
    dimensions: Optional[List[int]] = None,
    projection_type: str = 'dense',
    epochs: int = 30,
    save_dir: str = 'results',
    n_seeds: int = 1,
    n_bootstrap: int = 0,
    force: bool = False
):
    """
    Run full MNIST FC experiment.
    
    Tests various subspace dimensions to measure d_int90.
    """
    print("=" * 60)
    print("MNIST FC Intrinsic Dimension Experiment")
    print("=" * 60)
    
    if dimensions is None:
        # Paper tests these dimensions approximately
        dimensions = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
    dims: List[int] = list(dimensions)
    
    # Cache handling
    os.makedirs(save_dir, exist_ok=True)
    if n_seeds <= 1:
        result_path = os.path.join(save_dir, 'mnist_fc_results.json')
    else:
        result_path = os.path.join(save_dir, 'mnist_fc_results_multiseed.json')
    
    acc_std: Optional[List[float]] = None
    accuracies: List[float] = []
    d_int90: float = 0.0
    baseline_accs: List[float] = []
    cached = os.path.exists(result_path) and not force
    if cached:
        print(f"Found cached results at {result_path}, skipping training.")
        with open(result_path, 'r') as f:
            cached = json.load(f)
        baseline_acc = cached.get('baseline_acc', cached.get('baseline_mean', 0.0))
        threshold_acc = cached.get('threshold_acc', cached.get('threshold', 0.0))
        dims = cached.get('dimensions_tested') or cached.get('dimensions') or dims
        accuracies = cached.get('accuracies', cached.get('accuracies_mean', [])) or []
        acc_std = cached.get('accuracies_std')
        d_int90 = float(cached.get('d_int90', cached.get('d_int90_mean', 0.0)))
    else:
        # Get baseline
        seeds = [42 + i for i in range(n_seeds)]
        baseline_accs = [get_baseline_accuracy(device, seed=s) for s in seeds]
        baseline_acc = float(np.mean(baseline_accs))
        threshold_acc = 0.9 * baseline_acc
    
    print(f"\nThreshold for d_int90: {threshold_acc:.2f}%")
    print(f"\nTesting subspace dimensions: {dims}")
    
    if not cached:
        if n_seeds <= 1:
            accuracies = []
            for d in dims:
                print(f"\nTraining with d = {d}...")
                acc = train_subspace(d, device, projection_type=projection_type, epochs=epochs)
                accuracies.append(acc)
                print(f"  d={d}: {acc:.2f}% (threshold: {threshold_acc:.2f}%)")
                
                if acc >= threshold_acc:
                    print(f"  [OK] Achieved threshold!")
            
            # Find d_int90
            d_int90 = find_dint90(dims, accuracies, baseline_acc)
            result_obj = IntrinsicDimResult(
                d_int90=int(d_int90),
                baseline_acc=baseline_acc,
                threshold_acc=threshold_acc,
                dimensions_tested=dims,
                accuracies=accuracies
            )
            result_obj.save(result_path)
        else:
            def train_fn(dim: int, seed_val: int) -> float:
                return train_subspace(
                    dim, device, projection_type=projection_type, epochs=epochs, seed=seed_val
                )
            
            stats = measure_dint90_with_bootstrap(
                train_fn,
                dims,
                baseline_accs,
                n_seeds=n_seeds,
                n_bootstrap=n_bootstrap or 300,
                seed=42
            )
            accuracies = stats['accuracies_mean']
            acc_std = stats['accuracies_std']
            d_int90 = stats['d_int90_mean']
            payload = {
                'd_int90_mean': d_int90,
                'd_int90_std': stats.get('d_int90_std', 0.0),
                'baseline_mean': baseline_acc,
                'threshold': threshold_acc,
                'dimensions': dims,
                'accuracies_mean': accuracies,
                'accuracies_std': acc_std,
                'n_seeds': n_seeds,
                'n_bootstrap': n_bootstrap
            }
            with open(result_path, 'w') as f:
                json.dump(payload, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print(f"Results Summary")
    print(f"=" * 60)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    print(f"Threshold (90%): {threshold_acc:.2f}%")
    print(f"Measured d_int90: {d_int90}")
    print(f"Expected d_int90: ~750 (from paper)")
    
    # Results saved during training or loaded from cache
    
    # Plot
    plt.figure(figsize=(10, 6))
    if n_seeds <= 1:
        plt.plot(dims, accuracies, 'bo-', markersize=8, linewidth=2, label='Subspace training')
    else:
        yerr = acc_std
        plt.errorbar(dims, accuracies, yerr=yerr, fmt='o-', color='b', markersize=8, linewidth=2,
                     label='Subspace training')
    plt.axhline(y=baseline_acc, color='g', linestyle='-', linewidth=2, label=f'Baseline ({baseline_acc:.1f}%)')
    plt.axhline(y=threshold_acc, color='r', linestyle='--', linewidth=2, label=f'90% threshold ({threshold_acc:.1f}%)')
    plt.axvline(x=d_int90, color='orange', linestyle=':', linewidth=2, label=f'd_int90 = {d_int90}')
    
    plt.xlabel('Subspace Dimension (d)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('MNIST FC: Intrinsic Dimension Measurement', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.savefig(os.path.join(save_dir, 'mnist_fc_intrinsic_dim.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_dir}/mnist_fc_intrinsic_dim.png")
    plt.show()
    
    result = {
        'baseline_acc': baseline_acc,
        'threshold_acc': threshold_acc,
        'dimensions': dims,
        'accuracies': accuracies,
        'accuracies_std': acc_std,
        'd_int90': d_int90
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST FC Intrinsic Dimension Experiment')
    parser.add_argument('--projection', type=str, default='dense', choices=['dense', 'sparse', 'fastfood'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--n_seeds', type=int, default=1)
    parser.add_argument('--n_bootstrap', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_mnist_fc_experiment(
        device=device,
        projection_type=args.projection,
        epochs=args.epochs,
        save_dir=args.save_dir,
        n_seeds=args.n_seeds,
        n_bootstrap=args.n_bootstrap,
        force=args.force
    )

