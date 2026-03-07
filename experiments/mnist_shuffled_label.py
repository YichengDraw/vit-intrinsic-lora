"""
MNIST Shuffled Label Experiment (Section 3.4)

Train models on MNIST with randomly shuffled labels to study memorization.

Paper results:
- 100% shuffled labels: d_int90 ≈ 190,000 (pure memorization)
- 50% shuffled labels: d_int90 ≈ 130,000
- 10% shuffled labels: d_int90 ≈ 90,000
- 0% shuffled (original): d_int90 ≈ 750

Key insight: Memorizing random labels requires much higher intrinsic dimension
than learning generalizable patterns.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SubspaceModel
from src.models.fc import fc_shuffled_label  # FC-5 network
from src.utils.data import get_mnist_shuffled_label_loaders
from src.utils.training import train_subspace_model
from src.utils.metrics import find_dint90


def train_subspace(
    subspace_dim: int,
    shuffle_fraction: float,
    device: torch.device,
    projection_type: str = 'fastfood',  # Use fastfood for large models
    epochs: int = 50,
    lr: float = 0.01,
) -> tuple:
    """Train FC-5 in subspace on shuffled labels."""
    train_loader, test_loader = get_mnist_shuffled_label_loaders(
        shuffle_fraction=shuffle_fraction
    )
    
    base_model = fc_shuffled_label(width=200)
    model = SubspaceModel(base_model, subspace_dim, projection_type=projection_type)
    
    results = train_subspace_model(
        model, train_loader, test_loader, device,
        epochs=epochs, lr=lr, verbose=False
    )
    
    # Return both train and test accuracy
    return results['final_train_acc'], results['best_test_acc']


def run_shuffled_label_experiment(
    device: torch.device,
    shuffle_fractions: list = None,
    dimensions: list = None,
    epochs: int = 50,
    save_dir: str = 'results'
):
    """
    Run shuffled label experiment for various shuffle fractions.
    
    This experiment requires significant compute for high shuffle fractions,
    as it needs to test very high subspace dimensions.
    """
    print("=" * 60)
    print("MNIST Shuffled Label Experiment")
    print("=" * 60)
    print("\nWARNING: This experiment is computationally expensive!")
    print("Full reproduction requires testing dimensions up to ~200,000")
    
    if shuffle_fractions is None:
        shuffle_fractions = [0.0, 0.1, 0.5, 1.0]
    
    if dimensions is None:
        # For full experiment, need much higher dimensions for high shuffle fractions
        # Paper tests up to 200k for 100% shuffle
        # We use a reduced set for practical demonstration
        dimensions = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    
    results_all = {}
    
    for frac in shuffle_fractions:
        print(f"\n{'=' * 40}")
        print(f"Shuffle Fraction: {frac * 100:.0f}%")
        print(f"{'=' * 40}")
        
        train_accs = []
        test_accs = []
        
        for d in dimensions:
            print(f"\nTraining with d = {d}...")
            train_acc, test_acc = train_subspace(
                d, frac, device, epochs=epochs
            )
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print(f"  d={d}: train={train_acc:.2f}%, test={test_acc:.2f}%")
        
        results_all[frac] = {
            'dimensions': dimensions,
            'train_accs': train_accs,
            'test_accs': test_accs
        }
    
    # Plot results
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['green', 'blue', 'orange', 'red']
    
    # Train accuracy (memorization)
    ax1 = axes[0]
    for frac, color in zip(shuffle_fractions, colors):
        r = results_all[frac]
        ax1.semilogx(r['dimensions'], r['train_accs'], 'o-', color=color,
                     markersize=8, linewidth=2, label=f'{frac*100:.0f}% shuffled')
    ax1.set_xlabel('Subspace Dimension (d)', fontsize=12)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax1.set_title('Training Accuracy (Memorization)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Test accuracy (generalization)
    ax2 = axes[1]
    for frac, color in zip(shuffle_fractions, colors):
        r = results_all[frac]
        ax2.semilogx(r['dimensions'], r['test_accs'], 'o-', color=color,
                     markersize=8, linewidth=2, label=f'{frac*100:.0f}% shuffled')
    ax2.set_xlabel('Subspace Dimension (d)', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy (Generalization)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mnist_shuffled_label.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_dir}/mnist_shuffled_label.png")
    plt.show()
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"Shuffled Label Results Summary")
    print(f"=" * 60)
    print("\nPaper findings:")
    print("  - 0% shuffle: d_int90 ≈ 750 (learning patterns)")
    print("  - 10% shuffle: d_int90 ≈ 90,000")
    print("  - 50% shuffle: d_int90 ≈ 130,000")
    print("  - 100% shuffle: d_int90 ≈ 190,000 (pure memorization)")
    print("\nConclusion: Memorization requires much higher intrinsic dimension")
    
    return results_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Shuffled Label Experiment')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--quick', action='store_true', help='Use smaller dimension range for quick test')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.quick:
        dimensions = [500, 1000, 2000, 5000]
    else:
        dimensions = None  # Use full range
    
    run_shuffled_label_experiment(
        device=device, 
        epochs=args.epochs, 
        save_dir=args.save_dir,
        dimensions=dimensions
    )
