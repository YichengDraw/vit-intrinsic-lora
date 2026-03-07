"""
Regularization Ablation Study (Supplementary S8)

Compare subspace training with traditional regularization methods:
1. L2 penalty (weight decay)
2. Dropout
3. Subspace training (implicit regularization)

Paper findings:
- Subspace training provides implicit regularization by restricting solution space
- L2 and Dropout interact with intrinsic dimension differently
- Subspace training can outperform traditional regularizers when d is properly chosen
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SubspaceModel
from src.models.fc import fc_mnist
from src.utils.data import get_mnist_loaders
from src.utils.training import train_subspace_model, train_epoch, evaluate


def train_with_regularization(
    device: torch.device,
    epochs: int = 30,
    weight_decay: float = 0.0,
    dropout: float = 0.0,
) -> dict:
    """Train FC model with traditional regularization."""
    train_loader, test_loader = get_mnist_loaders()
    model = fc_mnist(width=200, depth=2, dropout=dropout)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    train_accs = []
    test_accs = []
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    return {
        'train_acc': train_accs[-1],
        'test_acc': test_accs[-1],
        'best_test_acc': max(test_accs),
        'gap': train_accs[-1] - test_accs[-1]  # Generalization gap
    }


def train_subspace_with_reg(
    device: torch.device,
    subspace_dim: int,
    epochs: int = 30,
    weight_decay: float = 0.0,
    dropout: float = 0.0,
) -> dict:
    """Train FC in subspace with additional regularization."""
    train_loader, test_loader = get_mnist_loaders()
    base_model = fc_mnist(width=200, depth=2, dropout=dropout)
    model = SubspaceModel(base_model, subspace_dim)
    
    results = train_subspace_model(
        model, train_loader, test_loader, device,
        epochs=epochs, lr=0.01, weight_decay=weight_decay, verbose=False
    )
    
    return {
        'train_acc': results['final_train_acc'],
        'test_acc': results['final_test_acc'],
        'best_test_acc': results['best_test_acc'],
        'gap': results['final_train_acc'] - results['final_test_acc']
    }


def run_l2_comparison(device: torch.device, save_dir: str = 'results'):
    """Compare L2 penalty with subspace training."""
    print("=" * 60)
    print("L2 Penalty vs Subspace Training")
    print("=" * 60)
    
    # L2 penalties to test
    weight_decays = [0, 1e-5, 1e-4, 1e-3, 1e-2]
    
    # Subspace dimensions to test
    subspace_dims = [100, 200, 500, 1000, 2000, 5000]
    
    # Direct training with L2
    print("\nDirect training with L2 penalty:")
    l2_results = []
    for wd in weight_decays:
        print(f"  Weight decay = {wd}...")
        result = train_with_regularization(device, weight_decay=wd)
        l2_results.append(result)
        print(f"    Train: {result['train_acc']:.2f}%, Test: {result['test_acc']:.2f}%, Gap: {result['gap']:.2f}%")
    
    # Subspace training (no L2)
    print("\nSubspace training (no L2):")
    subspace_results = []
    for d in subspace_dims:
        print(f"  d = {d}...")
        result = train_subspace_with_reg(device, d)
        subspace_results.append(result)
        print(f"    Train: {result['train_acc']:.2f}%, Test: {result['test_acc']:.2f}%, Gap: {result['gap']:.2f}%")
    
    # Plot
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Test accuracy comparison
    ax1 = axes[0]
    ax1.plot(range(len(weight_decays)), [r['test_acc'] for r in l2_results], 
             'bo-', markersize=8, linewidth=2, label='L2 penalty')
    ax1.set_xticks(range(len(weight_decays)))
    ax1.set_xticklabels([str(wd) for wd in weight_decays])
    ax1.set_xlabel('Weight Decay', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Direct Training with L2 Penalty', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.semilogx(subspace_dims, [r['test_acc'] for r in subspace_results],
                  'go-', markersize=8, linewidth=2, label='Subspace')
    ax2.set_xlabel('Subspace Dimension', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Subspace Training (Implicit Regularization)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'regularization_l2.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_dir}/regularization_l2.png")
    plt.show()
    
    return {'l2': l2_results, 'subspace': subspace_results}


def run_dropout_comparison(device: torch.device, save_dir: str = 'results'):
    """Compare Dropout with subspace training."""
    print("=" * 60)
    print("Dropout vs Subspace Training")
    print("=" * 60)
    
    # Dropout rates to test
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Subspace dimensions
    subspace_dims = [100, 200, 500, 1000, 2000, 5000]
    
    # Direct training with Dropout
    print("\nDirect training with Dropout:")
    dropout_results = []
    for dr in dropout_rates:
        print(f"  Dropout = {dr}...")
        result = train_with_regularization(device, dropout=dr)
        dropout_results.append(result)
        print(f"    Train: {result['train_acc']:.2f}%, Test: {result['test_acc']:.2f}%, Gap: {result['gap']:.2f}%")
    
    # Subspace training (no dropout)
    print("\nSubspace training (no Dropout):")
    subspace_results = []
    for d in subspace_dims:
        print(f"  d = {d}...")
        result = train_subspace_with_reg(device, d, dropout=0.0)
        subspace_results.append(result)
        print(f"    Train: {result['train_acc']:.2f}%, Test: {result['test_acc']:.2f}%, Gap: {result['gap']:.2f}%")
    
    # Plot generalization gap comparison
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Generalization gap
    ax1 = axes[0]
    ax1.plot(dropout_rates, [r['gap'] for r in dropout_results], 
             'ro-', markersize=8, linewidth=2, label='Dropout')
    ax1.set_xlabel('Dropout Rate', fontsize=12)
    ax1.set_ylabel('Train-Test Gap (%)', fontsize=12)
    ax1.set_title('Generalization Gap with Dropout', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.semilogx(subspace_dims, [r['gap'] for r in subspace_results],
                  'go-', markersize=8, linewidth=2, label='Subspace')
    ax2.set_xlabel('Subspace Dimension', fontsize=12)
    ax2.set_ylabel('Train-Test Gap (%)', fontsize=12)
    ax2.set_title('Generalization Gap with Subspace Training', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'regularization_dropout.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_dir}/regularization_dropout.png")
    plt.show()
    
    return {'dropout': dropout_results, 'subspace': subspace_results}


def run_combined_analysis(device: torch.device, save_dir: str = 'results'):
    """Run full regularization comparison analysis."""
    print("=" * 60)
    print("Combined Regularization Analysis")
    print("=" * 60)
    
    l2_results = run_l2_comparison(device, save_dir)
    dropout_results = run_dropout_comparison(device, save_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nKey findings from the paper:")
    print("1. Subspace training provides implicit regularization")
    print("2. Lower d leads to smaller generalization gap")
    print("3. When d is properly chosen, subspace training can match or exceed")
    print("   traditional regularizers")
    print("4. L2 and Dropout interact with d_int90 - they change the objective landscape")
    
    return {'l2': l2_results, 'dropout': dropout_results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regularization Ablation Study')
    parser.add_argument('--experiment', type=str, default='all', 
                       choices=['all', 'l2', 'dropout'])
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.experiment == 'all':
        run_combined_analysis(device, args.save_dir)
    elif args.experiment == 'l2':
        run_l2_comparison(device, args.save_dir)
    else:
        run_dropout_comparison(device, args.save_dir)
