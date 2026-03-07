"""
ConvNet Architecture Variants Study (Supplementary S10)

Investigate the contribution of local receptive fields and weight-tying
to the intrinsic dimension of convolutional networks.

Four LeNet variants:
1. Standard LeNet: Local connections + tied weights (Conv2d)
2. UntiedLeNet: Local connections, untied weights
3. FCTiedLeNet: Global connections + tied weights  
4. FCLeNet: Global connections, untied weights (FC equivalent)

Paper results (MNIST):
- Standard LeNet: d_int90 ≈ 290
- Untied LeNet: d_int90 ≈ 600
- FCTied LeNet: d_int90 ≈ 425
- FC LeNet: d_int90 ≈ 2000

Conclusion: Both local connectivity AND weight-tying contribute to lower d_int90
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SubspaceModel
from src.models.lenet import LeNet, lenet_mnist, lenet_cifar
from src.models.lenet_variants import UntiedLeNet, FCTiedLeNet, FCLeNet
from src.utils.data import get_mnist_loaders, get_cifar10_loaders
from src.utils.training import train_subspace_model
from src.utils.metrics import find_dint90


def get_model(model_type: str, dataset: str = 'mnist'):
    """Create model by type and dataset."""
    if dataset == 'mnist':
        input_channels, input_size = 1, 28
    else:
        input_channels, input_size = 3, 32
    
    if model_type == 'lenet':
        return LeNet(input_channels=input_channels, input_size=input_size)
    elif model_type == 'untied':
        return UntiedLeNet(input_channels=input_channels, input_size=input_size)
    elif model_type == 'fctied':
        return FCTiedLeNet(input_channels=input_channels, input_size=input_size)
    elif model_type == 'fc':
        return FCLeNet(input_channels=input_channels, input_size=input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_baseline_accuracy(
    model_type: str, 
    dataset: str,
    device: torch.device, 
    epochs: int = 30
) -> float:
    """Train full model to get baseline accuracy."""
    print(f"Training baseline {model_type} on {dataset}...")
    
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders()
    else:
        train_loader, test_loader = get_cifar10_loaders()
    
    model = get_model(model_type, dataset)
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                correct += output.argmax(1).eq(target).sum().item()
        
        acc = 100.0 * correct / len(test_loader.dataset)
        best_acc = max(best_acc, acc)
    
    print(f"  Baseline accuracy: {best_acc:.2f}%")
    return best_acc


def train_subspace(
    model_type: str,
    dataset: str,
    subspace_dim: int,
    device: torch.device,
    epochs: int = 30,
) -> float:
    """Train model variant in subspace."""
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders()
    else:
        train_loader, test_loader = get_cifar10_loaders()
    
    base_model = get_model(model_type, dataset)
    
    # Use appropriate projection based on model size
    num_params = sum(p.numel() for p in base_model.parameters())
    projection = 'fastfood' if num_params > 100000 else 'dense'
    
    model = SubspaceModel(base_model, subspace_dim, projection_type=projection)
    
    results = train_subspace_model(
        model, train_loader, test_loader, device,
        epochs=epochs, lr=0.01, verbose=False
    )
    
    return results['best_test_acc']


def run_convnet_variants_experiment(
    device: torch.device,
    dataset: str = 'mnist',
    epochs: int = 30,
    save_dir: str = 'results'
):
    """
    Run ConvNet variants experiment.
    
    Compare intrinsic dimension across:
    1. Standard LeNet (local + tied)
    2. Untied LeNet (local + untied)
    3. FCTied LeNet (global + tied)
    4. FC LeNet (global + untied)
    """
    print("=" * 60)
    print(f"ConvNet Variants Experiment on {dataset.upper()}")
    print("=" * 60)
    
    model_types = ['lenet', 'untied', 'fctied', 'fc']
    
    # Different dimension ranges for different models
    if dataset == 'mnist':
        dimensions = {
            'lenet': [100, 200, 290, 400, 500, 750],
            'untied': [200, 400, 600, 800, 1000, 1500],
            'fctied': [150, 300, 425, 600, 800, 1000],
            'fc': [500, 1000, 1500, 2000, 3000, 5000]
        }
        expected_dint = {
            'lenet': 290,
            'untied': 600,
            'fctied': 425,
            'fc': 2000
        }
    else:  # CIFAR-10
        dimensions = {
            'lenet': [500, 1000, 1500, 2000, 3000],
            'untied': [1000, 2000, 2750, 4000, 5000],
            'fctied': [1000, 2000, 2500, 4000, 5000],
            'fc': [10000, 20000, 35000, 50000, 75000]
        }
        expected_dint = {
            'lenet': 1000,
            'untied': 2750,
            'fctied': 2500,
            'fc': 35000
        }
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'=' * 40}")
        print(f"Model: {model_type}")
        print(f"{'=' * 40}")
        
        baseline = get_baseline_accuracy(model_type, dataset, device, epochs)
        threshold = 0.9 * baseline
        
        dims = dimensions[model_type]
        accuracies = []
        
        for d in dims:
            print(f"\nTraining {model_type} with d = {d}...")
            acc = train_subspace(model_type, dataset, d, device, epochs)
            accuracies.append(acc)
            print(f"  d={d}: {acc:.2f}% (threshold: {threshold:.2f}%)")
        
        d_int90 = find_dint90(dims, accuracies, baseline)
        
        results[model_type] = {
            'baseline': baseline,
            'threshold': threshold,
            'd_int90': d_int90,
            'expected_dint': expected_dint[model_type],
            'dimensions': dims,
            'accuracies': accuracies
        }
        
        print(f"\n{model_type} d_int90: {d_int90} (expected: ~{expected_dint[model_type]})")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"ConvNet Variants Results ({dataset.upper()})")
    print("=" * 60)
    print(f"{'Model':>15} | {'d_int90':>10} | {'Expected':>10} | {'Local':>10} | {'Tied':>10}")
    print("-" * 65)
    
    properties = {
        'lenet': ('Yes', 'Yes'),
        'untied': ('Yes', 'No'),
        'fctied': ('No', 'Yes'),
        'fc': ('No', 'No')
    }
    
    for model_type in model_types:
        r = results[model_type]
        local, tied = properties[model_type]
        print(f"{model_type:>15} | {r['d_int90']:>10} | {r['expected_dint']:>10} | {local:>10} | {tied:>10}")
    
    print(f"\nConclusion: Both local connectivity AND weight-tying contribute to lower d_int90")
    
    # Plot
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    colors = {'lenet': 'blue', 'untied': 'orange', 'fctied': 'green', 'fc': 'red'}
    labels = {
        'lenet': 'LeNet (local+tied)',
        'untied': 'Untied (local only)',
        'fctied': 'FCTied (tied only)',
        'fc': 'FC (neither)'
    }
    
    for model_type in model_types:
        r = results[model_type]
        plt.semilogx(r['dimensions'], r['accuracies'], 'o-', 
                     color=colors[model_type], markersize=8, linewidth=2,
                     label=f"{labels[model_type]} (d_int90={r['d_int90']})")
    
    plt.xlabel('Subspace Dimension (d)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'ConvNet Variants: Local Connectivity vs Weight-Tying ({dataset.upper()})', fontsize=14)
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, f'convnet_variants_{dataset}.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_dir}/convnet_variants_{dataset}.png")
    plt.show()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ConvNet Variants Experiment')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_convnet_variants_experiment(
        device=device,
        dataset=args.dataset,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
