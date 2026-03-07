"""
CIFAR-10 Experiment with ResNet (Section 3.2)

Train ResNet-20 on CIFAR-10 in random subspaces to measure d_int90.

Paper results:
- ResNet on CIFAR-10: d_int90 between 20,000 and 50,000
- FC on CIFAR-10: d_int90 ≈ 9,000
- LeNet on CIFAR-10: d_int90 ≈ 2,900

Uses Fastfood projection for efficiency with large parameter counts.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SubspaceModel
from src.models.resnet import ResNet20
from src.models.lenet import lenet_cifar
from src.models.fc import fc_cifar
from src.utils.data import get_cifar10_loaders
from src.utils.training import train_subspace_model
from src.utils.metrics import find_dint90, IntrinsicDimResult


def get_baseline_accuracy(model_type: str, device: torch.device, epochs: int = 100) -> float:
    """Train full model to get baseline accuracy."""
    print(f"Training baseline {model_type} on CIFAR-10...")
    
    train_loader, test_loader = get_cifar10_loaders()
    
    if model_type == 'resnet':
        model = ResNet20()
    elif model_type == 'lenet':
        model = lenet_cifar()
    else:
        model = fc_cifar(width=200, depth=2)
    
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
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
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    correct += output.argmax(1).eq(target).sum().item()
            
            acc = 100.0 * correct / len(test_loader.dataset)
            best_acc = max(best_acc, acc)
            print(f"  Epoch {epoch}: {acc:.2f}%")
    
    print(f"Baseline accuracy: {best_acc:.2f}%")
    return best_acc


def train_subspace(
    model_type: str,
    subspace_dim: int,
    device: torch.device,
    projection_type: str = 'fastfood',
    epochs: int = 100,
    lr: float = 0.01,
) -> float:
    """Train model in subspace on CIFAR-10."""
    train_loader, test_loader = get_cifar10_loaders()
    
    if model_type == 'resnet':
        base_model = ResNet20()
    elif model_type == 'lenet':
        base_model = lenet_cifar()
    else:
        base_model = fc_cifar(width=200, depth=2)
    
    model = SubspaceModel(base_model, subspace_dim, projection_type=projection_type)
    
    results = train_subspace_model(
        model, train_loader, test_loader, device,
        epochs=epochs, lr=lr, verbose=False
    )
    
    return results['best_test_acc']


def run_cifar_experiment(
    device: torch.device,
    model_types: list = None,
    dimensions: dict = None,
    epochs: int = 100,
    save_dir: str = 'results'
):
    """
    Run CIFAR-10 intrinsic dimension experiment.
    """
    print("=" * 60)
    print("CIFAR-10 Intrinsic Dimension Experiment")
    print("=" * 60)
    
    if model_types is None:
        model_types = ['fc', 'lenet', 'resnet']
    
    if dimensions is None:
        dimensions = {
            'fc': [1000, 2000, 5000, 9000, 15000, 20000],
            'lenet': [500, 1000, 2000, 2900, 5000, 10000],
            'resnet': [5000, 10000, 20000, 35000, 50000, 75000]
        }
    
    results_all = {}
    
    for model_type in model_types:
        print(f"\n{'=' * 40}")
        print(f"Model: {model_type.upper()}")
        print(f"{'=' * 40}")
        
        baseline_acc = get_baseline_accuracy(model_type, device, epochs=epochs)
        threshold_acc = 0.9 * baseline_acc
        
        dims = dimensions[model_type]
        accuracies = []
        
        for d in dims:
            print(f"\nTraining {model_type} with d = {d}...")
            acc = train_subspace(model_type, d, device, epochs=epochs)
            accuracies.append(acc)
            print(f"  d={d}: {acc:.2f}% (threshold: {threshold_acc:.2f}%)")
        
        d_int90 = find_dint90(dims, accuracies, baseline_acc)
        
        results_all[model_type] = {
            'baseline': baseline_acc,
            'threshold': threshold_acc,
            'd_int90': d_int90,
            'dimensions': dims,
            'accuracies': accuracies
        }
        
        print(f"\n{model_type.upper()} d_int90: {d_int90}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"CIFAR-10 Results Summary")
    print(f"=" * 60)
    print(f"{'Model':>10} | {'d_int90':>10} | {'Expected':>10} | {'Baseline':>10}")
    print("-" * 50)
    expected = {'fc': '~9000', 'lenet': '~2900', 'resnet': '~35000'}
    for model_type in model_types:
        r = results_all[model_type]
        print(f"{model_type:>10} | {r['d_int90']:>10} | {expected[model_type]:>10} | {r['baseline']:>9.1f}%")
    
    # Plot
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    colors = {'fc': 'blue', 'lenet': 'orange', 'resnet': 'green'}
    
    for model_type in model_types:
        r = results_all[model_type]
        plt.semilogx(r['dimensions'], r['accuracies'], 'o-', color=colors[model_type],
                     markersize=8, linewidth=2, label=f"{model_type.upper()} (d_int90={r['d_int90']})")
        plt.axhline(y=r['threshold'], color=colors[model_type], linestyle='--', alpha=0.5)
    
    plt.xlabel('Subspace Dimension (d)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('CIFAR-10: Intrinsic Dimension by Architecture', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'cifar10_intrinsic_dim.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_dir}/cifar10_intrinsic_dim.png")
    plt.show()
    
    return results_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 Intrinsic Dimension Experiment')
    parser.add_argument('--model', type=str, default='all', choices=['all', 'fc', 'lenet', 'resnet'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model == 'all':
        model_types = ['fc', 'lenet', 'resnet']
    else:
        model_types = [args.model]
    
    run_cifar_experiment(
        device=device,
        model_types=model_types,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
