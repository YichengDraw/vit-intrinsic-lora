"""
MNIST Shuffled Pixel Experiment (Section 3.3)

Train models on MNIST with randomly permuted pixels.
This destroys spatial structure, so LeNet loses its advantage over FC.

Paper results:
- FC: d_int90 ≈ 750 (same as original MNIST)
- LeNet: d_int90 ≈ 650 (higher than original 275)

Key insight: Without spatial structure, convolutions provide less benefit.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SubspaceModel
from src.models.fc import fc_mnist
from src.models.lenet import lenet_mnist
from src.utils.data import get_mnist_shuffled_pixel_loaders
from src.utils.training import train_subspace_model
from src.utils.metrics import find_dint90, IntrinsicDimResult


def get_baseline_accuracy(model_type: str, device: torch.device, epochs: int = 20) -> float:
    """Train full model on shuffled pixels to get baseline."""
    print(f"Training baseline {model_type} on shuffled pixels...")
    
    train_loader, test_loader = get_mnist_shuffled_pixel_loaders()
    
    if model_type == 'fc':
        model = fc_mnist(width=200, depth=2)
    else:
        model = lenet_mnist()
    
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
        print(f"  Epoch {epoch}: {acc:.2f}%")
    
    return best_acc


def train_subspace(
    model_type: str,
    subspace_dim: int,
    device: torch.device,
    projection_type: str = 'dense',
    epochs: int = 30,
    lr: float = 0.01,
) -> float:
    """Train model in subspace on shuffled pixels."""
    train_loader, test_loader = get_mnist_shuffled_pixel_loaders()
    
    if model_type == 'fc':
        base_model = fc_mnist(width=200, depth=2)
    else:
        base_model = lenet_mnist()
    
    model = SubspaceModel(base_model, subspace_dim, projection_type=projection_type)
    
    results = train_subspace_model(
        model, train_loader, test_loader, device,
        epochs=epochs, lr=lr, verbose=False
    )
    
    return results['best_test_acc']


def run_shuffled_pixel_experiment(
    device: torch.device,
    dimensions: list = None,
    epochs: int = 30,
    save_dir: str = 'results'
):
    """
    Run shuffled pixel experiment for both FC and LeNet.
    """
    print("=" * 60)
    print("MNIST Shuffled Pixel Experiment")
    print("=" * 60)
    
    if dimensions is None:
        dimensions = [100, 200, 300, 500, 650, 750, 1000, 1500, 2000]
    
    results_all = {}
    
    for model_type in ['fc', 'lenet']:
        print(f"\n{'=' * 40}")
        print(f"Model: {model_type.upper()}")
        print(f"{'=' * 40}")
        
        baseline_acc = get_baseline_accuracy(model_type, device)
        threshold_acc = 0.9 * baseline_acc
        
        accuracies = []
        for d in dimensions:
            print(f"\nTraining {model_type} with d = {d}...")
            acc = train_subspace(model_type, d, device, epochs=epochs)
            accuracies.append(acc)
            print(f"  d={d}: {acc:.2f}%")
        
        d_int90 = find_dint90(dimensions, accuracies, baseline_acc)
        
        results_all[model_type] = {
            'baseline': baseline_acc,
            'threshold': threshold_acc,
            'd_int90': d_int90,
            'dimensions': dimensions,
            'accuracies': accuracies
        }
        
        print(f"\n{model_type.upper()} d_int90: {d_int90}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"Shuffled Pixel Results Summary")
    print(f"=" * 60)
    print(f"{'Model':>10} | {'d_int90':>10} | {'Expected':>10}")
    print("-" * 35)
    print(f"{'FC':>10} | {results_all['fc']['d_int90']:>10} | {'~750':>10}")
    print(f"{'LeNet':>10} | {results_all['lenet']['d_int90']:>10} | {'~650':>10}")
    print(f"\nNote: LeNet d_int90 increases from ~275 to ~650 when spatial structure is destroyed")
    
    # Plot comparison
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for model_type, color in [('fc', 'blue'), ('lenet', 'orange')]:
        r = results_all[model_type]
        plt.plot(r['dimensions'], r['accuracies'], 'o-', color=color, 
                markersize=8, linewidth=2, label=f"{model_type.upper()} (d_int90={r['d_int90']})")
        plt.axhline(y=r['threshold'], color=color, linestyle='--', alpha=0.5)
    
    plt.xlabel('Subspace Dimension (d)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('MNIST Shuffled Pixels: FC vs LeNet', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.savefig(os.path.join(save_dir, 'mnist_shuffled_pixel.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_dir}/mnist_shuffled_pixel.png")
    plt.show()
    
    return results_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Shuffled Pixel Experiment')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_shuffled_pixel_experiment(device=device, epochs=args.epochs, save_dir=args.save_dir)
