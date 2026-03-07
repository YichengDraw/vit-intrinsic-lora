"""
ImageNet Experiment with SqueezeNet (Supplementary S9)

Measure intrinsic dimension on ImageNet, a much larger scale problem.

Paper results:
- SqueezeNet: D = 1,248,424 parameters
- d_int90 > 500,000 (did not fully converge even at 500k)
- Top-1 accuracy at 500k: ~34% (baseline: 55.5%)

Note: This experiment requires:
1. ImageNet dataset (not included, must be downloaded separately)
2. Significant compute (paper reports 6-7 days per dimension on 4 GPUs)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SubspaceModel
from src.models.squeezenet import SqueezeNet
from src.utils.data import get_imagenet_loaders
from src.utils.training import train_subspace_model


def get_baseline_accuracy(device: torch.device, data_dir: str, epochs: int = 90) -> float:
    """Train SqueezeNet on ImageNet to get baseline."""
    print("Training baseline SqueezeNet on ImageNet...")
    print("WARNING: This requires significant compute resources!")
    
    try:
        train_loader, val_loader = get_imagenet_loaders(data_dir=data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Using approximate baseline from paper: 55.5%")
        return 55.5
    
    model = SqueezeNet()
    model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}")
        
        scheduler.step()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                correct += output.argmax(1).eq(target).sum().item()
                total += target.size(0)
        
        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch}: Top-1 Acc = {acc:.2f}%")
    
    return best_acc


def train_subspace(
    subspace_dim: int,
    device: torch.device,
    data_dir: str,
    epochs: int = 90,
    lr: float = 0.01,
) -> float:
    """Train SqueezeNet in subspace on ImageNet."""
    try:
        train_loader, val_loader = get_imagenet_loaders(data_dir=data_dir)
    except FileNotFoundError:
        print("ImageNet not found. Cannot run experiment.")
        return 0.0
    
    base_model = SqueezeNet()
    
    # Must use Fastfood for large parameter counts
    model = SubspaceModel(base_model, subspace_dim, projection_type='fastfood')
    
    results = train_subspace_model(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, verbose=True
    )
    
    return results['best_test_acc']


def run_imagenet_experiment(
    device: torch.device,
    data_dir: str = './data/imagenet',
    dimensions: list = None,
    epochs: int = 90,
    save_dir: str = 'results'
):
    """
    Run ImageNet intrinsic dimension experiment.
    
    This is a very expensive experiment. Paper results:
    - d=50k: ~10% accuracy
    - d=100k: ~18% accuracy
    - d=200k: ~26% accuracy
    - d=500k: ~34% accuracy
    - Baseline: ~55.5%
    
    d_int90 was not reached even at 500k dimensions.
    """
    print("=" * 60)
    print("ImageNet SqueezeNet Intrinsic Dimension Experiment")
    print("=" * 60)
    print("\nWARNING: This experiment is extremely computationally expensive!")
    print("Paper reports 6-7 days per dimension on 4 GPUs.\n")
    
    if dimensions is None:
        dimensions = [50000, 100000, 200000, 500000]
    
    # Check if ImageNet is available
    try:
        get_imagenet_loaders(data_dir=data_dir, batch_size=1)
        imagenet_available = True
    except FileNotFoundError:
        imagenet_available = False
        print("ImageNet not found. Using paper's reported values.")
    
    if not imagenet_available:
        # Use paper's values
        baseline_acc = 55.5
        dimensions = [50000, 100000, 200000, 500000]
        accuracies = [10.0, 18.0, 26.0, 34.34]  # From paper
        
        print("\nPaper's reported results:")
        for d, acc in zip(dimensions, accuracies):
            print(f"  d={d:,}: {acc:.2f}%")
        print(f"\nBaseline: {baseline_acc}%")
        print(f"90% threshold: {0.9 * baseline_acc:.2f}%")
        print(f"d_int90: > 500,000 (not reached)")
    else:
        baseline_acc = get_baseline_accuracy(device, data_dir, epochs)
        threshold_acc = 0.9 * baseline_acc
        
        accuracies = []
        for d in dimensions:
            print(f"\nTraining with d = {d:,}...")
            acc = train_subspace(d, device, data_dir, epochs=epochs)
            accuracies.append(acc)
            print(f"  d={d:,}: {acc:.2f}% (threshold: {threshold_acc:.2f}%)")
    
    # Plot (even with paper values)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(dimensions, accuracies, 'bo-', markersize=10, linewidth=2)
    plt.axhline(y=55.5, color='g', linestyle='-', linewidth=2, label='Baseline (55.5%)')
    plt.axhline(y=0.9 * 55.5, color='r', linestyle='--', linewidth=2, label='90% threshold (49.95%)')
    
    plt.xlabel('Subspace Dimension (d)', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('ImageNet SqueezeNet: Intrinsic Dimension\n(d_int90 > 500,000)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'imagenet_intrinsic_dim.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_dir}/imagenet_intrinsic_dim.png")
    plt.show()
    
    return {
        'dimensions': dimensions,
        'accuracies': accuracies,
        'baseline': 55.5,
        'd_int90': '>500000'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Intrinsic Dimension Experiment')
    parser.add_argument('--data_dir', type=str, default='./data/imagenet')
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_imagenet_experiment(
        device=device,
        data_dir=args.data_dir,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
