"""
Toy Problem Experiment (Section 2 of the paper)

Demonstrates the concept of intrinsic dimension with a simple optimization problem:
- Vector θ ∈ R^1000
- Constraint: sum of elements in each of 10 groups of 100 must equal 1, 2, ..., 10
- Solution manifold is 990-dimensional (1000 - 10 constraints)
- Intrinsic dimension is 10

Expected results:
- d < 10: Cannot achieve zero loss
- d >= 10: Can achieve (near) zero loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SubspaceModel


class VectorModel(nn.Module):
    """Simple model that just holds a parameter vector."""
    
    def __init__(self, size: int = 1000):
        super().__init__()
        self.param = nn.Parameter(torch.randn(size) * 0.01)
    
    def forward(self, x=None):
        return self.param


def toy_loss(theta: torch.Tensor, num_groups: int = 10) -> torch.Tensor:
    """
    Compute the toy problem loss.
    
    Loss = sum_i (sum(group_i) - target_i)^2
    where target_i = i + 1 for i in 0..9
    """
    D = theta.shape[0]
    group_size = D // num_groups
    
    loss = torch.tensor(0.0, device=theta.device)
    for i in range(num_groups):
        group = theta[i * group_size : (i + 1) * group_size]
        target = float(i + 1)
        loss = loss + (group.sum() - target) ** 2
    
    return loss


def train_toy_subspace(
    subspace_dim: int,
    full_dim: int = 1000,
    num_groups: int = 10,
    steps: int = 5000,
    lr: float = 0.01,
    projection_type: str = 'dense',
    seed: int = 42
) -> dict:
    """
    Train the toy problem in a random subspace.
    
    Args:
        subspace_dim: Dimension of the subspace to train in
        full_dim: Full parameter dimension (D)
        num_groups: Number of constraint groups
        steps: Number of optimization steps
        lr: Learning rate
        projection_type: Type of projection ('dense', 'sparse', 'fastfood')
        seed: Random seed
        
    Returns:
        Dictionary with final loss and loss history
    """
    torch.manual_seed(seed)
    
    # Create base model
    base_model = VectorModel(full_dim)
    
    # Wrap in subspace
    model = SubspaceModel(base_model, subspace_dim, projection_type=projection_type, seed=seed)
    
    # Optimizer (only optimize theta)
    optimizer = optim.Adam([model.theta], lr=lr)
    
    losses = []
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Get full parameter vector
        theta_full = model(None)
        
        # Compute loss
        loss = toy_loss(theta_full, num_groups)
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 500 == 0:
            print(f"  d={subspace_dim}, step={step}, loss={loss.item():.6f}")
    
    return {
        'final_loss': losses[-1],
        'losses': losses,
        'subspace_dim': subspace_dim
    }


def run_toy_experiment(
    full_dim: int = 1000,
    num_groups: int = 10,
    save_path: str = None
):
    """
    Run the full toy experiment.
    
    Tests various subspace dimensions to find the intrinsic dimension.
    Expected: d_int = num_groups (10 by default)
    """
    print(f"=" * 60)
    print(f"Toy Problem Experiment")
    print(f"Full dimension D = {full_dim}")
    print(f"Number of constraints = {num_groups}")
    print(f"Expected intrinsic dimension = {num_groups}")
    print(f"=" * 60)
    
    # Test dimensions around the expected intrinsic dimension
    dimensions = [1, 2,3,4, 5,6,7, 8, 9, 10, 11, 12,13,14, 15, 20,30, 50]
    
    results = []
    for d in dimensions:
        print(f"\nTraining with subspace dimension d = {d}...")
        result = train_toy_subspace(d, full_dim=full_dim, num_groups=num_groups)
        results.append(result)
        print(f"  Final loss: {result['final_loss']:.6f}")
    
    # Extract data for plotting
    dims = [r['subspace_dim'] for r in results]
    final_losses = [r['final_loss'] for r in results]
    
    # Find intrinsic dimension (first d where loss is very small)
    threshold = 1e-4
    d_int = None
    for d, loss in zip(dims, final_losses):
        if loss < threshold:
            d_int = d
            break
    
    # Compute performance for summary
    perfs = [1.0 / (1.0 + loss) for loss in final_losses]
    
    print(f"\n" + "=" * 60)
    print(f"Results Summary")
    print(f"=" * 60)
    print(f"{'Dimension':>10} | {'Final Loss':>15} | {'Performance':>12}")
    print("-" * 45)
    for d, loss, perf in zip(dims, final_losses, perfs):
        marker = " <-- d_int" if d == d_int else ""
        print(f"{d:>10} | {loss:>15.6f} | {perf:>11.4f}{marker}")
    
    print(f"\nMeasured intrinsic dimension: {d_int}")
    print(f"Expected intrinsic dimension: {num_groups}")
    print(f"\nPerformance = 1 / (1 + loss)  [maps loss=0 → perf=1]")
    
    # Convert loss to performance (matching paper's Figure 1 style)
    # Performance = 1 / (1 + loss) maps: loss=0 → perf=1, loss→∞ → perf→0
    performances = [np.exp(-loss) for loss in final_losses]
    
    # Find d_int90 (first d where performance >= 0.9)
    d_int90 = None
    for d, perf in zip(dims, performances):
        if perf >= 0.9:
            d_int90 = d
            break
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Performance vs subspace dimension (matching paper's style)
    ax1 = axes[0]
    ax1.plot(dims, performances, 'o-', color='#6495ED', markersize=8, linewidth=2)
    ax1.axhline(y=0.9, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add annotation for d_int90 = d_int100 = 10
    if d_int90 is not None:
        ax1.annotate(f'$d_{{int90}} = d_{{int100}} = {d_int90}$', 
                    xy=(d_int90 + 2, 0.92), fontsize=11, style='italic')
    
    ax1.set_xlabel('Subspace dim $d$', fontsize=12)
    ax1.set_ylabel('Performance', fontsize=12)
    ax1.set_title('Toy Problem: Intrinsic Dimension', fontsize=14)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(-2, max(dims) + 5)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves for selected dimensions (keep as-is for understanding optimization)
    ax2 = axes[1]
    selected_dims = [5,8, 9, 10, 11,12,15, 20]
    for result in results:
        if result['subspace_dim'] in selected_dims:
            ax2.semilogy(result['losses'], label=f"d = {result['subspace_dim']}")
    ax2.set_xlabel('Optimization Step', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Optimization Trajectories', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'results/toy_experiment.png'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    
    plt.show()
    
    return {
        'dimensions': dims,
        'final_losses': final_losses,
        'd_int': d_int,
        'expected_d_int': num_groups
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Toy Problem Experiment')
    parser.add_argument('--full_dim', type=int, default=1000, help='Full parameter dimension D')
    parser.add_argument('--num_groups', type=int, default=10, help='Number of constraint groups')
    parser.add_argument('--save_path', type=str, default='results/toy_experiment.png')
    args = parser.parse_args()
    
    run_toy_experiment(
        full_dim=args.full_dim,
        num_groups=args.num_groups,
        save_path=args.save_path
    )
