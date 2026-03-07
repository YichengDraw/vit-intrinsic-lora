"""
Metrics for measuring intrinsic dimension.

The key metric is d_int90: the smallest subspace dimension d 
at which the model achieves 90% of its baseline performance.

This provides a measure of the "effective dimensionality" of the
objective landscape for a given problem.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
import json


@dataclass
class IntrinsicDimResult:
    """Result of intrinsic dimension measurement."""
    d_int90: int
    baseline_acc: float
    threshold_acc: float
    dimensions_tested: List[int]
    accuracies: List[float]
    
    def to_dict(self) -> Dict:
        return {
            'd_int90': self.d_int90,
            'baseline_acc': self.baseline_acc,
            'threshold_acc': self.threshold_acc,
            'dimensions_tested': self.dimensions_tested,
            'accuracies': self.accuracies
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'IntrinsicDimResult':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def find_dint90(
    dimensions: List[int],
    accuracies: List[float],
    baseline_acc: float,
    threshold_ratio: float = 0.90
) -> int:
    """
    Find d_int90 from a list of (dimension, accuracy) pairs.
    
    Args:
        dimensions: List of subspace dimensions tested
        accuracies: Corresponding test accuracies
        baseline_acc: Baseline accuracy (full model)
        threshold_ratio: Fraction of baseline to achieve (default 0.90)
        
    Returns:
        Smallest dimension achieving threshold_ratio * baseline_acc
    """
    threshold = threshold_ratio * baseline_acc
    
    # Sort by dimension
    sorted_pairs = sorted(zip(dimensions, accuracies), key=lambda x: x[0])
    
    for dim, acc in sorted_pairs:
        if acc >= threshold:
            return dim
    
    # If no dimension achieved threshold, return the largest tested
    return max(dimensions)


def measure_intrinsic_dimension(
    train_fn: Callable[[int], float],
    baseline_acc: float,
    dimensions: Optional[List[int]] = None,
    threshold_ratio: float = 0.90,
    use_binary_search: bool = False,
    verbose: bool = True
) -> IntrinsicDimResult:
    """
    Measure intrinsic dimension by training at various subspace dimensions.
    
    Args:
        train_fn: Function that takes subspace_dim and returns test accuracy
        baseline_acc: Baseline accuracy (training in full parameter space)
        dimensions: List of dimensions to test (if not using binary search)
        threshold_ratio: Fraction of baseline for d_int threshold
        use_binary_search: Use binary search for efficiency
        verbose: Print progress
        
    Returns:
        IntrinsicDimResult with d_int90 and all measurements
    """
    threshold = threshold_ratio * baseline_acc
    
    if dimensions is None:
        # Default dimension schedule (logarithmic)
        dimensions = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    tested_dims = []
    accuracies = []
    
    if use_binary_search:
        # Binary search for d_int90
        low, high = 0, len(dimensions) - 1
        
        while low < high:
            mid = (low + high) // 2
            dim = dimensions[mid]
            
            if verbose:
                print(f"Testing d={dim}...")
            
            acc = train_fn(dim)
            tested_dims.append(dim)
            accuracies.append(acc)
            
            if verbose:
                print(f"  d={dim}: {acc:.2f}% (threshold: {threshold:.2f}%)")
            
            if acc >= threshold:
                high = mid
            else:
                low = mid + 1
        
        # Ensure we have the final value
        final_dim = dimensions[low]
        if final_dim not in tested_dims:
            acc = train_fn(final_dim)
            tested_dims.append(final_dim)
            accuracies.append(acc)
    
    else:
        # Test all dimensions
        for dim in dimensions:
            if verbose:
                print(f"Testing d={dim}...")
            
            acc = train_fn(dim)
            tested_dims.append(dim)
            accuracies.append(acc)
            
            if verbose:
                print(f"  d={dim}: {acc:.2f}% (threshold: {threshold:.2f}%)")
            
            # Early stop if we've found d_int90
            if acc >= threshold:
                if verbose:
                    print(f"  Found d_int90 = {dim}")
                break
    
    # Calculate d_int90
    d_int90 = find_dint90(tested_dims, accuracies, baseline_acc, threshold_ratio)
    
    return IntrinsicDimResult(
        d_int90=d_int90,
        baseline_acc=baseline_acc,
        threshold_acc=threshold,
        dimensions_tested=tested_dims,
        accuracies=accuracies
    )


def interpolate_dint(
    dimensions: List[int],
    accuracies: List[float],
    baseline_acc: float,
    threshold_ratio: float = 0.90
) -> float:
    """
    Interpolate to find more precise d_int90.
    
    Uses linear interpolation between measured points.
    """
    threshold = threshold_ratio * baseline_acc
    
    # Sort by dimension
    sorted_pairs = sorted(zip(dimensions, accuracies), key=lambda x: x[0])
    dims, accs = zip(*sorted_pairs)
    
    # Find crossing point
    for i in range(len(dims) - 1):
        if accs[i] < threshold <= accs[i + 1]:
            # Linear interpolation
            t = (threshold - accs[i]) / (accs[i + 1] - accs[i])
            d_int = dims[i] + t * (dims[i + 1] - dims[i])
            return d_int
    
    # If threshold not crossed, return boundary
    if accs[0] >= threshold:
        return dims[0]
    return dims[-1]


def estimate_compression_ratio(d_int90: int, total_params: int) -> float:
    """
    Estimate achievable compression ratio from intrinsic dimension.
    
    Paper shows networks can be compressed by factor of D/d_int90.
    """
    return total_params / d_int90


def measure_dint90_with_bootstrap(
    train_fn: Callable[[int, int], float],
    dimensions: List[int],
    baseline_acc,
    n_seeds: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 42,
    threshold_ratio: float = 0.90,
    use_interpolation: bool = True
) -> Dict:
    """
    Measure d_int90 with multiple seeds and bootstrap CI.
    
    Args:
        train_fn: Function (dim, seed) -> test accuracy
        dimensions: List of subspace dimensions to test
        baseline_acc: Baseline accuracy (float) or list of per-seed accuracies
        n_seeds: Number of random seeds (ignored if baseline_acc is list)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for bootstrap sampling
        threshold_ratio: Fraction of baseline for d_int90 (default 0.90)
        use_interpolation: Whether to interpolate d_int90 between points
    
    Returns:
        Dict with mean/std accuracies and bootstrap d_int90 stats.
    """
    rng = np.random.RandomState(seed)
    
    # Handle baseline accuracies (single value or per-seed list)
    if isinstance(baseline_acc, (list, tuple, np.ndarray)):
        baseline_accs = np.array(baseline_acc, dtype=float)
        n_seeds = int(baseline_accs.shape[0])
    else:
        baseline_accs = np.full(n_seeds, float(baseline_acc), dtype=float)
    
    seeds = [seed + i for i in range(n_seeds)]
    
    # Collect accuracies for each dimension/seed
    acc_matrix = np.zeros((len(dimensions), n_seeds), dtype=float)
    for i, dim in enumerate(dimensions):
        for j, s in enumerate(seeds):
            acc_matrix[i, j] = train_fn(dim, s)
    
    acc_mean = acc_matrix.mean(axis=1).tolist()
    acc_std = acc_matrix.std(axis=1).tolist()
    
    baseline_mean = float(baseline_accs.mean())
    baseline_std = float(baseline_accs.std())
    threshold = threshold_ratio * baseline_mean
    
    if use_interpolation:
        d_int90_mean = float(interpolate_dint(dimensions, acc_mean, baseline_mean, threshold_ratio))
    else:
        d_int90_mean = float(find_dint90(dimensions, acc_mean, baseline_mean, threshold_ratio))
    
    # Bootstrap d_int90
    d_samples = []
    if n_bootstrap > 0:
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n_seeds, size=n_seeds)
            sample_baseline = float(baseline_accs[idx].mean())
            sample_acc = acc_matrix[:, idx].mean(axis=1).tolist()
            if use_interpolation:
                d = interpolate_dint(dimensions, sample_acc, sample_baseline, threshold_ratio)
            else:
                d = find_dint90(dimensions, sample_acc, sample_baseline, threshold_ratio)
            d_samples.append(float(d))
    
    d_int90_std = float(np.std(d_samples)) if d_samples else 0.0
    
    return {
        'dimensions': list(dimensions),
        'accuracies_mean': acc_mean,
        'accuracies_std': acc_std,
        'acc_matrix': acc_matrix.tolist(),
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'threshold': float(threshold),
        'd_int90_mean': d_int90_mean,
        'd_int90_std': d_int90_std,
        'n_seeds': n_seeds,
        'n_bootstrap': n_bootstrap,
        'seeds': seeds
    }
