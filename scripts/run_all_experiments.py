"""
Master script to run all experiments from the Intrinsic Dimension paper.

Usage:
    python scripts/run_all_experiments.py --experiments all
    python scripts/run_all_experiments.py --experiments toy,mnist
    python scripts/run_all_experiments.py --experiments cifar --quick
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_experiment(name: str, quick: bool = False):
    """Run a single experiment by name."""
    print("\n" + "=" * 70)
    print(f"Running experiment: {name}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    try:
        if name == 'toy':
            from experiments.toy_problem import run_toy_experiment
            run_toy_experiment()
        
        elif name == 'mnist_fc':
            from experiments.mnist_fc import run_mnist_fc_experiment
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dims = [100, 300, 500, 750, 1000] if quick else None
            epochs = 15 if quick else 30
            run_mnist_fc_experiment(device, dimensions=dims, epochs=epochs)
        
        elif name == 'mnist_lenet':
            from experiments.mnist_lenet import run_mnist_lenet_experiment
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dims = [100, 200, 275, 400, 500] if quick else None
            epochs = 15 if quick else 30
            run_mnist_lenet_experiment(device, dimensions=dims, epochs=epochs)
        
        elif name == 'mnist_shuffled_pixel':
            from experiments.mnist_shuffled_pixel import run_shuffled_pixel_experiment
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            epochs = 15 if quick else 30
            run_shuffled_pixel_experiment(device, epochs=epochs)
        
        elif name == 'mnist_shuffled_label':
            from experiments.mnist_shuffled_label import run_shuffled_label_experiment
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dims = [500, 1000, 2000, 5000] if quick else None
            epochs = 25 if quick else 50
            run_shuffled_label_experiment(device, dimensions=dims, epochs=epochs)
        
        elif name == 'cifar':
            from experiments.cifar_resnet import run_cifar_experiment
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            models = ['lenet'] if quick else None  # Full: ['fc', 'lenet', 'resnet']
            epochs = 50 if quick else 100
            run_cifar_experiment(device, model_types=models, epochs=epochs)
        
        elif name == 'rl':
            from experiments.rl_experiments import run_all_rl_experiments
            run_all_rl_experiments()
        
        elif name == 'imagenet':
            from experiments.imagenet_squeezenet import run_imagenet_experiment
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            run_imagenet_experiment(device)
        
        elif name == 'regularization':
            from experiments.ablation_regularization import run_combined_analysis
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            run_combined_analysis(device)
        
        elif name == 'convnet_variants':
            from experiments.ablation_convnet_variants import run_convnet_variants_experiment
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            epochs = 15 if quick else 30
            run_convnet_variants_experiment(device, dataset='mnist', epochs=epochs)
        
        else:
            print(f"Unknown experiment: {name}")
            return False
        
        elapsed = time.time() - start_time
        print(f"\n[OK] Experiment '{name}' completed in {elapsed:.1f}s")
        return True
    
    except Exception as e:
        print(f"\n[FAIL] Experiment '{name}' failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run Intrinsic Dimension experiments')
    parser.add_argument('--experiments', type=str, default='toy',
                       help='Comma-separated list of experiments or "all"')
    parser.add_argument('--quick', action='store_true',
                       help='Use reduced settings for quick testing')
    args = parser.parse_args()
    
    all_experiments = [
        'toy',
        'mnist_fc',
        'mnist_lenet',
        'mnist_shuffled_pixel',
        'mnist_shuffled_label',
        'cifar',
        'rl',
        'regularization',
        'convnet_variants',
        # 'imagenet',  # Excluded by default (requires ImageNet dataset)
    ]
    
    if args.experiments == 'all':
        experiments = all_experiments
    else:
        experiments = [e.strip() for e in args.experiments.split(',')]
    
    print("=" * 70)
    print("Intrinsic Dimension Paper - Full Reproduction")
    print("Li et al., ICLR 2018")
    print("=" * 70)
    print(f"\nExperiments to run: {experiments}")
    print(f"Quick mode: {args.quick}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run experiments
    results = {}
    total_start = time.time()
    
    for exp in experiments:
        results[exp] = run_experiment(exp, quick=args.quick)
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 70)
    print("Experiment Summary")
    print("=" * 70)
    
    for exp, success in results.items():
        status = "[OK] Success" if success else "[FAIL] Failed"
        print(f"  {exp:25s} {status}")
    
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"\nResults saved to: results/")


if __name__ == '__main__':
    main()

