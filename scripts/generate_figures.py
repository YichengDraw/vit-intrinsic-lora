"""
Generate publication-quality figures summarizing all experiments.

Creates:
1. Summary table of d_int90 across all datasets/models
2. Comparison plots matching paper figures
3. Combined visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Paper's reported d_int90 values for comparison
PAPER_RESULTS = {
    'MNIST FC': {'D': 199210, 'd_int90': 750},
    'MNIST LeNet': {'D': 44426, 'd_int90': 275},
    'MNIST-SP FC': {'D': 199210, 'd_int90': 750},
    'MNIST-SP LeNet': {'D': 44426, 'd_int90': 650},
    'MNIST-SL 100%': {'D': 959610, 'd_int90': 190000},
    'MNIST-SL 50%': {'D': 959610, 'd_int90': 130000},
    'MNIST-SL 10%': {'D': 959610, 'd_int90': 90000},
    'CIFAR FC': {'D': 1055610, 'd_int90': 9000},
    'CIFAR LeNet': {'D': 62006, 'd_int90': 2900},
    'ImageNet SqueezeNet': {'D': 1248424, 'd_int90': '>500000'},
    'CartPole': {'D': 199210, 'd_int90': 25},
    'Inverted Pendulum': {'D': 562, 'd_int90': 4},
    'Humanoid': {'D': 166673, 'd_int90': 700},
    'Atari Pong': {'D': 1005974, 'd_int90': 6000},
}


def create_summary_table(save_path: str = 'results/summary_table.png'):
    """Create summary table of all d_int90 values."""
    print("Creating summary table...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Table data
    columns = ['Dataset/Task', 'Model', 'Parameters (D)', 'd_int90', 'Compression']
    rows = []
    
    for name, data in PAPER_RESULTS.items():
        D = data['D']
        d_int = data['d_int90']
        
        if isinstance(d_int, int):
            compression = f"{D / d_int:.0f}x"
        else:
            compression = "<2.5x"
        
        # Parse name to get dataset and model
        parts = name.split()
        if len(parts) >= 2:
            dataset = parts[0]
            model = ' '.join(parts[1:])
        else:
            dataset = name
            model = '-'
        
        rows.append([dataset, model, f"{D:,}", str(d_int), compression])
    
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.2, 0.2, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')
    
    plt.title('Intrinsic Dimension Summary (d_int90)\nLi et al., ICLR 2018', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to {save_path}")
    plt.close()


def create_comparison_plot(save_path: str = 'results/dint90_comparison.png'):
    """Create bar chart comparing d_int90 across tasks."""
    print("Creating comparison plot...")
    
    # Select subset for visualization
    tasks = [
        'Inv. Pendulum', 'CartPole', 'MNIST LeNet', 'MNIST FC',
        'CIFAR LeNet', 'Pong', 'CIFAR FC', 'Humanoid'
    ]
    
    d_int_values = [4, 25, 275, 750, 2900, 6000, 9000, 700]
    
    # Sort by d_int90
    sorted_pairs = sorted(zip(tasks, d_int_values), key=lambda x: x[1])
    tasks, d_int_values = zip(*sorted_pairs)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(tasks)))
    bars = ax.barh(range(len(tasks)), d_int_values, color=colors)
    
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=11)
    ax.set_xlabel('Intrinsic Dimension (d_int90)', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('Comparison of Intrinsic Dimensions Across Tasks', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, d_int_values)):
        ax.text(val * 1.1, i, f'{val:,}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def create_problem_difficulty_plot(save_path: str = 'results/problem_difficulty.png'):
    """
    Create the paper's main insight figure:
    d_int90 as a measure of problem difficulty.
    """
    print("Creating problem difficulty plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Data points from paper
    problems = {
        'Inverted Pendulum': (562, 4),
        'CartPole': (199210, 25),
        'MNIST LeNet': (44426, 275),
        'MNIST FC': (199210, 750),
        'CIFAR LeNet': (62006, 2900),
        'Atari Pong': (1005974, 6000),
        'CIFAR FC': (1055610, 9000),
    }
    
    # Categorize by task type
    rl_tasks = ['Inverted Pendulum', 'CartPole', 'Atari Pong']
    mnist_tasks = ['MNIST LeNet', 'MNIST FC']
    cifar_tasks = ['CIFAR LeNet', 'CIFAR FC']
    
    for name, (D, d_int) in problems.items():
        if name in rl_tasks:
            color, marker = 'green', 's'
        elif name in mnist_tasks:
            color, marker = 'blue', 'o'
        else:
            color, marker = 'red', '^'
        
        ax.scatter(D, d_int, s=200, c=color, marker=marker, edgecolors='black', linewidth=1.5)
        ax.annotate(name, (D, d_int), textcoords="offset points", 
                   xytext=(10, 5), fontsize=10)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Total Parameters (D)', fontsize=12)
    ax.set_ylabel('Intrinsic Dimension (d_int90)', fontsize=12)
    ax.set_title('Intrinsic Dimension as a Measure of Problem Difficulty', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=10, label='RL Tasks'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='MNIST'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
               markersize=10, label='CIFAR-10'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def create_all_figures(save_dir: str = 'results'):
    """Generate all summary figures."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Summary Figures")
    print("=" * 60)
    
    create_summary_table(os.path.join(save_dir, 'summary_table.png'))
    create_comparison_plot(os.path.join(save_dir, 'dint90_comparison.png'))
    create_problem_difficulty_plot(os.path.join(save_dir, 'problem_difficulty.png'))
    
    print("\n[OK] All figures generated successfully!")
    print(f"Check the '{save_dir}' directory for outputs.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate summary figures')
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()
    
    create_all_figures(args.save_dir)

