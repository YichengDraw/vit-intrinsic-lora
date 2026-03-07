"""
Reinforcement Learning Experiments (Section 3.5)

Measure intrinsic dimension of RL policy optimization using Evolution Strategies (ES).

Paper results:
- CartPole: d_int90 ≈ 25
- Inverted Pendulum: d_int90 ≈ 4
- Humanoid: d_int90 ≈ 700
- Atari Pong: d_int90 ≈ 6000

Key insight: ES works well for subspace training because it doesn't require
backpropagation through the environment.

Compatible with PyTorch 1.8.0+ and both gym/gymnasium.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SubspaceModel
from src.models.policy import (
    cartpole_policy, 
    pendulum_policy, 
    humanoid_policy, 
    atari_pong_policy,
    FCPolicy,
    ConvPolicy
)

# Try to import gymnasium first, fall back to gym
HAS_GYM = False
GYM_MODULE = None
USE_NEW_API = False  # True for gymnasium, False for old gym

try:
    import gymnasium as gym
    HAS_GYM = True
    GYM_MODULE = gym
    USE_NEW_API = True
    print("Using gymnasium (new API)")
except ImportError:
    try:
        import gym
        HAS_GYM = True
        GYM_MODULE = gym
        USE_NEW_API = False
        print("Using gym (old API)")
    except ImportError:
        print("Warning: Neither gymnasium nor gym installed. RL experiments will not run.")


def evaluate_policy(model, env_name: str, num_episodes: int = 5) -> float:
    """Evaluate policy on environment and return average reward."""
    if not HAS_GYM:
        raise ImportError("gymnasium or gym is required for RL experiments")
    
    env = GYM_MODULE.make(env_name)
    total_rewards = []
    
    for _ in range(num_episodes):
        # Reset - API differs between gym and gymnasium
        if USE_NEW_API:
            obs, _ = env.reset()
        else:
            obs = env.reset()
        
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                action_output = model(obs_tensor)
                
                # Handle discrete vs continuous
                if hasattr(GYM_MODULE.spaces, 'Discrete') and isinstance(env.action_space, GYM_MODULE.spaces.Discrete):
                    action = action_output.argmax().item()
                else:
                    action = action_output.numpy()
            
            # Step - API differs between gym and gymnasium
            if USE_NEW_API:
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                obs, reward, done, _ = env.step(action)
            
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    env.close()
    return np.mean(total_rewards)


def train_es(
    model: nn.Module,
    env_name: str,
    iterations: int = 100,
    population_size: int = 16,
    sigma: float = 0.1,
    lr: float = 0.05,
    verbose: bool = True
) -> dict:
    """
    Train policy using Evolution Strategies.
    
    ES is ideal for subspace training because:
    1. It doesn't require backprop through environment
    2. It naturally handles non-differentiable reward signals
    3. It can work with very low-dimensional subspaces
    """
    if not HAS_GYM:
        raise ImportError("gymnasium or gym is required for RL experiments")
    
    # Get theta (subspace parameter)
    if hasattr(model, 'theta'):
        theta = model.theta.data.clone()
    else:
        # Flatten all parameters
        theta = torch.cat([p.flatten() for p in model.parameters()])
    
    rewards_history = []
    best_reward = -float('inf')
    
    for iter in range(iterations):
        # Sample perturbations
        grad = torch.zeros_like(theta)
        batch_rewards = []
        
        for i in range(population_size):
            epsilon = torch.randn_like(theta)
            
            # Positive perturbation
            if hasattr(model, 'theta'):
                model.theta.data = theta + sigma * epsilon
            r_pos = evaluate_policy(model, env_name, num_episodes=1)
            
            # Negative perturbation
            if hasattr(model, 'theta'):
                model.theta.data = theta - sigma * epsilon
            r_neg = evaluate_policy(model, env_name, num_episodes=1)
            
            # ES gradient estimate
            grad += (r_pos - r_neg) * epsilon
            batch_rewards.extend([r_pos, r_neg])
        
        grad /= (2 * population_size * sigma)
        
        # Update theta
        theta += lr * grad
        if hasattr(model, 'theta'):
            model.theta.data = theta.clone()
        
        avg_reward = np.mean(batch_rewards)
        max_reward = np.max(batch_rewards)
        best_reward = max(best_reward, max_reward)
        rewards_history.append(avg_reward)
        
        if verbose and iter % 10 == 0:
            print(f"  Iter {iter}: avg={avg_reward:.1f}, max={max_reward:.1f}, best={best_reward:.1f}")
    
    return {
        'best_reward': best_reward,
        'final_avg_reward': rewards_history[-1] if rewards_history else 0,
        'rewards_history': rewards_history
    }


def train_subspace_es(
    env_name: str,
    subspace_dim: int,
    base_policy_fn,
    iterations: int = 100,
    **es_kwargs
) -> float:
    """Train policy in subspace using ES."""
    base_policy = base_policy_fn()
    model = SubspaceModel(base_policy, subspace_dim, projection_type='dense')
    
    result = train_es(model, env_name, iterations=iterations, **es_kwargs)
    return result['best_reward']


def run_cartpole_experiment(
    dimensions: list = None,
    iterations: int = 100,
    save_dir: str = 'results'
):
    """
    Run CartPole intrinsic dimension experiment.
    
    Expected d_int90 ≈ 25
    """
    print("=" * 60)
    print("CartPole Intrinsic Dimension Experiment")
    print("=" * 60)
    
    if not HAS_GYM:
        print("Skipping: gymnasium/gym not installed")
        return None
    
    if dimensions is None:
        dimensions = [5, 10, 15, 20, 25, 30, 50, 100]
    
    # Get baseline (solve threshold for CartPole is 475)
    baseline_reward = 500.0  # Max possible
    threshold = 0.9 * baseline_reward  # 450
    
    rewards = []
    for d in dimensions:
        print(f"\nTraining with d = {d}...")
        reward = train_subspace_es(
            'CartPole-v1', d, cartpole_policy,
            iterations=iterations
        )
        rewards.append(reward)
        print(f"  d={d}: reward={reward:.1f} (threshold: {threshold:.1f})")
    
    # Find d_int90
    d_int90 = dimensions[-1]
    for d, r in zip(dimensions, rewards):
        if r >= threshold:
            d_int90 = d
            break
    
    print(f"\n{'=' * 40}")
    print(f"CartPole d_int90: {d_int90} (expected: ~25)")
    
    # Plot
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, rewards, 'bo-', markersize=8, linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'90% threshold = {threshold}')
    plt.axvline(x=d_int90, color='orange', linestyle=':', label=f'd_int90 = {d_int90}')
    plt.xlabel('Subspace Dimension (d)', fontsize=12)
    plt.ylabel('Best Reward', fontsize=12)
    plt.title('CartPole: Intrinsic Dimension', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'cartpole_intrinsic_dim.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return {'dimensions': dimensions, 'rewards': rewards, 'd_int90': d_int90}


def run_pendulum_experiment(
    dimensions: list = None,
    iterations: int = 100,
    save_dir: str = 'results'
):
    """
    Run Inverted Pendulum intrinsic dimension experiment.
    
    Expected d_int90 ≈ 4 (very simple task)
    """
    print("=" * 60)
    print("Inverted Pendulum Intrinsic Dimension Experiment")
    print("=" * 60)
    
    if not HAS_GYM:
        print("Skipping: gymnasium/gym not installed")
        return None
    
    if dimensions is None:
        dimensions = [1, 2, 3, 4, 5, 10, 20, 50]
    
    # Inverted Pendulum reward threshold
    baseline_reward = 1000.0
    threshold = 0.9 * baseline_reward
    
    # Environment name differs between gym versions
    env_name = 'InvertedPendulum-v4' if USE_NEW_API else 'InvertedPendulum-v2'
    
    rewards = []
    for d in dimensions:
        print(f"\nTraining with d = {d}...")
        reward = train_subspace_es(
            env_name, d, pendulum_policy,
            iterations=iterations
        )
        rewards.append(reward)
        print(f"  d={d}: reward={reward:.1f}")
    
    d_int90 = dimensions[-1]
    for d, r in zip(dimensions, rewards):
        if r >= threshold:
            d_int90 = d
            break
    
    print(f"\nInverted Pendulum d_int90: {d_int90} (expected: ~4)")
    
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, rewards, 'go-', markersize=8, linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'90% threshold')
    plt.axvline(x=d_int90, color='orange', linestyle=':', label=f'd_int90 = {d_int90}')
    plt.xlabel('Subspace Dimension (d)', fontsize=12)
    plt.ylabel('Best Reward', fontsize=12)
    plt.title('Inverted Pendulum: Intrinsic Dimension', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'pendulum_intrinsic_dim.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return {'dimensions': dimensions, 'rewards': rewards, 'd_int90': d_int90}


def run_all_rl_experiments(save_dir: str = 'results'):
    """Run all RL experiments and create summary plot."""
    if not HAS_GYM:
        print("Cannot run RL experiments: gymnasium/gym not installed")
        return
    
    results = {}
    
    # Run experiments
    results['cartpole'] = run_cartpole_experiment(save_dir=save_dir)
    results['pendulum'] = run_pendulum_experiment(save_dir=save_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("RL Experiments Summary")
    print("=" * 60)
    print(f"{'Environment':>20} | {'Measured d_int90':>15} | {'Paper d_int90':>15}")
    print("-" * 55)
    
    expected = {
        'cartpole': 25,
        'pendulum': 4,
        'humanoid': 700,
        'pong': 6000
    }
    
    for env, result in results.items():
        if result:
            print(f"{env:>20} | {result['d_int90']:>15} | {expected[env]:>15}")
    
    print("\nNote: Humanoid and Atari Pong require additional dependencies")
    print("and significant compute resources.")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Intrinsic Dimension Experiments')
    parser.add_argument('--env', type=str, default='all', 
                       choices=['all', 'cartpole', 'pendulum', 'humanoid', 'pong'])
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()
    
    if args.env == 'all':
        run_all_rl_experiments(save_dir=args.save_dir)
    elif args.env == 'cartpole':
        run_cartpole_experiment(iterations=args.iterations, save_dir=args.save_dir)
    elif args.env == 'pendulum':
        run_pendulum_experiment(iterations=args.iterations, save_dir=args.save_dir)
    else:
        print(f"Environment {args.env} not yet fully implemented")
