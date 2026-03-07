"""
Policy Networks for Reinforcement Learning experiments.

The paper tests intrinsic dimension on several RL tasks:
- CartPole: d_int90 ≈ 25 (very simple)
- Inverted Pendulum: d_int90 ≈ 4 (simpler than CartPole)
- Humanoid: d_int90 ≈ 700 (complex continuous control)
- Atari Pong: d_int90 ≈ 6000 (visual RL)

Training uses Evolution Strategies (ES) instead of policy gradients
because ES works better in low-dimensional subspaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FCPolicy(nn.Module):
    """
    Fully connected policy network for simple environments.
    
    Used for: CartPole, Inverted Pendulum, Humanoid, etc.
    
    Args:
        obs_dim: Observation space dimension
        act_dim: Action space dimension
        hidden_dims: Hidden layer sizes
        continuous: Whether action space is continuous
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...] = (32,),
        continuous: bool = False
    ):
        super().__init__()
        
        self.continuous = continuous
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Tanh is common for RL policies
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, act_dim))
        
        self.network = nn.Sequential(*layers)
        
        # For continuous control, optionally bound outputs
        if continuous:
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = None
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action logits/values from observation.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Action logits (discrete) or action values (continuous)
        """
        x = self.network(obs)
        
        if self.output_activation:
            x = self.output_activation(x)
        
        return x
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from observation (for evaluation)."""
        with torch.no_grad():
            output = self.forward(obs)
            
            if self.continuous:
                return output
            else:
                return output.argmax(dim=-1)


class ConvPolicy(nn.Module):
    """
    Convolutional policy network for visual RL (Atari).
    
    Architecture similar to DQN:
    - Conv(32, 8x8, stride=4)
    - Conv(64, 4x4, stride=2)
    - Conv(64, 3x3, stride=1)
    - FC(512)
    - FC(action_dim)
    
    Paper reports d_int90 ≈ 6000 for Atari Pong (~1M parameters).
    """
    
    def __init__(
        self,
        input_channels: int = 4,  # Stacked frames
        act_dim: int = 18,  # Atari has up to 18 actions
        input_size: int = 84  # Standard Atari preprocessing
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calculate conv output size
        # 84 -> 20 -> 9 -> 7 (for 84x84 input)
        def conv_out_size(size, kernel, stride):
            return (size - kernel) // stride + 1
        
        size = input_size
        size = conv_out_size(size, 8, 4)  # 20
        size = conv_out_size(size, 4, 2)  # 9
        size = conv_out_size(size, 3, 1)  # 7
        
        self.fc_input = 64 * size * size
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action logits from observation.
        
        Args:
            obs: Image observation (batch, channels, height, width)
            
        Returns:
            Action logits
        """
        # Normalize pixel values if needed
        if obs.max() > 1.0:
            obs = obs / 255.0
        
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from observation (for evaluation)."""
        with torch.no_grad():
            return self.forward(obs).argmax(dim=-1)


# Convenience functions for specific environments

def cartpole_policy() -> FCPolicy:
    """Policy for CartPole-v1 (obs_dim=4, act_dim=2)."""
    return FCPolicy(obs_dim=4, act_dim=2, hidden_dims=(32,), continuous=False)


def pendulum_policy() -> FCPolicy:
    """Policy for InvertedPendulum-v4 (obs_dim=4, act_dim=1)."""
    return FCPolicy(obs_dim=4, act_dim=1, hidden_dims=(32,), continuous=True)


def humanoid_policy() -> FCPolicy:
    """Policy for Humanoid-v4 (obs_dim=376, act_dim=17)."""
    return FCPolicy(obs_dim=376, act_dim=17, hidden_dims=(256, 256), continuous=True)


def atari_pong_policy() -> ConvPolicy:
    """Policy for ALE/Pong-v5 (6 actions for Pong)."""
    return ConvPolicy(input_channels=4, act_dim=6, input_size=84)
