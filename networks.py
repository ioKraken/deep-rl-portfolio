"""
LSTM-based Actor-Critic Network for portfolio allocation.
Actor outputs Gaussian logits → softmax → portfolio weights.
Critic outputs scalar value estimate.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config


class LSTMFeatureExtractor(nn.Module):
    """Processes a sequence of market features through a multi-layer LSTM."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(config.LSTM_HIDDEN_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, input_dim)
        Returns:
            h: (batch, hidden_size) — last hidden state
        """
        lstm_out, _ = self.lstm(x)  # (batch, lookback, hidden)
        h = lstm_out[:, -1, :]       # Last timestep
        h = self.layer_norm(h)
        return h


class ActorHead(nn.Module):
    """
    Outputs mean logits for portfolio weights in unconstrained space.
    Uses Gaussian distribution + softmax to produce valid portfolio weights.
    This is MPS-compatible (unlike Dirichlet).
    """

    def __init__(self, hidden_dim: int, n_assets: int):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim + n_assets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_assets),
        )
        self.log_std = nn.Parameter(torch.zeros(n_assets) - 0.5)  # Learnable log std
        self.n_assets = n_assets

    def forward(self, features: torch.Tensor, current_weights: torch.Tensor):
        """
        Args:
            features: (batch, hidden_dim) from LSTM
            current_weights: (batch, n_assets) current portfolio weights
        Returns:
            mean: (batch, n_assets) logits
            std: (batch, n_assets) standard deviations
        """
        x = torch.cat([features, current_weights], dim=-1)
        mean = self.mean_net(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class CriticHead(nn.Module):
    """Outputs scalar state value estimate."""

    def __init__(self, hidden_dim: int, n_assets: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + n_assets, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, current_weights: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, current_weights], dim=-1)
        return self.net(x).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic for PPO-based portfolio optimization.
    Uses LSTM to process market history, then Actor outputs Gaussian logits→softmax weights,
    Critic outputs value estimate.
    """

    def __init__(self, obs_feature_dim: int, n_assets: int):
        super().__init__()
        self.feature_extractor = LSTMFeatureExtractor(obs_feature_dim)
        self.actor = ActorHead(config.LSTM_HIDDEN_SIZE, n_assets)
        self.critic = CriticHead(config.LSTM_HIDDEN_SIZE, n_assets)
        self.n_assets = n_assets

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

    def forward(self, market_window: torch.Tensor, current_weights: torch.Tensor):
        """
        Args:
            market_window: (batch, lookback, obs_feature_dim)
            current_weights: (batch, n_assets)
        Returns:
            mean, std: Gaussian params in logit space
            value: (batch,) value estimates
        """
        features = self.feature_extractor(market_window)
        mean, std = self.actor(features, current_weights)
        value = self.critic(features, current_weights)
        return mean, std, value

    def _logits_to_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert unconstrained logits to portfolio weights via softmax."""
        return F.softmax(logits, dim=-1)

    def act(self, market_window: torch.Tensor, current_weights: torch.Tensor, deterministic=False):
        """
        Sample action + compute log prob and value for a single step.
        Returns: (numpy weights, numpy logits, scalar log_prob, scalar value)
        """
        mean, std, value = self.forward(market_window, current_weights)
        dist = Normal(mean, std)

        if deterministic:
            logits = mean
        else:
            logits = dist.rsample()

        # Sum log probs across assets for joint probability
        log_prob = dist.log_prob(logits).sum(dim=-1)

        # Convert logits → weights
        weights = self._logits_to_weights(logits)

        return weights.detach().cpu().numpy(), logits.detach().cpu().numpy(), log_prob, value

    def evaluate(self, market_window: torch.Tensor, current_weights: torch.Tensor, action_logits: torch.Tensor):
        """
        Re-evaluate actions for PPO update.
        Returns log_probs, values, entropy.
        """
        mean, std, value = self.forward(market_window, current_weights)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action_logits).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value, entropy
