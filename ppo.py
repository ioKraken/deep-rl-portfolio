"""
Proximal Policy Optimization (PPO) for portfolio allocation.
Implements clipped objective, GAE, entropy bonus, and minibatch updates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

import config
from networks import ActorCritic
from environment import PortfolioEnv


class RolloutBuffer:
    """Stores trajectory data for PPO updates."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.market_windows = []
        self.current_weights = []
        self.actions = []          # softmax weights for env
        self.action_logits = []    # raw logits for evaluate()
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, market_window, current_weight, action, action_logit, log_prob, reward, value, done):
        self.market_windows.append(market_window)
        self.current_weights.append(current_weight)
        self.actions.append(action)
        self.action_logits.append(action_logit)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value: float):
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)

        n = len(rewards)
        advantages = np.zeros(n)
        gae = 0

        for t in reversed(range(n)):
            delta = rewards[t] + config.GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values)
        return advantages, returns

    def get_tensors(self, advantages, returns, device):
        """Convert buffer to tensors for training."""
        market_w = torch.FloatTensor(np.array(self.market_windows)).to(device)
        weights = torch.FloatTensor(np.array(self.current_weights)).to(device)
        action_logits = torch.FloatTensor(np.array(self.action_logits)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        adv = torch.FloatTensor(advantages).to(device)
        ret = torch.FloatTensor(returns).to(device)

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return market_w, weights, action_logits, old_log_probs, adv, ret


class PPOAgent:
    """PPO agent for portfolio optimization."""

    def __init__(self, obs_feature_dim: int, n_assets: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.n_assets = n_assets

        self.model = ActorCritic(obs_feature_dim, n_assets).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, eps=1e-5)
        self.buffer = RolloutBuffer()

        # Tracking
        self.training_stats = defaultdict(list)

    def select_action(self, market_window: np.ndarray, current_weights: np.ndarray, deterministic=False):
        """Select action given observation. Returns (weights, logits, log_prob, value)."""
        mw = torch.FloatTensor(market_window).unsqueeze(0).to(self.device)
        cw = torch.FloatTensor(current_weights).unsqueeze(0).to(self.device)

        with torch.no_grad():
            weights, logits, log_prob, value = self.model.act(mw, cw, deterministic=deterministic)

        return weights.squeeze(0), logits.squeeze(0), log_prob.item(), value.item()

    def collect_trajectory(self, env: PortfolioEnv):
        """Collect one trajectory of experience."""
        self.buffer.clear()
        obs = env.reset()

        total_reward = 0
        steps = 0

        while not env.done:
            market_window, current_weights = obs
            weights, logits, log_prob, value = self.select_action(market_window, current_weights)

            obs, reward, done, info = env.step(weights)
            self.buffer.add(market_window, current_weights, weights, logits, log_prob, reward, value, done)

            total_reward += reward
            steps += 1

        # Get last value for GAE
        if not done:
            market_window, current_weights = obs
            _, _, _, last_value = self.select_action(market_window, current_weights)
        else:
            last_value = 0.0

        return total_reward, steps, last_value

    def update(self, last_value: float):
        """Perform PPO update using collected trajectory."""
        advantages, returns = self.buffer.compute_gae(last_value)
        market_w, weights, action_logits, old_log_probs, adv, ret = self.buffer.get_tensors(
            advantages, returns, self.device
        )

        n_samples = len(adv)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(config.PPO_EPOCHS):
            # Generate random minibatch indices
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, config.MINIBATCH_SIZE):
                end = min(start + config.MINIBATCH_SIZE, n_samples)
                mb_idx = indices[start:end]

                mb_market = market_w[mb_idx]
                mb_weights = weights[mb_idx]
                mb_logits = action_logits[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = ret[mb_idx]

                # Evaluate current policy on old actions
                new_log_probs, new_values, entropy = self.model.evaluate(
                    mb_market, mb_weights, mb_logits
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - config.PPO_CLIP, 1 + config.PPO_CLIP) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(new_values, mb_ret)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + config.VALUE_LOSS_COEF * value_loss
                    + config.ENTROPY_COEF * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_GRAD_NORM)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        avg_policy_loss = total_policy_loss / max(n_updates, 1)
        avg_value_loss = total_value_loss / max(n_updates, 1)
        avg_entropy = total_entropy / max(n_updates, 1)

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
        }

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
