"""
Portfolio Trading Environment (Gym-style).
Implements differentiable Sharpe Ratio reward with transaction costs and position constraints.
"""

import numpy as np
import config


class PortfolioEnv:
    """
    A portfolio environment for RL-based trading.

    State:  (lookback, n_assets * n_features) window  +  current weights
    Action: target portfolio weights (n_assets,) — will be clamped & normalized
    Reward: online differential Sharpe ratio (Moody & Saffell) with transaction cost penalty
    """

    def __init__(self, features: np.ndarray, prices: np.ndarray, is_train: bool = True):
        self.features = features
        self.prices = prices
        self.n_days = len(features)
        self.n_assets = features.shape[1]
        self.n_features = features.shape[2]
        self.is_train = is_train
        self.lookback = config.LOOKBACK_WINDOW

        # State dimension
        self.obs_feature_dim = self.n_assets * self.n_features
        self.obs_dim = self.obs_feature_dim + self.n_assets

        # Pre-compute daily returns from prices
        self.returns = np.zeros((self.n_days, self.n_assets))
        self.returns[1:] = (self.prices[1:] - self.prices[:-1]) / (self.prices[:-1] + 1e-8)

        self.reset()

    def reset(self, start_idx=None):
        """Reset environment, return initial observation."""
        if start_idx is not None:
            self.t = start_idx
        elif self.is_train:
            max_start = self.n_days - config.TRAJECTORY_LENGTH - 1
            max_start = max(self.lookback, max_start)
            self.t = np.random.randint(self.lookback, max_start + 1)
        else:
            self.t = self.lookback

        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.portfolio_returns = []
        self.done = False

        # Online Sharpe tracking (Moody & Saffell EMA)
        self._A = 0.0
        self._B = 1e-8

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        window = self.features[self.t - self.lookback : self.t]
        window_flat = window.reshape(self.lookback, -1)
        return window_flat, self.weights.copy()

    def _apply_constraints(self, target_weights: np.ndarray) -> np.ndarray:
        w = np.clip(target_weights, 0.0, config.MAX_POSITION_SIZE)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(self.n_assets) / self.n_assets
        w = np.clip(w, 0.0, config.MAX_POSITION_SIZE)
        w = w / (w.sum() + 1e-8)
        return w

    def step(self, action: np.ndarray):
        target_weights = self._apply_constraints(action)

        # Transaction costs
        turnover = np.sum(np.abs(target_weights - self.weights))
        transaction_cost = config.TRANSACTION_COST * turnover

        self.weights = target_weights
        self.t += 1

        asset_returns = self.returns[self.t]
        port_return = np.dot(self.weights, asset_returns) - transaction_cost

        self.portfolio_value *= (1 + port_return)
        self.portfolio_returns.append(port_return)

        reward = self._compute_reward(port_return)

        if self.is_train:
            steps_taken = len(self.portfolio_returns)
            self.done = (self.t >= self.n_days - 1) or (steps_taken >= config.TRAJECTORY_LENGTH)
        else:
            self.done = (self.t >= self.n_days - 1)

        obs = self._get_obs()

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": port_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
        }

        return obs, reward, self.done, info

    def _compute_reward(self, port_return: float) -> float:
        """
        Online Differential Sharpe Ratio (Moody & Saffell, 1998).
        Reward is clipped to [-2, 2] for training stability.
        """
        R = port_return
        eta = 0.05

        dA = R - self._A
        dB = R * R - self._B

        denom = self._B - self._A ** 2
        if denom > 1e-12:
            dS = (self._B * dA - 0.5 * self._A * dB) / (denom ** 1.5)
        else:
            dS = R * 100

        self._A += eta * dA
        self._B += eta * dB

        reward = np.clip(dS * 0.1, -2.0, 2.0)
        return reward

    def get_episode_stats(self) -> dict:
        returns = np.array(self.portfolio_returns)
        if len(returns) == 0:
            return {}

        total_return = self.portfolio_value - 1.0
        n_days = len(returns)
        ann_factor = 252 / max(n_days, 1)
        ann_return = (1 + total_return) ** ann_factor - 1

        daily_rf = config.RISK_FREE_RATE / 252
        excess = returns - daily_rf
        sharpe = np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "sharpe_ratio": sharpe,
            "n_days": n_days,
            "final_portfolio_value": self.portfolio_value,
        }
