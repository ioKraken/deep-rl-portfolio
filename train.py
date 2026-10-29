"""
Training orchestrator for the PPO portfolio agent.
Manages the training loop, logging, and model checkpointing.
"""

import os
import time
import numpy as np
import torch

import config
from environment import PortfolioEnv
from ppo import PPOAgent


def select_device() -> str:
    """RL_DEVICE env var wins; otherwise MPS if available, else CPU."""
    override = os.environ.get("RL_DEVICE")
    if override:
        return override
    return "mps" if torch.backends.mps.is_available() else "cpu"


def evaluate_policy(agent: PPOAgent, features: np.ndarray, prices: np.ndarray) -> dict:
    """Run the deterministic policy once over the whole slice and return episode stats."""
    env = PortfolioEnv(features, prices, is_train=False)
    obs = env.reset()
    while not env.done:
        window, current_weights = obs
        action, _, _, _ = agent.select_action(window, current_weights, deterministic=True)
        obs, _, _, _ = env.step(action)
    return env.get_episode_stats()


def train(train_features: np.ndarray, train_prices: np.ndarray) -> PPOAgent:
    """
    Train the PPO agent on the training data.

    Args:
        train_features: (n_train_days, n_assets, n_features)
        train_prices:   (n_train_days, n_assets)

    Returns:
        Trained PPOAgent
    """
    config.ensure_dirs()

    # Set seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = select_device()
    print(f"[Train] Using device: {device}")

    # Hold out the tail of the training period for model selection. The slice keeps
    # LOOKBACK_WINDOW days of leading context so every validation day is scored.
    n_val = int(len(train_features) * config.VAL_RATIO)
    fit_end = len(train_features) - n_val
    val_start = max(fit_end - config.LOOKBACK_WINDOW, 0)
    val_features, val_prices = train_features[val_start:], train_prices[val_start:]
    train_features, train_prices = train_features[:fit_end], train_prices[:fit_end]
    print(f"[Train] Fit: {len(train_features)} days | Validation: {n_val} days")

    # Create environment and agent
    env = PortfolioEnv(train_features, train_prices, is_train=True)
    obs_feature_dim = env.obs_feature_dim
    n_assets = env.n_assets

    agent = PPOAgent(obs_feature_dim, n_assets, device=device)

    print(f"[Train] Model parameters: {sum(p.numel() for p in agent.model.parameters()):,}")
    print(f"[Train] Starting training for {config.NUM_TRAINING_ITERATIONS} iterations")
    print(f"[Train] Trajectory length: ~{config.TRAJECTORY_LENGTH} days")
    print()

    best_val_sharpe = -np.inf
    best_model_path = os.path.join(config.MODEL_DIR, "best_model.pt")

    training_log = {
        "rewards": [],
        "sharpe_ratios": [],
        "returns": [],
        "policy_losses": [],
        "value_losses": [],
        "val_sharpe": [],
    }

    start_time = time.time()

    for iteration in range(1, config.NUM_TRAINING_ITERATIONS + 1):
        # Collect trajectory
        total_reward, steps, last_value = agent.collect_trajectory(env)

        # PPO update
        update_stats = agent.update(last_value)

        # Get episode stats
        episode_stats = env.get_episode_stats()
        ep_sharpe = episode_stats.get("sharpe_ratio", 0)
        ep_return = episode_stats.get("annualized_return", 0)

        # Log
        training_log["rewards"].append(total_reward)
        training_log["sharpe_ratios"].append(ep_sharpe)
        training_log["returns"].append(ep_return)
        training_log["policy_losses"].append(update_stats["policy_loss"])
        training_log["value_losses"].append(update_stats["value_loss"])

        # Checkpoint on validation Sharpe (a single random training window is too noisy to select on)
        val_msg = ""
        if iteration % config.VAL_EVAL_EVERY == 0 or iteration == 1 or iteration == config.NUM_TRAINING_ITERATIONS:
            val_sharpe = evaluate_policy(agent, val_features, val_prices).get("sharpe_ratio", -np.inf)
            if not np.isfinite(val_sharpe):
                val_sharpe = -np.inf
            training_log["val_sharpe"].append((iteration, val_sharpe))
            val_msg = f" | Val Sharpe: {val_sharpe:>6.2f}"
            if val_sharpe > best_val_sharpe or not os.path.exists(best_model_path):
                best_val_sharpe = val_sharpe
                agent.save(best_model_path)

        # Print progress
        if iteration % 5 == 0 or iteration == 1:
            elapsed = time.time() - start_time
            avg_reward = np.mean(training_log["rewards"][-10:])
            avg_sharpe = np.mean(training_log["sharpe_ratios"][-10:])
            avg_return = np.mean(training_log["returns"][-10:])

            print(
                f"  Iter {iteration:>4}/{config.NUM_TRAINING_ITERATIONS} | "
                f"Reward: {avg_reward:>8.2f} | "
                f"Sharpe: {avg_sharpe:>6.2f} | "
                f"Ann.Ret: {avg_return*100:>7.2f}% | "
                f"π Loss: {update_stats['policy_loss']:>8.4f} | "
                f"V Loss: {update_stats['value_loss']:>8.4f} | "
                f"Entropy: {update_stats['entropy']:>6.3f}"
                f"{val_msg} | [{elapsed:.0f}s]"
            )

    total_time = time.time() - start_time
    print(f"\n[Train] Training complete in {total_time:.1f}s")
    print(f"[Train] Best validation Sharpe: {best_val_sharpe:.3f}")
    print(f"[Train] Best model saved to {best_model_path}")

    # Load best model for backtesting
    agent.load(best_model_path)

    # Save training curves
    _save_training_plots(training_log)

    return agent


def _save_training_plots(training_log: dict):
    """Save training progress plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.ensure_dirs()
    colors = {"bg": "#1a1a2e", "text": "#e0e0e0", "line": "#00D4AA", "line2": "#FF6B6B"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(colors["bg"])
    fig.suptitle("Training Progress", fontsize=18, color=colors["text"], fontweight="bold")

    data = [
        ("Episode Reward", training_log["rewards"]),
        ("Sharpe Ratio", training_log["sharpe_ratios"]),
        ("Annualized Return", training_log["returns"]),
        ("Policy Loss", training_log["policy_losses"]),
    ]

    for ax, (title, values) in zip(axes.flatten(), data):
        ax.set_facecolor(colors["bg"])
        ax.plot(values, color=colors["line"], alpha=0.3, linewidth=0.5)

        # Smoothed
        if len(values) > 10:
            smoothed = np.convolve(values, np.ones(10) / 10, mode="valid")
            ax.plot(range(9, 9 + len(smoothed)), smoothed, color=colors["line"], linewidth=2)

        ax.set_title(title, fontsize=13, color=colors["text"])
        ax.tick_params(colors=colors["text"])
        for spine in ax.spines.values():
            spine.set_color(colors["text"])

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "training_progress.png"), dpi=150, facecolor=colors["bg"])
    plt.close()
    print(f"[Train] Training plots saved to {config.RESULTS_DIR}/training_progress.png")
