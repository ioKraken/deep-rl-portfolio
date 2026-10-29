import numpy as np
import pytest

import config
from environment import PortfolioEnv, apply_position_cap


def test_cap_holds_for_extreme_softmax_inputs():
    rng = np.random.default_rng(0)
    for _ in range(500):
        z = rng.normal(0, 4, config.N_ASSETS)
        w = apply_position_cap(np.exp(z) / np.exp(z).sum(), config.MAX_POSITION_SIZE)
        assert w.max() <= config.MAX_POSITION_SIZE + 1e-9
        assert w.min() >= 0
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_cap_one_hot_input():
    w = apply_position_cap(np.eye(config.N_ASSETS)[0], 0.10)
    assert w.max() <= 0.10 + 1e-9 and w.sum() == pytest.approx(1.0)


def test_cap_degenerate_inputs_fall_back_to_equal_weight():
    for bad in (np.zeros(10), np.full(10, np.nan), -np.ones(10)):
        assert np.allclose(apply_position_cap(bad, 0.5), 0.1)


def test_cap_infeasible_raises():
    with pytest.raises(ValueError):
        apply_position_cap(np.ones(5) / 5, 0.1)


def test_step_return_uses_next_day_and_charges_costs(synthetic):
    features, prices, _ = synthetic
    env = PortfolioEnv(features, prices, is_train=False)
    env.reset()
    target = np.zeros(config.N_ASSETS)
    target[:10] = 0.1
    _, _, _, info = env.step(target)

    t = config.LOOKBACK_WINDOW + 1
    day_ret = prices[t] / prices[t - 1] - 1
    turnover = np.abs(target - 1 / config.N_ASSETS).sum()
    expected = target @ day_ret - config.TRANSACTION_COST * turnover
    assert info["portfolio_return"] == pytest.approx(expected)
    assert info["turnover"] == pytest.approx(turnover)


def test_observation_never_includes_current_day(synthetic):
    features, prices, _ = synthetic
    env = PortfolioEnv(features, prices, is_train=False)
    (window, _) = env.reset()
    assert env.t == config.LOOKBACK_WINDOW
    assert np.array_equal(window, features[: config.LOOKBACK_WINDOW].reshape(config.LOOKBACK_WINDOW, -1))


def test_eval_episode_terminates_at_end_of_data(synthetic):
    features, prices, _ = synthetic
    env = PortfolioEnv(features, prices, is_train=False)
    env.reset()
    steps = 0
    while not env.done:
        _, _, _, info = env.step(np.ones(config.N_ASSETS))
        steps += 1
    assert info["terminated"] and not info["truncated"]
    assert steps == len(features) - 1 - config.LOOKBACK_WINDOW
    with pytest.raises(RuntimeError):
        env.step(np.ones(config.N_ASSETS))


def test_train_episode_ends_as_truncated_not_terminated(synthetic):
    features, prices, _ = synthetic
    env = PortfolioEnv(features, prices, is_train=True)
    env.reset(start_idx=config.LOOKBACK_WINDOW)
    while not env.done:
        _, _, _, info = env.step(np.ones(config.N_ASSETS))
    assert info["truncated"] and not info["terminated"]
    assert len(env.portfolio_returns) == config.TRAJECTORY_LENGTH


def test_train_env_rejects_too_little_data():
    n = config.LOOKBACK_WINDOW
    env_args = (np.zeros((n, 3, 5)), np.ones((n, 3)))
    with pytest.raises(ValueError):
        PortfolioEnv(*env_args, is_train=True)


def test_rewards_are_finite_and_bounded(synthetic):
    features, prices, _ = synthetic
    env = PortfolioEnv(features, prices, is_train=True)
    env.reset(start_idx=config.LOOKBACK_WINDOW)
    rewards = []
    while not env.done:
        rewards.append(env.step(np.random.rand(config.N_ASSETS))[1])
    assert np.all(np.isfinite(rewards)) and np.max(np.abs(rewards)) <= 2.0


def test_episode_stats_total_loss_does_not_produce_nan(monkeypatch):
    monkeypatch.setattr(config, "MAX_POSITION_SIZE", 1.0)
    n = 80
    prices = np.vstack([np.linspace(100, 1, n)] * 3).T
    env = PortfolioEnv(np.zeros((n, 3, 5)), prices, is_train=False)
    env.reset()
    while not env.done:
        env.step(np.ones(3))
    assert np.isfinite(env.get_episode_stats()["annualized_return"])


def test_excess_reward_is_zero_for_equal_weight_and_signed_otherwise(synthetic, monkeypatch):
    monkeypatch.setattr(config, "REWARD_MODE", "excess")
    monkeypatch.setattr(config, "TRANSACTION_COST", 0.0)
    features, prices, _ = synthetic
    env = PortfolioEnv(features, prices, is_train=False)
    env.reset()
    _, r_equal, _, _ = env.step(np.ones(config.N_ASSETS))
    assert r_equal == pytest.approx(0.0, abs=1e-9)

    env.reset()
    day = env.returns[config.LOOKBACK_WINDOW + 1]
    best = np.zeros(config.N_ASSETS)
    best[np.argsort(day)[-10:]] = 1.0       # the ten best performers of the next day
    _, r_best, _, _ = env.step(best)
    assert r_best > 0
