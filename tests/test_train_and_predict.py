import json
import os

import numpy as np
import pandas as pd
import pytest

import config
import predict
from backtest import run_backtest
from ppo import PPOAgent
from train import evaluate_policy, train


def test_train_smoke_saves_loadable_checkpoint(synthetic, tiny_config):
    features, prices, bench = synthetic
    agent = train(features, prices)
    assert os.path.exists(os.path.join(config.MODEL_DIR, "best_model.pt"))
    assert os.path.exists(os.path.join(config.RESULTS_DIR, "training_progress.png"))

    result = run_backtest(agent, features, prices, bench)
    assert np.isfinite(result["metrics"]["sharpe_ratio"])


def test_training_is_reproducible_for_a_given_seed(synthetic, tiny_config):
    features, prices, _ = synthetic
    a = train(features, prices)
    b = train(features, prices)
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert np.allclose(pa.detach().numpy(), pb.detach().numpy())


def test_validation_slice_is_not_trained_on(synthetic, tiny_config, monkeypatch):
    """Training episodes must stay inside the fit slice, never reaching validation days."""
    features, prices, _ = synthetic
    n_val = int(len(features) * config.VAL_RATIO)
    fit_end = len(features) - n_val
    seen_max = []
    orig = PPOAgent.collect_trajectory

    def spy(self, env):
        out = orig(self, env)
        seen_max.append(env.t)
        return out

    monkeypatch.setattr(PPOAgent, "collect_trajectory", spy)
    train(features, prices)
    assert max(seen_max) <= fit_end - 1


def test_evaluate_policy_returns_stats(synthetic):
    features, prices, _ = synthetic
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    stats = evaluate_policy(agent, features, prices)
    assert {"sharpe_ratio", "annualized_return", "n_days"} <= set(stats)


def _price_table(n):
    rng = np.random.default_rng(0)
    cols = config.TICKERS + [config.BENCHMARK_TICKER]
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, len(cols))), axis=0),
                        index=pd.bdate_range("2024-01-01", periods=n), columns=cols)


def test_predict_returns_valid_capped_weights():
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    prices = _price_table(predict.MIN_HISTORY_DAYS + 20)
    out = predict.compute_target_weights(prices, agent)
    w = np.array(list(out["weights"].values()))
    assert list(out["weights"]) == config.TICKERS
    assert out["as_of"] == str(prices.index[-1].date())
    assert w.sum() == pytest.approx(1.0) and w.max() <= config.MAX_POSITION_SIZE + 1e-9


def test_predict_rejects_short_history():
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    with pytest.raises(ValueError, match="trading days"):
        predict.compute_target_weights(_price_table(50), agent)


def test_predict_window_is_what_the_environment_shows_the_agent():
    """Live inference must see exactly the observation the env builds at t = len(features)."""
    import data_pipeline as dp
    from environment import PortfolioEnv

    prices = _price_table(300)
    feats, _ = dp.compute_features(prices)
    feats = dp.normalize_features(feats, config.LOOKBACK_WINDOW)

    seen = {}

    class Spy:
        def select_action(self, window, current_weights, deterministic=False):
            seen["window"] = window
            return np.ones(config.N_ASSETS), None, 0.0, 0.0

    predict.compute_target_weights(prices, Spy())

    env = PortfolioEnv(feats, prices[config.TICKERS].values)
    window, _ = env.reset(start_idx=len(feats))
    assert np.array_equal(seen["window"], window)


def test_load_current_weights_validates(tmp_path):
    good = tmp_path / "w.json"
    good.write_text(json.dumps({"AAPL": 3, "MSFT": 1}))
    w = predict.load_current_weights(str(good))
    assert w.sum() == pytest.approx(1.0) and w[config.TICKERS.index("AAPL")] == pytest.approx(0.75)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"NOPE": 1}))
    with pytest.raises(ValueError):
        predict.load_current_weights(str(bad))


def test_load_agent_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load_agent(str(tmp_path / "nope.pt"))
