import json
import os

import numpy as np
import pytest

import config
from backtest import compute_metrics, generate_plots, print_results, run_backtest, save_metrics
from ppo import PPOAgent


def _metrics(daily, bench=None):
    daily = np.asarray(daily, dtype=float)
    pv = np.cumprod(np.r_[1.0, 1 + daily])
    bench = daily if bench is None else np.asarray(bench, dtype=float)
    bv = np.cumprod(np.r_[1.0, 1 + bench])
    return compute_metrics(daily, pv, bench, bv)


def test_sortino_uses_downside_deviation_over_all_observations():
    x = np.array([0.01, -0.01, 0.02, -0.02, 0.01, 0.0])
    ex = x - config.RISK_FREE_RATE / 252
    expected = ex.mean() / np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(252)
    assert _metrics(x)["sortino_ratio"] == pytest.approx(expected, rel=1e-4)


def test_sharpe_and_drawdown_known_values():
    x = np.array([0.10, -0.20, 0.05])
    m = _metrics(x)
    assert m["max_drawdown"] == pytest.approx(-0.20)
    ex = x - config.RISK_FREE_RATE / 252
    assert m["sharpe_ratio"] == pytest.approx(ex.mean() / ex.std() * np.sqrt(252), rel=1e-6)
    assert m["total_return"] == pytest.approx(1.1 * 0.8 * 1.05 - 1)
    assert m["win_rate"] == pytest.approx(2 / 3)


def test_beta_of_identical_series_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 200)
    assert _metrics(x)["beta"] == pytest.approx(1.0, abs=1e-3)


def test_total_loss_gives_finite_metrics():
    m = _metrics([-0.5, -0.9, -1.0])
    assert all(np.isfinite(v) for k, v in m.items() if isinstance(v, float))


def test_run_backtest_is_aligned_and_deterministic(synthetic):
    features, prices, bench = synthetic
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    r1 = run_backtest(agent, features, prices, bench)
    r2 = run_backtest(agent, features, prices, bench)
    assert np.array_equal(r1["portfolio_values"], r2["portfolio_values"])

    n_steps = len(features) - 1 - config.LOOKBACK_WINDOW
    assert len(r1["daily_returns"]) == n_steps == len(r1["bench_returns"])
    assert len(r1["portfolio_values"]) == len(r1["benchmark_values"]) == n_steps + 1
    assert r1["weights_history"].max() <= config.MAX_POSITION_SIZE + 1e-9
    assert r1["benchmark_values"][0] == 1.0
    assert "avg_daily_turnover" in r1["metrics"]


def test_plots_and_metrics_files_are_written(synthetic, tmp_path):
    features, prices, bench = synthetic
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    result = run_backtest(agent, features, prices, bench)
    generate_plots(result, str(tmp_path))
    save_metrics(result["metrics"], str(tmp_path))

    for name in ["cumulative_returns", "drawdown", "rolling_sharpe", "weight_allocation",
                 "monthly_returns_heatmap", "performance_summary"]:
        assert os.path.getsize(tmp_path / f"{name}.png") > 1000
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["n_trading_days"] == len(result["daily_returns"])
    assert "turnovers" not in saved


def test_print_results_runs(synthetic, capsys):
    features, prices, bench = synthetic
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    print_results(run_backtest(agent, features, prices, bench)["metrics"])
    assert "BACKTEST RESULTS" in capsys.readouterr().out


def test_equal_weight_baseline_matches_uniform_env_run(synthetic):
    from backtest import run_equal_weight
    features, prices, bench = synthetic
    values, rets = run_equal_weight(features, prices)
    t0 = config.LOOKBACK_WINDOW
    expected = np.cumprod(1 + (prices[t0 + 1:] / prices[t0:-1] - 1).mean(axis=1))
    assert values[1:] == pytest.approx(expected)

    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS)
    m = run_backtest(agent, features, prices, bench)["metrics"]
    assert m["equal_weight_sharpe_ratio"] == pytest.approx(
        _metrics(rets)["sharpe_ratio"], rel=1e-3)
