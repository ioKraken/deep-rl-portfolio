"""
Backtesting engine: run a trained agent through test data,
compute performance metrics, and generate visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import config
from environment import PortfolioEnv
from ppo import PPOAgent


def run_backtest(agent: PPOAgent, test_features: np.ndarray, test_prices: np.ndarray,
                 benchmark_prices: np.ndarray) -> dict:
    """
    Run the agent through the test period day by day.
    Returns comprehensive performance data.
    """
    env = PortfolioEnv(test_features, test_prices, is_train=False)
    obs = env.reset()

    # Track everything
    portfolio_values = [1.0]
    daily_returns = []
    weights_history = [env.weights.copy()]
    turnovers = []

    while not env.done:
        market_window, current_weights = obs
        action, _, _, _ = agent.select_action(market_window, current_weights, deterministic=True)
        obs, reward, done, info = env.step(action)

        portfolio_values.append(info["portfolio_value"])
        daily_returns.append(info["portfolio_return"])
        weights_history.append(env.weights.copy())
        turnovers.append(info["turnover"])

    portfolio_values = np.array(portfolio_values)
    daily_returns = np.array(daily_returns)
    weights_history = np.array(weights_history)

    # Benchmark (S&P 500) performance
    bench = benchmark_prices[config.LOOKBACK_WINDOW:]
    bench_values = bench / bench[0]
    bench_returns = np.diff(bench) / bench[:-1]

    # Align lengths
    n = min(len(portfolio_values), len(bench_values))
    portfolio_values = portfolio_values[:n]
    bench_values = bench_values[:n]
    if len(bench_returns) > len(daily_returns):
        bench_returns = bench_returns[:len(daily_returns)]
    elif len(bench_returns) < len(daily_returns):
        daily_returns = daily_returns[:len(bench_returns)]

    # Compute metrics
    metrics = compute_metrics(daily_returns, portfolio_values, bench_returns, bench_values)
    metrics["turnovers"] = turnovers

    return {
        "portfolio_values": portfolio_values,
        "benchmark_values": bench_values,
        "daily_returns": daily_returns,
        "bench_returns": bench_returns,
        "weights_history": weights_history[:n],
        "metrics": metrics,
    }


def compute_metrics(daily_returns, portfolio_values, bench_returns, bench_values) -> dict:
    """Compute comprehensive performance metrics."""
    n_days = len(daily_returns)
    ann_factor = 252 / max(n_days, 1)

    # Portfolio metrics
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    ann_return = (1 + total_return) ** ann_factor - 1

    daily_rf = config.RISK_FREE_RATE / 252
    excess = daily_returns - daily_rf
    sharpe = np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252)

    # Sortino (downside deviation)
    downside = excess[excess < 0]
    downside_std = np.std(downside) + 1e-8 if len(downside) > 0 else 1e-8
    sortino = np.mean(excess) / downside_std * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_dd = np.min(drawdown)

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0

    # Win rate
    win_rate = np.mean(daily_returns > 0)

    # Daily volatility annualized
    ann_vol = np.std(daily_returns) * np.sqrt(252)

    # Benchmark metrics
    bench_total = bench_values[-1] / bench_values[0] - 1
    bench_ann = (1 + bench_total) ** ann_factor - 1
    bench_excess = bench_returns - daily_rf
    bench_sharpe = np.mean(bench_excess) / (np.std(bench_excess) + 1e-8) * np.sqrt(252)
    bench_peak = np.maximum.accumulate(bench_values)
    bench_dd = (bench_values - bench_peak) / bench_peak
    bench_max_dd = np.min(bench_dd)

    # Alpha and Beta
    if len(daily_returns) == len(bench_returns) and len(daily_returns) > 1:
        cov_matrix = np.cov(daily_returns, bench_returns)
        beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-8)
        alpha = (ann_return - config.RISK_FREE_RATE) - beta * (bench_ann - config.RISK_FREE_RATE)
    else:
        beta = 0
        alpha = 0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "annualized_volatility": ann_vol,
        "alpha": alpha,
        "beta": beta,
        "n_trading_days": n_days,
        "benchmark_total_return": bench_total,
        "benchmark_annualized_return": bench_ann,
        "benchmark_sharpe": bench_sharpe,
        "benchmark_max_drawdown": bench_max_dd,
    }


def generate_plots(backtest_result: dict, save_dir: str):
    """Generate comprehensive performance visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    pv = backtest_result["portfolio_values"]
    bv = backtest_result["benchmark_values"]
    dr = backtest_result["daily_returns"]
    br = backtest_result["bench_returns"]
    wh = backtest_result["weights_history"]
    metrics = backtest_result["metrics"]

    plt.style.use("seaborn-v0_8-darkgrid")
    colors = {"portfolio": "#00D4AA", "benchmark": "#FF6B6B", "bg": "#1a1a2e", "text": "#e0e0e0"}

    # ─── 1. Cumulative Returns ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])

    ax.plot(pv, color=colors["portfolio"], linewidth=2, label=f'RL Portfolio ({metrics["annualized_return"]*100:.1f}% ann.)')
    ax.plot(bv, color=colors["benchmark"], linewidth=2, alpha=0.8, label=f'S&P 500 ({metrics["benchmark_annualized_return"]*100:.1f}% ann.)')
    ax.fill_between(range(len(pv)), 1, pv, alpha=0.1, color=colors["portfolio"])

    ax.set_title("Cumulative Returns: RL Portfolio vs S&P 500", fontsize=16, color=colors["text"], fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=12, color=colors["text"])
    ax.set_ylabel("Portfolio Value ($1 initial)", fontsize=12, color=colors["text"])
    ax.legend(fontsize=12, facecolor=colors["bg"], edgecolor=colors["text"], labelcolor=colors["text"])
    ax.tick_params(colors=colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["text"])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cumulative_returns.png"), dpi=150, facecolor=colors["bg"])
    plt.close()

    # ─── 2. Drawdown Chart ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])

    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / peak * 100
    bench_peak = np.maximum.accumulate(bv)
    bench_dd = (bv - bench_peak) / bench_peak * 100

    ax.fill_between(range(len(dd)), dd, 0, alpha=0.4, color=colors["portfolio"], label="RL Portfolio")
    ax.fill_between(range(len(bench_dd)), bench_dd, 0, alpha=0.3, color=colors["benchmark"], label="S&P 500")
    ax.plot(dd, color=colors["portfolio"], linewidth=1)
    ax.plot(bench_dd, color=colors["benchmark"], linewidth=1, alpha=0.7)

    ax.set_title("Drawdown Analysis", fontsize=16, color=colors["text"], fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=12, color=colors["text"])
    ax.set_ylabel("Drawdown (%)", fontsize=12, color=colors["text"])
    ax.legend(fontsize=12, facecolor=colors["bg"], edgecolor=colors["text"], labelcolor=colors["text"])
    ax.tick_params(colors=colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["text"])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "drawdown.png"), dpi=150, facecolor=colors["bg"])
    plt.close()

    # ─── 3. Rolling Sharpe Ratio ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])

    window = 60
    if len(dr) > window:
        rolling_sharpe = pd.Series(dr).rolling(window).apply(
            lambda x: (x.mean() - config.RISK_FREE_RATE/252) / (x.std() + 1e-8) * np.sqrt(252)
        )
        min_len = min(len(rolling_sharpe), len(br))
        rolling_bench_sharpe = pd.Series(br[:min_len]).rolling(window).apply(
            lambda x: (x.mean() - config.RISK_FREE_RATE/252) / (x.std() + 1e-8) * np.sqrt(252)
        )
        ax.plot(rolling_sharpe, color=colors["portfolio"], linewidth=1.5, label="RL Portfolio")
        ax.plot(rolling_bench_sharpe, color=colors["benchmark"], linewidth=1.5, alpha=0.7, label="S&P 500")
        ax.axhline(y=0, color=colors["text"], linestyle="--", alpha=0.3)

    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=16, color=colors["text"], fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=12, color=colors["text"])
    ax.set_ylabel("Sharpe Ratio", fontsize=12, color=colors["text"])
    ax.legend(fontsize=12, facecolor=colors["bg"], edgecolor=colors["text"], labelcolor=colors["text"])
    ax.tick_params(colors=colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["text"])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rolling_sharpe.png"), dpi=150, facecolor=colors["bg"])
    plt.close()

    # ─── 4. Portfolio Weight Allocation Over Time ────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])

    # Show top 15 holdings by average weight
    avg_weights = np.mean(wh, axis=0)
    top_idx = np.argsort(avg_weights)[-15:]
    top_tickers = [config.TICKERS[i] for i in top_idx]
    top_weights = wh[:, top_idx]

    cmap = plt.cm.get_cmap("tab20", len(top_idx))
    ax.stackplot(range(len(top_weights)), top_weights.T,
                 labels=top_tickers,
                 colors=[cmap(i) for i in range(len(top_idx))],
                 alpha=0.8)

    ax.set_title("Portfolio Allocation (Top 15 Holdings)", fontsize=16, color=colors["text"], fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=12, color=colors["text"])
    ax.set_ylabel("Weight", fontsize=12, color=colors["text"])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9,
              facecolor=colors["bg"], edgecolor=colors["text"], labelcolor=colors["text"])
    ax.tick_params(colors=colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["text"])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weight_allocation.png"), dpi=150, facecolor=colors["bg"], bbox_inches="tight")
    plt.close()

    # ─── 5. Monthly Returns Heatmap ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])

    n_months = len(dr) // 21
    if n_months > 0:
        monthly_returns = []
        for m in range(n_months):
            s = m * 21
            e = min(s + 21, len(dr))
            monthly_ret = np.prod(1 + dr[s:e]) - 1
            monthly_returns.append(monthly_ret * 100)

        # Reshape into approximate year x month grid
        n_years = max(1, n_months // 12)
        n_cols = 12
        n_rows = (n_months + n_cols - 1) // n_cols
        padded = monthly_returns + [0] * (n_rows * n_cols - len(monthly_returns))
        grid = np.array(padded).reshape(n_rows, n_cols)

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        sns.heatmap(grid, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                    xticklabels=month_labels[:n_cols], yticklabels=[f"Y{i+1}" for i in range(n_rows)],
                    ax=ax, cbar_kws={"label": "Return (%)"})

        ax.set_title("Monthly Returns Heatmap (%)", fontsize=16, color=colors["text"], fontweight="bold")
        ax.tick_params(colors=colors["text"])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "monthly_returns_heatmap.png"), dpi=150, facecolor=colors["bg"])
    plt.close()

    # ─── 6. Metrics Summary Card ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    ax.axis("off")

    title = "📊 DEEP RL PORTFOLIO — PERFORMANCE REPORT"
    ax.text(0.5, 0.95, title, fontsize=20, fontweight="bold", color=colors["portfolio"],
            ha="center", va="top", transform=ax.transAxes)

    lines = [
        ("", "RL PORTFOLIO", "S&P 500 (Benchmark)"),
        ("─" * 50, "─" * 20, "─" * 20),
        ("Annualized Return", f"{metrics['annualized_return']*100:.2f}%", f"{metrics['benchmark_annualized_return']*100:.2f}%"),
        ("Total Return", f"{metrics['total_return']*100:.2f}%", f"{metrics['benchmark_total_return']*100:.2f}%"),
        ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", f"{metrics['benchmark_sharpe']:.2f}"),
        ("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}", "-"),
        ("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%", f"{metrics['benchmark_max_drawdown']*100:.2f}%"),
        ("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}", "-"),
        ("Win Rate", f"{metrics['win_rate']*100:.1f}%", "-"),
        ("Ann. Volatility", f"{metrics['annualized_volatility']*100:.2f}%", "-"),
        ("Alpha", f"{metrics['alpha']*100:.2f}%", "-"),
        ("Beta", f"{metrics['beta']:.2f}", "-"),
        ("Trading Days", f"{metrics['n_trading_days']}", f"{metrics['n_trading_days']}"),
    ]

    y = 0.85
    for label, port_val, bench_val in lines:
        ax.text(0.05, y, label, fontsize=13, color=colors["text"],
                transform=ax.transAxes, fontfamily="monospace")
        ax.text(0.55, y, str(port_val), fontsize=13, color=colors["portfolio"],
                transform=ax.transAxes, fontfamily="monospace", fontweight="bold")
        ax.text(0.80, y, str(bench_val), fontsize=13, color=colors["benchmark"],
                transform=ax.transAxes, fontfamily="monospace")
        y -= 0.065

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_summary.png"), dpi=150, facecolor=colors["bg"])
    plt.close()

    print(f"[Backtest] Saved all plots to {save_dir}/")


def save_metrics(metrics: dict, save_dir: str):
    """Save metrics as JSON."""
    # Convert numpy types
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        elif isinstance(v, (list, np.ndarray)):
            continue  # skip arrays
        else:
            clean[k] = v

    path = os.path.join(save_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[Backtest] Saved metrics to {path}")


def print_results(metrics: dict):
    """Print a formatted results table to console."""
    print("\n" + "=" * 72)
    print("  📊 DEEP RL PORTFOLIO OPTIMIZATION — BACKTEST RESULTS")
    print("=" * 72)
    print(f"  {'Metric':<30} {'RL Portfolio':>18} {'S&P 500':>18}")
    print(f"  {'─'*30} {'─'*18} {'─'*18}")
    print(f"  {'Annualized Return':<30} {metrics['annualized_return']*100:>17.2f}% {metrics['benchmark_annualized_return']*100:>17.2f}%")
    print(f"  {'Total Return':<30} {metrics['total_return']*100:>17.2f}% {metrics['benchmark_total_return']*100:>17.2f}%")
    print(f"  {'Sharpe Ratio':<30} {metrics['sharpe_ratio']:>18.2f} {metrics['benchmark_sharpe']:>18.2f}")
    print(f"  {'Sortino Ratio':<30} {metrics['sortino_ratio']:>18.2f} {'—':>18}")
    print(f"  {'Max Drawdown':<30} {metrics['max_drawdown']*100:>17.2f}% {metrics['benchmark_max_drawdown']*100:>17.2f}%")
    print(f"  {'Calmar Ratio':<30} {metrics['calmar_ratio']:>18.2f} {'—':>18}")
    print(f"  {'Win Rate':<30} {metrics['win_rate']*100:>17.1f}% {'—':>18}")
    print(f"  {'Ann. Volatility':<30} {metrics['annualized_volatility']*100:>17.2f}% {'—':>18}")
    print(f"  {'Alpha':<30} {metrics['alpha']*100:>17.2f}% {'—':>18}")
    print(f"  {'Beta':<30} {metrics['beta']:>18.2f} {'—':>18}")
    print(f"  {'Trading Days':<30} {metrics['n_trading_days']:>18} {metrics['n_trading_days']:>18}")
    print("=" * 72)
    print(f"  Transaction Cost: {config.TRANSACTION_COST*100:.2f}%  |  "
          f"Max Position: {config.MAX_POSITION_SIZE*100:.0f}%  |  "
          f"Assets: {config.N_ASSETS}")
    print("=" * 72 + "\n")
