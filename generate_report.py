"""
Generate a standalone HTML report with all results and observations.
Embeds all chart images as base64 for a single-file, no-dependencies report.
"""

import base64
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from networks import ActorCritic


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_report():
    results_dir = config.RESULTS_DIR
    metrics_path = os.path.join(results_dir, "metrics.json")

    with open(metrics_path) as f:
        m = json.load(f)

    # Load all images as base64
    images = {}
    for name in [
        "performance_summary", "cumulative_returns", "drawdown",
        "rolling_sharpe", "weight_allocation", "monthly_returns_heatmap",
        "training_progress",
    ]:
        path = os.path.join(results_dir, f"{name}.png")
        if os.path.exists(path):
            images[name] = img_to_base64(path)

    # Wording that depends on the results is derived from the numbers, never hardcoded.
    ret_gap = m["annualized_return"] - m["benchmark_annualized_return"]
    sharpe_gap = m["sharpe_ratio"] - m["benchmark_sharpe"]
    dd_better = m["max_drawdown"] > m["benchmark_max_drawdown"]
    ew_gap = m["sharpe_ratio"] - m.get("equal_weight_sharpe_ratio", m["sharpe_ratio"])
    ew_word = "above" if ew_gap > 0 else "below"
    ret_word = "above" if ret_gap > 0 else "below"
    sharpe_word = "above" if sharpe_gap > 0 else "below"
    dd_word = "shallower" if dd_better else "deeper"
    alpha_word = "positive" if m["alpha"] > 0 else "negative"
    n_params = sum(p.numel() for p in ActorCritic(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS).parameters())
    turnover_line = (
        f"Average daily turnover was {m['avg_daily_turnover']*100:.1f}% of the portfolio "
        f"at a {config.TRANSACTION_COST*100:.2f}% cost per unit of turnover."
        if "avg_daily_turnover" in m
        else f"A {config.TRANSACTION_COST*100:.2f}% cost is charged per unit of turnover."
    )
    generated = datetime.date.today().strftime("%B %Y")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep RL Portfolio Optimization — Results Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0a1a;
            --bg-secondary: #121228;
            --bg-card: #1a1a35;
            --bg-card-hover: #222250;
            --accent: #00D4AA;
            --accent-dim: #00a888;
            --accent-glow: rgba(0, 212, 170, 0.15);
            --red: #FF6B6B;
            --red-dim: #cc5555;
            --gold: #FFD700;
            --text-primary: #e8e8f0;
            --text-secondary: #9999bb;
            --text-muted: #666688;
            --border: #2a2a55;
            --gradient-1: linear-gradient(135deg, #00D4AA 0%, #00B4D8 100%);
            --gradient-2: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
            --gradient-hero: linear-gradient(135deg, #0a0a1a 0%, #1a1040 50%, #0a0a1a 100%);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
        }}

        /* ─── Animated Background ─────────────────────────── */
        .bg-grid {{
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-image:
                linear-gradient(rgba(0,212,170,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,212,170,0.03) 1px, transparent 1px);
            background-size: 60px 60px;
            z-index: 0;
            pointer-events: none;
        }}

        .content {{
            position: relative;
            z-index: 1;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 24px;
        }}

        /* ─── Hero Section ────────────────────────────────── */
        .hero {{
            text-align: center;
            padding: 80px 0 60px;
            position: relative;
        }}

        .hero::before {{
            content: '';
            position: absolute;
            top: 0; left: 50%;
            transform: translateX(-50%);
            width: 600px; height: 600px;
            background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
            z-index: -1;
        }}

        .hero .badge {{
            display: inline-block;
            padding: 6px 18px;
            background: var(--accent-glow);
            border: 1px solid var(--accent-dim);
            border-radius: 100px;
            font-size: 12px;
            font-weight: 600;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 24px;
        }}

        .hero h1 {{
            font-size: 3.2rem;
            font-weight: 900;
            letter-spacing: -1px;
            line-height: 1.15;
            margin-bottom: 16px;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .hero .subtitle {{
            font-size: 1.15rem;
            color: var(--text-secondary);
            max-width: 700px;
            margin: 0 auto 32px;
        }}

        .hero .tech-stack {{
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
        }}

        .hero .tech-tag {{
            padding: 6px 14px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        /* ─── KPI Row ─────────────────────────────────────── */
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 48px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .kpi-card:hover {{
            transform: translateY(-4px);
            border-color: var(--accent-dim);
            box-shadow: 0 8px 32px var(--accent-glow);
        }}

        .kpi-card .label {{
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
        }}

        .kpi-card .value {{
            font-size: 2rem;
            font-weight: 800;
            color: var(--accent);
        }}

        .kpi-card .value.red {{
            color: var(--red);
        }}

        .kpi-card .value.gold {{
            color: var(--gold);
        }}

        .kpi-card .sub {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        /* ─── Section Headers ─────────────────────────────── */
        .section {{
            margin-bottom: 56px;
        }}

        .section-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }}

        .section-header .icon {{
            width: 40px; height: 40px;
            background: var(--accent-glow);
            border: 1px solid var(--accent-dim);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }}

        .section-header h2 {{
            font-size: 1.5rem;
            font-weight: 700;
        }}

        /* ─── Comparison Table ─────────────────────────────── */
        .table-wrapper {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        thead th {{
            background: var(--bg-secondary);
            padding: 14px 20px;
            text-align: left;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
        }}

        thead th:nth-child(2) {{ text-align: right; color: var(--accent); }}
        thead th:nth-child(3) {{ text-align: right; color: var(--red); }}

        tbody td {{
            padding: 14px 20px;
            border-bottom: 1px solid rgba(42,42,85,0.5);
            font-size: 14px;
        }}

        tbody td:first-child {{
            font-weight: 600;
            color: var(--text-secondary);
        }}

        tbody td:nth-child(2) {{
            text-align: right;
            font-weight: 700;
            color: var(--accent);
            font-variant-numeric: tabular-nums;
        }}

        tbody td:nth-child(3) {{
            text-align: right;
            font-weight: 500;
            color: var(--text-muted);
            font-variant-numeric: tabular-nums;
        }}

        tbody tr:hover {{
            background: var(--bg-card-hover);
        }}

        /* ─── Chart Cards ─────────────────────────────────── */
        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 24px;
            transition: all 0.3s ease;
        }}

        .chart-card:hover {{
            border-color: var(--accent-dim);
            box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        }}

        .chart-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .chart-card .caption {{
            padding: 16px 20px;
            font-size: 13px;
            color: var(--text-secondary);
            border-top: 1px solid var(--border);
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}

        /* ─── Observation Cards ────────────────────────────── */
        .obs-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .obs-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            border-left: 4px solid var(--accent);
            transition: all 0.3s ease;
        }}

        .obs-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }}

        .obs-card.warn {{
            border-left-color: var(--gold);
        }}

        .obs-card.red {{
            border-left-color: var(--red);
        }}

        .obs-card h4 {{
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 8px;
            color: var(--text-primary);
        }}

        .obs-card p {{
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.7;
        }}

        /* ─── Architecture Block ──────────────────────────── */
        .arch-block {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            font-family: 'Inter', monospace;
        }}

        .arch-flow {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            padding: 16px 0;
        }}

        .arch-node {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
            text-align: center;
            min-width: 140px;
        }}

        .arch-node .node-title {{
            font-size: 12px;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 4px;
        }}

        .arch-node .node-desc {{
            font-size: 11px;
            color: var(--text-muted);
        }}

        .arch-arrow {{
            font-size: 24px;
            color: var(--accent-dim);
        }}

        /* ─── Config Block ────────────────────────────────── */
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
        }}

        .config-item {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .config-item .key {{
            font-size: 13px;
            color: var(--text-secondary);
        }}

        .config-item .val {{
            font-size: 14px;
            font-weight: 700;
            color: var(--accent);
            font-variant-numeric: tabular-nums;
        }}

        /* ─── Footer ──────────────────────────────────────── */
        .footer {{
            text-align: center;
            padding: 48px 0;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 13px;
        }}

        @media (max-width: 768px) {{
            .hero h1 {{ font-size: 2rem; }}
            .chart-grid, .obs-grid {{ grid-template-columns: 1fr; }}
            .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="bg-grid"></div>
    <div class="content">

        <!-- ═══ HERO ═══ -->
        <div class="hero">
            <div class="badge">Deep Reinforcement Learning</div>
            <h1>Portfolio Optimization<br>Results Report</h1>
            <p class="subtitle">
                LSTM-based PPO framework optimizing allocation across 50 equities
                with differentiable Sharpe Ratio objective, {config.TRANSACTION_COST*100:.2f}% transaction costs,
                and position constraints — evaluated on {m['n_trading_days']} held-out trading days.
            </p>
            <div class="tech-stack">
                <span class="tech-tag">Python</span>
                <span class="tech-tag">PyTorch</span>
                <span class="tech-tag">PPO</span>
                <span class="tech-tag">LSTM</span>
                <span class="tech-tag">Pandas</span>
                <span class="tech-tag">NumPy</span>
            </div>
        </div>

        <!-- ═══ KPI ROW ═══ -->
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="label">Annualized Return</div>
                <div class="value">{m['annualized_return']*100:.1f}%</div>
                <div class="sub">vs {m['benchmark_annualized_return']*100:.1f}% S&P 500</div>
            </div>
            <div class="kpi-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value gold">{m['sharpe_ratio']:.2f}</div>
                <div class="sub">vs {m['benchmark_sharpe']:.2f} benchmark</div>
            </div>
            <div class="kpi-card">
                <div class="label">Sortino Ratio</div>
                <div class="value">{m['sortino_ratio']:.2f}</div>
                <div class="sub">Downside risk-adjusted</div>
            </div>
            <div class="kpi-card">
                <div class="label">Max Drawdown</div>
                <div class="value red">{m['max_drawdown']*100:.1f}%</div>
                <div class="sub">vs {m['benchmark_max_drawdown']*100:.1f}% S&P 500</div>
            </div>
            <div class="kpi-card">
                <div class="label">Calmar Ratio</div>
                <div class="value">{m['calmar_ratio']:.2f}</div>
                <div class="sub">Return / Max DD</div>
            </div>
            <div class="kpi-card">
                <div class="label">Win Rate</div>
                <div class="value gold">{m['win_rate']*100:.1f}%</div>
                <div class="sub">{m['n_trading_days']} trading days</div>
            </div>
        </div>

        <!-- ═══ PERFORMANCE TABLE ═══ -->
        <div class="section">
            <div class="section-header">
                <div class="icon">📊</div>
                <h2>Performance Comparison</h2>
            </div>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>RL Portfolio</th>
                            <th>S&P 500 (Benchmark)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Annualized Return</td><td>{m['annualized_return']*100:.2f}%</td><td>{m['benchmark_annualized_return']*100:.2f}%</td></tr>
                        <tr><td>Total Return</td><td>{m['total_return']*100:.2f}%</td><td>{m['benchmark_total_return']*100:.2f}%</td></tr>
                        <tr><td>Sharpe Ratio</td><td>{m['sharpe_ratio']:.2f}</td><td>{m['benchmark_sharpe']:.2f}</td></tr>
                        <tr><td>Equal-weight baseline: Ann. Return / Sharpe</td><td colspan="2">{m.get('equal_weight_annualized_return', float('nan'))*100:.2f}% / {m.get('equal_weight_sharpe_ratio', float('nan')):.2f}</td></tr>
                        <tr><td>Sortino Ratio</td><td>{m['sortino_ratio']:.2f}</td><td>—</td></tr>
                        <tr><td>Max Drawdown</td><td>{m['max_drawdown']*100:.2f}%</td><td>{m['benchmark_max_drawdown']*100:.2f}%</td></tr>
                        <tr><td>Calmar Ratio</td><td>{m['calmar_ratio']:.2f}</td><td>—</td></tr>
                        <tr><td>Win Rate</td><td>{m['win_rate']*100:.1f}%</td><td>—</td></tr>
                        <tr><td>Annualized Volatility</td><td>{m['annualized_volatility']*100:.2f}%</td><td>—</td></tr>
                        <tr><td>Alpha (vs S&P)</td><td>{m['alpha']*100:.2f}%</td><td>—</td></tr>
                        <tr><td>Beta</td><td>{m['beta']:.2f}</td><td>1.00</td></tr>
                        <tr><td>Trading Days</td><td>{m['n_trading_days']}</td><td>{m['n_trading_days']}</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- ═══ CHARTS ═══ -->
        <div class="section">
            <div class="section-header">
                <div class="icon">📈</div>
                <h2>Cumulative Returns</h2>
            </div>
            <div class="chart-card">
                <img src="data:image/png;base64,{images.get('cumulative_returns', '')}" alt="Cumulative Returns">
                <div class="caption">
                    Over the out-of-sample period the RL portfolio returned {m['annualized_return']*100:.1f}% annualized
                    ({abs(ret_gap)*100:.1f} pts {ret_word} the S&P 500) with a Sharpe ratio of {m['sharpe_ratio']:.2f}
                    ({abs(sharpe_gap):.2f} {sharpe_word} the benchmark) and a beta of {m['beta']:.2f}.
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <div class="icon">📉</div>
                <h2>Drawdown & Risk Analysis</h2>
            </div>
            <div class="chart-grid">
                <div class="chart-card">
                    <img src="data:image/png;base64,{images.get('drawdown', '')}" alt="Drawdown">
                    <div class="caption">
                        Maximum drawdown of {m['max_drawdown']*100:.1f}% vs benchmark's {m['benchmark_max_drawdown']*100:.1f}%.
                        The agent's maximum drawdown was {dd_word} than the benchmark's.
                    </div>
                </div>
                <div class="chart-card">
                    <img src="data:image/png;base64,{images.get('rolling_sharpe', '')}" alt="Rolling Sharpe">
                    <div class="caption">
                        Rolling 60-day Sharpe ratio of the portfolio against the benchmark.
                        Short windows are noisy; read the trend, not individual peaks.
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <div class="icon">⚖️</div>
                <h2>Portfolio Allocation</h2>
            </div>
            <div class="chart-card">
                <img src="data:image/png;base64,{images.get('weight_allocation', '')}" alt="Weight Allocation">
                <div class="caption">
                    Top 15 holdings by average weight. The RL agent dynamically adjusts allocations
                    based on LSTM-processed market signals, subject to the {config.MAX_POSITION_SIZE*100:.0f}% max position constraint.
                    Measured beta versus the S&P 500 was {m['beta']:.2f}.
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <div class="icon">📅</div>
                <h2>Monthly Returns Heatmap</h2>
            </div>
            <div class="chart-card">
                <img src="data:image/png;base64,{images.get('monthly_returns_heatmap', '')}" alt="Monthly Returns">
                <div class="caption">
                    Returns per consecutive 21-trading-day period (not calendar months).
                    Daily win rate was {m['win_rate']*100:.1f}% and the Sortino ratio {m['sortino_ratio']:.2f}.
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <div class="icon">🧠</div>
                <h2>Training Progress</h2>
            </div>
            <div class="chart-card">
                <img src="data:image/png;base64,{images.get('training_progress', '')}" alt="Training Progress">
                <div class="caption">
                    PPO training over {config.NUM_TRAINING_ITERATIONS} iterations on the differential Sharpe
                    reward. The checkpoint used for the backtest is the one with the highest Sharpe
                    on a validation slice held out from the end of the training period.
                </div>
            </div>
        </div>

        <!-- ═══ OBSERVATIONS ═══ -->
        <div class="section">
            <div class="section-header">
                <div class="icon">🔍</div>
                <h2>Key Observations</h2>
            </div>
            <div class="obs-grid">
                <div class="obs-card">
                    <h4>🎯 Risk-Adjusted Returns</h4>
                    <p>
                        The RL portfolio has a <strong>Sharpe ratio of {m['sharpe_ratio']:.2f}</strong>
                        ({sharpe_word} the benchmark's {m['benchmark_sharpe']:.2f}) and a
                        <strong>Sortino ratio of {m['sortino_ratio']:.2f}</strong> on a single
                        out-of-sample period; an equal-weight portfolio of the same stocks scored
                        {m.get('equal_weight_sharpe_ratio', float('nan')):.2f} (the agent is {abs(ew_gap):.2f} {ew_word} it). One period and one training seed are not enough to
                        claim a durable edge.
                    </p>
                </div>
                <div class="obs-card">
                    <h4>📊 Market Exposure</h4>
                    <p>
                        With a <strong>Beta of {m['beta']:.2f}</strong>, the portfolio moved roughly {m['beta']*100:.0f}% as much as
                        the market, at {m['annualized_volatility']*100:.1f}% annualized volatility.
                        The {config.MAX_POSITION_SIZE*100:.0f}% position cap and {config.N_ASSETS}-asset universe bound concentration.
                    </p>
                </div>
                <div class="obs-card warn">
                    <h4>⚡ Transaction Cost Impact</h4>
                    <p>
                        {turnover_line} Costs are deducted from every simulated return.
                    </p>
                </div>
                <div class="obs-card">
                    <h4>🔄 Moody-Saffell Differential Sharpe</h4>
                    <p>
                        The online differential Sharpe ratio reward (Moody & Saffell, 1998) provides
                        a temporally smooth learning signal via exponential moving averages. This
                        avoids the noisy per-step return signals that destabilize RL training,
                        enabling stable convergence with bounded rewards.
                    </p>
                </div>
                <div class="obs-card">
                    <h4>🏗️ LSTM Temporal Modeling</h4>
                    <p>
                        The {config.LSTM_NUM_LAYERS}-layer LSTM with {config.LSTM_HIDDEN_SIZE} hidden units processes {config.LOOKBACK_WINDOW}-day lookback windows
                        of 5 features per asset (log returns, volatility, RSI, MACD, Bollinger %B),
                        and feeds the encoded history to the actor and critic heads.
                    </p>
                </div>
                <div class="obs-card warn">
                    <h4>📈 Alpha</h4>
                    <p>
                        Measured <strong>alpha is {alpha_word} at {m['alpha']*100:.2f}%</strong> annualized versus the S&P 500.
                        No significance test was run, so a value this close to zero should not be
                        read as evidence of skill.
                    </p>
                </div>
                <div class="obs-card red">
                    <h4>⚠️ Concentration Constraints</h4>
                    <p>
                        The 10% max position size and long-only constraints limit how far the agent
                        can overweight any single name. This is a deliberate risk-management choice
                        that can cost return in concentrated rallies.
                    </p>
                </div>
                <div class="obs-card">
                    <h4>🎲 Gaussian Policy with Softmax</h4>
                    <p>
                        Rather than Dirichlet distributions (which lack MPS support), the actor uses
                        Gaussian distributions in unconstrained logit space followed by softmax
                        normalization. This MPS-compatible approach produces valid portfolio weight
                        simplices while enabling efficient GPU-accelerated training on Apple Silicon.
                    </p>
                </div>
            </div>
        </div>

        <!-- ═══ MODEL ARCHITECTURE ═══ -->
        <div class="section">
            <div class="section-header">
                <div class="icon">🧬</div>
                <h2>Model Architecture</h2>
            </div>
            <div class="arch-block">
                <div class="arch-flow">
                    <div class="arch-node">
                        <div class="node-title">Market Features</div>
                        <div class="node-desc">{config.N_ASSETS} assets × 5 features<br>{config.LOOKBACK_WINDOW}-day lookback</div>
                    </div>
                    <div class="arch-arrow">→</div>
                    <div class="arch-node">
                        <div class="node-title">LSTM Encoder</div>
                        <div class="node-desc">{config.LSTM_NUM_LAYERS} layers, {config.LSTM_HIDDEN_SIZE} hidden<br>LayerNorm</div>
                    </div>
                    <div class="arch-arrow">→</div>
                    <div class="arch-node">
                        <div class="node-title">Actor Head</div>
                        <div class="node-desc">Gaussian → Softmax<br>Portfolio Weights</div>
                    </div>
                    <div class="arch-arrow">→</div>
                    <div class="arch-node">
                        <div class="node-title">Portfolio Env</div>
                        <div class="node-desc">Sharpe Reward<br>{config.TRANSACTION_COST*100:.2f}% Costs</div>
                    </div>
                    <div class="arch-arrow">→</div>
                    <div class="arch-node">
                        <div class="node-title">PPO Update</div>
                        <div class="node-desc">Clipped Objective<br>GAE λ={config.GAE_LAMBDA}</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ═══ CONFIGURATION ═══ -->
        <div class="section">
            <div class="section-header">
                <div class="icon">⚙️</div>
                <h2>Training Configuration</h2>
            </div>
            <div class="config-grid">
                <div class="config-item"><span class="key">Assets Universe</span><span class="val">{config.N_ASSETS} equities</span></div>
                <div class="config-item"><span class="key">Data Period</span><span class="val">{config.START_DATE[:4]}–{config.END_DATE[:4]}</span></div>
                <div class="config-item"><span class="key">Train / Test Split</span><span class="val">{config.TRAIN_RATIO:.0%} / {1-config.TRAIN_RATIO:.0%}</span></div>
                <div class="config-item"><span class="key">Lookback Window</span><span class="val">{config.LOOKBACK_WINDOW} days</span></div>
                <div class="config-item"><span class="key">LSTM Hidden Size</span><span class="val">{config.LSTM_HIDDEN_SIZE}</span></div>
                <div class="config-item"><span class="key">LSTM Layers</span><span class="val">{config.LSTM_NUM_LAYERS}</span></div>
                <div class="config-item"><span class="key">Learning Rate</span><span class="val">{config.LEARNING_RATE:g}</span></div>
                <div class="config-item"><span class="key">PPO Clip</span><span class="val">{config.PPO_CLIP}</span></div>
                <div class="config-item"><span class="key">Gamma (Discount)</span><span class="val">{config.GAMMA}</span></div>
                <div class="config-item"><span class="key">GAE Lambda</span><span class="val">{config.GAE_LAMBDA}</span></div>
                <div class="config-item"><span class="key">Transaction Cost</span><span class="val">{config.TRANSACTION_COST*100:.2f}%</span></div>
                <div class="config-item"><span class="key">Max Position Size</span><span class="val">{config.MAX_POSITION_SIZE*100:.0f}%</span></div>
                <div class="config-item"><span class="key">Training Iterations</span><span class="val">{config.NUM_TRAINING_ITERATIONS}</span></div>
                <div class="config-item"><span class="key">Trajectory Length</span><span class="val">{config.TRAJECTORY_LENGTH} days</span></div>
                <div class="config-item"><span class="key">Model Parameters</span><span class="val">{n_params:,}</span></div>
                <div class="config-item"><span class="key">Entropy Coefficient</span><span class="val">{config.ENTROPY_COEF}</span></div>
            </div>
        </div>

        <!-- ═══ FOOTER ═══ -->
        <div class="footer">
            <p>Deep RL Portfolio Optimization &nbsp;•&nbsp; LSTM-PPO &nbsp;•&nbsp; Generated {generated}</p>
            <p style="margin-top: 8px; color: var(--text-muted);">
                Built with Python, PyTorch, Pandas, NumPy &nbsp;|&nbsp; {m['n_trading_days']} out-of-sample trading days
            </p>
        </div>

    </div>
</body>
</html>"""

    output_path = os.path.join(config.PROJECT_DIR, "results_report.html")
    with open(output_path, "w") as f:
        f.write(html)

    print(f"✅ HTML report generated: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.0f} KB")
    return output_path


if __name__ == "__main__":
    generate_report()
