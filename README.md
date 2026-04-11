#  Deep RL Portfolio Optimizer

A Deep Reinforcement Learning agent (PPO) for portfolio optimization — trained to allocate assets dynamically using market data.

##  Project Structure

```
deep_rl_portfolio/
├── run.py               # Entry point
├── train.py             # Training loop
├── ppo.py               # PPO algorithm implementation
├── networks.py          # Neural network architectures
├── environment.py       # Custom trading environment (Gym-style)
├── backtest.py          # Backtesting engine
├── data_pipeline.py     # Data fetching & feature engineering
├── config.py            # Hyperparameters & settings
├── generate_report.py   # Results report generator
├── models/
│   └── best_model.pt    # Saved best model checkpoint
└── results/             # Training plots & performance metrics
```

##  Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the agent
```bash
python run.py
```

### 3. View results
Open `results_report.html` in your browser for a full performance report.

##  Algorithm

- **PPO (Proximal Policy Optimization)** for stable policy gradient training
- Custom **portfolio environment** with realistic trading constraints
- Technical indicators via `ta` library for state representation

##  Results

| Metric | Value |
|--------|-------|
| See `results/metrics.json` for full details | |

Training curves, drawdown charts, Sharpe ratios, and weight allocations are saved under `results/`.

##  Tech Stack

- PyTorch · NumPy · Pandas · yfinance · Matplotlib · Seaborn

##  License

MIT
