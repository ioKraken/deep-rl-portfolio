#  Deep RL Portfolio Optimizer

A Deep Reinforcement Learning agent (PPO) for portfolio optimization — trained to allocate assets dynamically using market data.

##  Project Structure

```
deep_rl_portfolio/
├── run.py               # Entry point: data -> train -> backtest -> report
├── predict.py           # Inference: target weights for today from a saved model
├── tests/               # pytest suite (no network needed)
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
python run.py                    # full run (needs internet for yfinance)
python run.py --iterations 20    # quick run
python run.py --prices-csv prices.csv   # use your own prices if Yahoo blocks you
```

### 3. View results
Open `results_report.html` in your browser for a full performance report.

### 4. Get today's target weights
```bash
python predict.py                                # starts from an equal-weight portfolio
python predict.py --current-weights held.json    # {"AAPL": 0.04, ...}
python predict.py --output weights.json
```

### 5. Run the tests
```bash
pip install -r requirements-dev.txt
python -m pytest
```

### Docker
```bash
docker build -t deep-rl-portfolio .
docker run --rm deep-rl-portfolio                         # predict with the bundled model
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/results:/app/results" \
    deep-rl-portfolio run.py                              # retrain
```

Set `RL_DEVICE=cpu|mps|cuda` to force a device. The checkpoint in `models/` is only
valid for the ticker list and feature set in `config.py`; loading it against a changed
config raises an error instead of silently producing wrong weights.

##  Algorithm

- **PPO (Proximal Policy Optimization)** for stable policy gradient training
- Custom **portfolio environment**: long-only, 10% position cap, proportional transaction costs
- Model selection on a validation slice held out from the end of the training period
- Technical indicators via `ta` library for state representation

##  Results

See `results/metrics.json` for the numbers from the latest run. Results come from a
single out-of-sample period and a single seed; treat them as a research prototype, not
trading advice.

Training curves, drawdown charts, Sharpe ratios, and weight allocations are saved under `results/`.

##  Tech Stack

- PyTorch · NumPy · Pandas · yfinance · Matplotlib · Seaborn

##  License

MIT
