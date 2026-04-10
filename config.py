"""
Configuration for Deep RL Portfolio Optimization.
All hyperparameters, tickers, date ranges, and paths are defined here.
"""

import os

# ─── Project Paths ──────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# ─── Universe: 50 Large-Cap Equities ────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA",
    "DIS", "BAC", "XOM", "ADBE", "CRM",
    "CSCO", "PFE", "CMCSA", "NFLX", "PEP",
    "TMO", "ABT", "COST", "AVGO", "NKE",
    "MRK", "WMT", "INTC", "T", "VZ",
    "CVX", "LLY", "MCD", "DHR", "NEE",
    "TXN", "PM", "LOW", "HON", "QCOM",
    "UPS", "MS", "GS", "AMD", "ISRG",
]

BENCHMARK_TICKER = "SPY"
N_ASSETS = len(TICKERS)

# ─── Date Ranges ────────────────────────────────────────────────────────────
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
TRAIN_RATIO = 0.70  # 70% train, 30% test

# ─── Feature Engineering ─────────────────────────────────────────────────────
LOOKBACK_WINDOW = 60        # Days of history the LSTM sees
ROLLING_VOL_WINDOW = 20     # For rolling volatility
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# ─── Environment ─────────────────────────────────────────────────────────────
TRANSACTION_COST = 0.0025   # 0.25% per unit of turnover
MAX_POSITION_SIZE = 0.10    # Max 10% in any single asset
SHARPE_WINDOW = 60          # Rolling window for Sharpe reward
RISK_FREE_RATE = 0.04       # Annual risk-free rate (≈ recent T-bill)

# ─── Network Architecture ────────────────────────────────────────────────────
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.1

# ─── PPO Hyperparameters ─────────────────────────────────────────────────────
LEARNING_RATE = 1e-4
GAMMA = 0.99                # Discount factor
GAE_LAMBDA = 0.95           # GAE lambda
PPO_CLIP = 0.2              # Clipping parameter
PPO_EPOCHS = 8              # Epochs per update
MINIBATCH_SIZE = 64
ENTROPY_COEF = 0.005        # Entropy bonus coefficient
VALUE_LOSS_COEF = 0.5       # Value loss coefficient
MAX_GRAD_NORM = 0.5         # Gradient clipping
TRAJECTORY_LENGTH = 200     # ~10 months trading days per trajectory

# ─── Training ────────────────────────────────────────────────────────────────
NUM_TRAINING_ITERATIONS = 150  # Number of training iterations
SEED = 42

# ─── Ensure directories exist ────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
