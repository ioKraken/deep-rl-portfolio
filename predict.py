"""
Inference entry point: load a trained checkpoint and print target portfolio weights
for the most recent trading day.

    python predict.py                              # equal-weight starting portfolio
    python predict.py --current-weights w.json     # {"AAPL": 0.03, ...}
    python predict.py --output weights.json
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

import config
from data_pipeline import download_data, compute_features, normalize_features
from environment import apply_position_cap
from ppo import PPOAgent
from train import select_device

# Rolling normalization needs LOOKBACK_WINDOW days of history before each of the
# LOOKBACK_WINDOW observation days, plus the longest indicator warm-up.
INDICATOR_WARMUP_DAYS = 40
MIN_HISTORY_DAYS = 2 * config.LOOKBACK_WINDOW + INDICATOR_WARMUP_DAYS


def load_agent(model_path: str) -> PPOAgent:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model checkpoint at {model_path}. Run `python run.py` to train one.")
    agent = PPOAgent(config.N_ASSETS * config.N_FEATURES, config.N_ASSETS, device=select_device())
    agent.load(model_path, load_optimizer=False)
    agent.model.eval()
    return agent


def compute_target_weights(prices: pd.DataFrame, agent: PPOAgent, current_weights: np.ndarray = None) -> dict:
    """Return {"as_of": date, "weights": {ticker: weight}} from a price table ending today."""
    if len(prices) < MIN_HISTORY_DAYS:
        raise ValueError(f"Need at least {MIN_HISTORY_DAYS} trading days of prices, got {len(prices)}")

    features, _ = compute_features(prices)
    features = normalize_features(features, config.LOOKBACK_WINDOW)
    window = features[-config.LOOKBACK_WINDOW:].reshape(config.LOOKBACK_WINDOW, -1)

    if current_weights is None:
        current_weights = np.ones(config.N_ASSETS) / config.N_ASSETS

    raw, _, _, _ = agent.select_action(window, current_weights, deterministic=True)
    weights = apply_position_cap(raw, config.MAX_POSITION_SIZE)

    return {
        "as_of": str(prices.index[-1].date()),
        "weights": {t: float(w) for t, w in zip(config.TICKERS, weights)},
    }


def load_current_weights(path: str) -> np.ndarray:
    with open(path) as f:
        held = json.load(f)
    unknown = set(held) - set(config.TICKERS)
    if unknown:
        raise ValueError(f"Unknown tickers in {path}: {sorted(unknown)}")
    w = np.array([float(held.get(t, 0.0)) for t in config.TICKERS])
    if w.sum() <= 0:
        raise ValueError(f"Weights in {path} sum to zero")
    return w / w.sum()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print target portfolio weights for today")
    parser.add_argument("--model", default=os.path.join(config.MODEL_DIR, "best_model.pt"))
    parser.add_argument("--current-weights", help="JSON file mapping ticker to current weight")
    parser.add_argument("--output", help="Write the result to this JSON file")
    parser.add_argument("--history-days", type=int, default=400,
                        help="Calendar days of price history to download (default: 400)")
    args = parser.parse_args(argv)

    agent = load_agent(args.model)
    current = load_current_weights(args.current_weights) if args.current_weights else None

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=args.history_days)).isoformat()
    prices = download_data(start=start, end=(today + datetime.timedelta(days=1)).isoformat())

    result = compute_target_weights(prices, agent, current)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(f"Target weights as of {result['as_of']}:")
        for ticker, w in sorted(result["weights"].items(), key=lambda kv: -kv[1]):
            print(f"  {ticker:<6} {w * 100:6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
