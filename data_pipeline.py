"""
Data Pipeline: Download stock data, compute technical features, normalize, and split.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

import config


def download_data() -> pd.DataFrame:
    """Download adjusted close prices for all tickers + benchmark."""
    all_tickers = config.TICKERS + [config.BENCHMARK_TICKER]
    print(f"[Data] Downloading {len(all_tickers)} tickers from {config.START_DATE} to {config.END_DATE} ...")
    data = yf.download(
        all_tickers,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=True,
        progress=False,
    )
    # yfinance returns MultiIndex columns: (Price, Ticker)
    prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    prices = prices.ffill().dropna()
    print(f"[Data] Downloaded {len(prices)} trading days, {prices.shape[1]} tickers")
    return prices


def compute_features(prices: pd.DataFrame) -> np.ndarray:
    """
    For each asset compute features:
      - Log returns
      - Rolling volatility
      - RSI
      - MACD histogram
      - Bollinger %B
    Returns array of shape (n_days, n_assets, n_features)
    """
    asset_tickers = config.TICKERS
    n_features = 5
    n_days = len(prices)
    n_assets = len(asset_tickers)

    features = np.zeros((n_days, n_assets, n_features))

    for i, ticker in enumerate(asset_tickers):
        if ticker not in prices.columns:
            print(f"  [Warning] {ticker} not in data, skipping")
            continue

        close = prices[ticker]

        # 1) Log returns
        log_ret = np.log(close / close.shift(1))

        # 2) Rolling volatility
        roll_vol = log_ret.rolling(config.ROLLING_VOL_WINDOW).std()

        # 3) RSI
        rsi = RSIIndicator(close=close, window=config.RSI_WINDOW).rsi() / 100.0

        # 4) MACD histogram
        macd_ind = MACD(
            close=close,
            window_fast=config.MACD_FAST,
            window_slow=config.MACD_SLOW,
            window_sign=config.MACD_SIGNAL,
        )
        macd_hist = macd_ind.macd_diff()

        # 5) Bollinger %B
        bb = BollingerBands(
            close=close,
            window=config.BOLLINGER_WINDOW,
            window_dev=config.BOLLINGER_STD,
        )
        boll_pctb = bb.bollinger_pband()

        features[:, i, 0] = log_ret.values
        features[:, i, 1] = roll_vol.values
        features[:, i, 2] = rsi.values
        features[:, i, 3] = macd_hist.values
        features[:, i, 4] = boll_pctb.values

    return features, n_features


def normalize_features(features: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling z-score normalization along the time axis."""
    n_days, n_assets, n_feat = features.shape
    normalized = np.zeros_like(features)

    for t in range(lookback, n_days):
        window = features[t - lookback : t]
        mean = np.nanmean(window, axis=0)
        std = np.nanstd(window, axis=0) + 1e-8
        normalized[t] = (features[t] - mean) / std

    # Replace any remaining NaN/inf
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized


def prepare_data():
    """
    Full pipeline: download → features → normalize → split.
    Returns:
        train_features: (n_train_days, n_assets, n_features)
        test_features:  (n_test_days, n_assets, n_features)
        train_prices:   (n_train_days, n_assets)  — raw prices for return calc
        test_prices:    (n_test_days, n_assets)
        benchmark_train: (n_train_days,)
        benchmark_test:  (n_test_days,)
        prices_df: full DataFrame for reference
    """
    prices_df = download_data()

    # Compute features
    features, n_feat = compute_features(prices_df)

    # Normalize
    features = normalize_features(features, config.LOOKBACK_WINDOW)

    # Trim the first LOOKBACK_WINDOW days (incomplete normalization)
    valid_start = config.LOOKBACK_WINDOW
    features = features[valid_start:]
    asset_prices = prices_df[config.TICKERS].values[valid_start:]
    benchmark_prices = prices_df[config.BENCHMARK_TICKER].values[valid_start:]

    # Train / Test split
    n_total = len(features)
    n_train = int(n_total * config.TRAIN_RATIO)

    train_features = features[:n_train]
    test_features = features[n_train:]
    train_prices = asset_prices[:n_train]
    test_prices = asset_prices[n_train:]
    benchmark_train = benchmark_prices[:n_train]
    benchmark_test = benchmark_prices[n_train:]

    print(f"[Data] Features shape per day: ({config.N_ASSETS}, {n_feat})")
    print(f"[Data] Train: {len(train_features)} days | Test: {len(test_features)} days")

    return (
        train_features,
        test_features,
        train_prices,
        test_prices,
        benchmark_train,
        benchmark_test,
        prices_df,
    )


if __name__ == "__main__":
    prepare_data()
