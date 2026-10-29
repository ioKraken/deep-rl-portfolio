"""
Data Pipeline: Download stock data, compute technical features, normalize, and split.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

import config


def _fetch_close(tickers: list, start: str, end: str) -> pd.DataFrame:
    """One yfinance call for a small batch; returns a Close table (possibly empty)."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,  # yfinance's parallel mode is what trips Yahoo's rate limiter
    )
    if data is None or data.empty:
        return pd.DataFrame()
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(axis=1, how="all")


CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _fetch_chart_api(ticker: str, start: str, end: str) -> pd.Series:
    """
    Fetch split- and dividend-adjusted closes straight from Yahoo's chart endpoint.
    Used when yfinance itself fails; returns an empty Series if the request does.
    """
    period = lambda d: int(pd.Timestamp(d, tz="UTC").timestamp())
    try:
        resp = requests.get(
            CHART_URL.format(ticker=ticker),
            params={"period1": period(start), "period2": period(end),
                    "interval": "1d", "events": "div,splits"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        stamps = result["timestamp"]
        adj = result["indicators"]["adjclose"][0]["adjclose"]
        offset = result["meta"].get("gmtoffset", 0)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return pd.Series(dtype=float)

    index = pd.to_datetime(np.array(stamps) + offset, unit="s").normalize()
    series = pd.Series(adj, index=index, dtype=float, name=ticker)
    return series[~series.index.duplicated(keep="last")].dropna()


def download_data(start: str = None, end: str = None, use_cache: bool = False) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers + benchmark.

    Tickers are fetched in small sequential batches and the ones that come back empty
    are retried with growing waits, because Yahoo throttles bulk requests by returning
    "possibly delisted" for tickers that exist. With use_cache the cleaned table is
    stored under data/ and reused on the next run.
    """
    start = start or config.START_DATE
    end = end or config.END_DATE
    required = config.TICKERS + [config.BENCHMARK_TICKER]

    cache_path = os.path.join(config.DATA_DIR, f"prices_{start}_{end}.csv")
    if use_cache and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if all(t in cached.columns for t in required):
            print(f"[Data] Loaded cached prices from {cache_path}")
            return clean_prices(cached)

    print(f"[Data] Downloading {len(required)} tickers from {start} to {end} ...")
    frames = {}
    for attempt in range(1, config.DOWNLOAD_RETRIES + 1):
        remaining = [t for t in required if t not in frames]
        if not remaining:
            break
        if attempt > 1:
            wait = config.DOWNLOAD_RETRY_WAIT * (attempt - 1)
            print(f"[Data] Retry {attempt - 1}: {len(remaining)} tickers missing, waiting {wait:.0f}s")
            time.sleep(wait)
        for i in range(0, len(remaining), config.DOWNLOAD_BATCH_SIZE):
            batch = remaining[i : i + config.DOWNLOAD_BATCH_SIZE]
            close = _fetch_close(batch, start, end)
            for t in close.columns:
                frames[t] = close[t]
            for t in batch:
                if t not in frames:
                    series = _fetch_chart_api(t, start, end)
                    if not series.empty:
                        frames[t] = series

    if not frames:
        raise RuntimeError("yfinance returned no data (network problem or rate limit?)")
    prices = pd.DataFrame(frames)
    cleaned = clean_prices(prices)

    if use_cache:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        cleaned.to_csv(cache_path)
    return cleaned


def load_prices_csv(path: str) -> pd.DataFrame:
    """
    Load prices from a CSV: first column is the date, one column per ticker
    (all of config.TICKERS plus config.BENCHMARK_TICKER), adjusted close prices.
    Rows outside START_DATE..END_DATE are dropped.
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    prices = prices.loc[config.START_DATE : config.END_DATE]
    print(f"[Data] Loaded prices from {path}")
    return clean_prices(prices)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a raw price table. A ticker with no data at all must be an error:
    left in, it would make dropna() silently wipe out every row.
    """
    required = config.TICKERS + [config.BENCHMARK_TICKER]
    missing = [t for t in required if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise RuntimeError(f"No price data returned for: {', '.join(missing)}")

    prices = prices[required].ffill().dropna()
    if prices.empty:
        raise RuntimeError("No overlapping trading days across all tickers")
    print(f"[Data] Downloaded {len(prices)} trading days, {prices.shape[1]} tickers")
    return prices


def compute_features(prices: pd.DataFrame) -> tuple:
    """
    For each asset compute features:
      - Log returns
      - Rolling volatility
      - RSI
      - MACD histogram
      - Bollinger %B
    Returns (features, n_features) where features has shape (n_days, n_assets, n_features)
    """
    asset_tickers = config.TICKERS
    n_features = config.N_FEATURES
    n_days = len(prices)
    n_assets = len(asset_tickers)

    features = np.zeros((n_days, n_assets, n_features))

    for i, ticker in enumerate(asset_tickers):
        if ticker not in prices.columns:
            raise KeyError(f"{ticker} missing from price data")

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
    """
    Rolling z-score normalization along the time axis. Statistics for day t come from
    the `lookback` days before it, so no future data leaks in. The first `lookback` rows
    are left at zero and should be discarded.
    """
    n_days, n_assets, n_feat = features.shape
    normalized = np.zeros_like(features)

    for t in range(lookback, n_days):
        window = features[t - lookback : t]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices during indicator warm-up
            mean = np.nanmean(window, axis=0)
            std = np.nanstd(window, axis=0) + 1e-8
        normalized[t] = (features[t] - mean) / std

    # Replace any remaining NaN/inf, then bound outliers from near-zero-variance windows
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(normalized, -config.FEATURE_CLIP, config.FEATURE_CLIP)


def prepare_data(prices_csv: str = None):
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
    prices_df = load_prices_csv(prices_csv) if prices_csv else download_data(use_cache=True)

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
