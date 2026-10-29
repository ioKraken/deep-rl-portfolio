import numpy as np
import pandas as pd
import pytest

import config
import data_pipeline as dp


def _prices(n=400, tickers=None, seed=0):
    tickers = tickers or config.TICKERS + [config.BENCHMARK_TICKER]
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, len(tickers))), axis=0),
                        index=idx, columns=tickers)


def test_clean_prices_fails_loudly_on_missing_ticker():
    p = _prices()
    p[config.TICKERS[3]] = np.nan
    with pytest.raises(RuntimeError, match=config.TICKERS[3]):
        dp.clean_prices(p)


def test_clean_prices_fails_on_absent_column():
    p = _prices().drop(columns=[config.TICKERS[0]])
    with pytest.raises(RuntimeError, match=config.TICKERS[0]):
        dp.clean_prices(p)


def test_clean_prices_forward_fills_gaps_and_orders_columns():
    p = _prices()
    p.iloc[50, 4] = np.nan
    out = dp.clean_prices(p.sample(frac=1, axis=1, random_state=1))
    assert len(out) == len(p)
    assert list(out.columns) == config.TICKERS + [config.BENCHMARK_TICKER]
    assert out.iloc[50, 4] == out.iloc[49, 4]


def test_compute_features_shape():
    feats, n = dp.compute_features(_prices())
    assert n == config.N_FEATURES
    assert feats.shape == (400, config.N_ASSETS, config.N_FEATURES)


def test_normalization_has_no_lookahead():
    feats, _ = dp.compute_features(_prices())
    base = dp.normalize_features(feats, config.LOOKBACK_WINDOW)

    altered = feats.copy()
    t = 250
    altered[t:] *= 50  # change the present and future
    out = dp.normalize_features(altered, config.LOOKBACK_WINDOW)
    assert np.array_equal(base[:t], out[:t])


def test_normalization_is_finite_and_bounded():
    feats, _ = dp.compute_features(_prices())
    feats[100:140, 0, 1] = 1.0  # a flat stretch: zero variance
    out = dp.normalize_features(feats, config.LOOKBACK_WINDOW)
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= config.FEATURE_CLIP


def test_prepare_data_split_and_alignment(monkeypatch):
    monkeypatch.setattr(dp, "download_data", lambda **k: _prices(500))
    tf, sf, tp, sp, bt, bs, df = dp.prepare_data()
    assert len(tf) == len(tp) == len(bt)
    assert len(sf) == len(sp) == len(bs)
    assert len(tf) + len(sf) == 500 - config.LOOKBACK_WINDOW
    assert tp.shape[1] == config.N_ASSETS
    # train precedes test chronologically: test prices continue the train series
    full = df[config.TICKERS].values[config.LOOKBACK_WINDOW:]
    assert np.array_equal(np.vstack([tp, sp]), full)


def _fake_yf(prices, fail_first_n_calls=0, calls=None):
    """Stand-in for yf.download that returns nothing for the first N calls (throttling)."""
    state = {"n": 0}

    def fake(tickers, **kwargs):
        state["n"] += 1
        if calls is not None:
            calls.append(list(tickers))
        assert kwargs.get("threads") is False
        if state["n"] <= fail_first_n_calls:
            return pd.DataFrame()
        return pd.concat({"Close": prices[list(tickers)]}, axis=1)

    return fake


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setattr(dp.time, "sleep", lambda s: None)


def test_download_batches_and_assembles_all_tickers(monkeypatch, fast_retries):
    calls = []
    monkeypatch.setattr(dp.yf, "download", _fake_yf(_prices(), calls=calls))
    out = dp.download_data()
    assert list(out.columns) == config.TICKERS + [config.BENCHMARK_TICKER]
    assert max(len(c) for c in calls) <= config.DOWNLOAD_BATCH_SIZE


def test_download_retries_throttled_batches(monkeypatch, fast_retries):
    monkeypatch.setattr(dp.yf, "download", _fake_yf(_prices(), fail_first_n_calls=3))
    out = dp.download_data()
    assert out.shape[1] == config.N_ASSETS + 1


def test_download_gives_up_with_clear_error(monkeypatch, fast_retries):
    monkeypatch.setattr(dp.yf, "download", _fake_yf(_prices(), fail_first_n_calls=10**6))
    with pytest.raises(RuntimeError, match="no data"):
        dp.download_data()


def test_download_cache_roundtrip(monkeypatch, fast_retries, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(dp.yf, "download", _fake_yf(_prices(), calls=calls))
    first = dp.download_data(use_cache=True)
    n_calls = len(calls)
    second = dp.download_data(use_cache=True)
    assert len(calls) == n_calls  # served from cache
    assert np.allclose(first.values, second.values)
