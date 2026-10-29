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
    monkeypatch.setattr(dp, "download_data", lambda: _prices(500))
    tf, sf, tp, sp, bt, bs, df = dp.prepare_data()
    assert len(tf) == len(tp) == len(bt)
    assert len(sf) == len(sp) == len(bs)
    assert len(tf) + len(sf) == 500 - config.LOOKBACK_WINDOW
    assert tp.shape[1] == config.N_ASSETS
    # train precedes test chronologically: test prices continue the train series
    full = df[config.TICKERS].values[config.LOOKBACK_WINDOW:]
    assert np.array_equal(np.vstack([tp, sp]), full)
