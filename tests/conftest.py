import os
import sys

import numpy as np
import pytest

os.environ.setdefault("RL_DEVICE", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


@pytest.fixture
def synthetic():
    """Random-walk prices and standard-normal features: (features, prices, benchmark)."""
    rng = np.random.default_rng(0)
    n = 320
    features = rng.normal(size=(n, config.N_ASSETS, config.N_FEATURES))
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, (n, config.N_ASSETS)), axis=0)
    bench = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return features, prices, bench


@pytest.fixture
def tiny_config(monkeypatch, tmp_path):
    """Shrink training so a full train() call takes seconds, and redirect outputs."""
    monkeypatch.setattr(config, "NUM_TRAINING_ITERATIONS", 3)
    monkeypatch.setattr(config, "TRAJECTORY_LENGTH", 40)
    monkeypatch.setattr(config, "LOOKBACK_WINDOW", 10)
    monkeypatch.setattr(config, "LSTM_HIDDEN_SIZE", 16)
    monkeypatch.setattr(config, "PPO_EPOCHS", 2)
    monkeypatch.setattr(config, "VAL_EVAL_EVERY", 1)
    monkeypatch.setattr(config, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
    return tmp_path
