# tests/conftest.py
"""
Shared synthetic fixtures — no yfinance, fully deterministic.
"""
import numpy as np
import pandas as pd
import pytest
from cpcv_analysis.splitters import CombinatorialPurgedKFold


def _make_index(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


@pytest.fixture
def small_dataset():
    """
    N=60 obs. Used for splitter property tests.
    X: 2 random features (seed=0)
    y: random binary labels (seed=0)
    t1: index + 5 business days (label horizon)
    """
    rng = np.random.default_rng(0)
    idx = _make_index(60)
    X = pd.DataFrame(rng.standard_normal((60, 2)), index=idx, columns=["f1", "f2"])
    y = pd.Series(rng.integers(0, 2, size=60), index=idx)
    t1 = pd.Series(idx.shift(5, freq="B"), index=idx)
    t1 = t1.clip(upper=idx[-1])  # cap at last date
    return X, y, t1


@pytest.fixture
def embargo_dataset():
    """
    N=100 obs. Used for testing exact embargo step size.
    """
    idx = _make_index(100)
    X = pd.DataFrame(np.zeros((100, 1)), index=idx, columns=["f1"])
    t1 = pd.Series(idx.shift(5, freq="B"), index=idx)
    t1 = t1.clip(upper=idx[-1])
    return X, t1


@pytest.fixture
def leakage_dataset():
    """
    N=200 obs. Used for leakage induction tests.
    X_clean: 3 random features (seed=1)
    X_leaked: X_clean + column 'future_label' = y shifted -1 (perfect future leakage)
    y: random binary labels (seed=1)
    t1: index + 5 business days
    fwd_ret: small random returns (seed=1)
    """
    rng = np.random.default_rng(1)
    idx = _make_index(200)
    X_clean = pd.DataFrame(
        rng.standard_normal((200, 3)), index=idx, columns=["f1", "f2", "f3"]
    )
    y = pd.Series(rng.integers(0, 2, size=200), index=idx)
    t1 = pd.Series(idx.shift(5, freq="B"), index=idx)
    t1 = t1.clip(upper=idx[-1])
    fwd_ret = pd.Series(rng.normal(0, 0.01, size=200), index=idx)

    # Inject future label as a feature: perfect leakage
    future_label = y.shift(-1).fillna(0).astype(float)
    X_leaked = X_clean.copy()
    X_leaked["future_label"] = future_label.values

    return X_clean, X_leaked, y, t1, fwd_ret
