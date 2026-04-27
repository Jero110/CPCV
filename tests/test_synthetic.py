import pytest
import pandas as pd
import numpy as np
from cpcv_analysis.config import COVID_DEV_START, COVID_HOLDOUT_END, GBM_REGIME_PARAMS


def test_generate_synthetic_prices_shape():
    from cpcv_analysis.synthetic import generate_synthetic_prices
    scenario = dict(
        id="test",
        name="test",
        regimes=[
            ("bull", COVID_DEV_START, "2019-01-01"),
            ("bear", "2019-01-01",    COVID_HOLDOUT_END),
        ],
    )
    prices = generate_synthetic_prices(scenario, seed=0)
    assert isinstance(prices, pd.DataFrame)
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(prices.columns)
    assert len(prices) > 100


def test_generate_synthetic_prices_continuous():
    from cpcv_analysis.synthetic import generate_synthetic_prices
    scenario = dict(
        id="test2",
        name="test2",
        regimes=[
            ("bull", COVID_DEV_START, "2019-01-01"),
            ("crash", "2019-01-01",   COVID_HOLDOUT_END),
        ],
    )
    prices = generate_synthetic_prices(scenario, seed=1)
    # No NaNs
    assert not prices["Close"].isna().any()
    # All prices positive
    assert (prices["Close"] > 0).all()
    # Continuity: no gap between regime boundaries
    closes = prices["Close"].values
    max_jump = np.max(np.abs(np.diff(np.log(closes))))
    assert max_jump < 1.0, f"Discontinuity detected: max_jump={max_jump:.4f}"


def test_generate_synthetic_prices_reproducible():
    from cpcv_analysis.synthetic import generate_synthetic_prices
    scenario = dict(
        id="test3",
        name="test3",
        regimes=[("bull", COVID_DEV_START, COVID_HOLDOUT_END)],
    )
    p1 = generate_synthetic_prices(scenario, seed=42)
    p2 = generate_synthetic_prices(scenario, seed=42)
    pd.testing.assert_frame_equal(p1, p2)


def test_generate_synthetic_prices_different_seeds():
    from cpcv_analysis.synthetic import generate_synthetic_prices
    scenario = dict(
        id="test4",
        name="test4",
        regimes=[("bull", COVID_DEV_START, COVID_HOLDOUT_END)],
    )
    p1 = generate_synthetic_prices(scenario, seed=1)
    p2 = generate_synthetic_prices(scenario, seed=2)
    assert not p1["Close"].equals(p2["Close"])
