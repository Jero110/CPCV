"""
synthetic.py
Generate synthetic OHLCV price series using a multi-regime GBM model.
Each regime is simulated independently with its own drift and vol;
regimes are stitched continuously (last price of one = first price of next).
"""
import numpy as np
import pandas as pd

from cpcv_analysis.config import GBM_REGIME_PARAMS


def _business_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end, freq="B")


def _simulate_gbm_segment(
    start_price: float,
    drift: float,
    vol: float,
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate n_days of daily GBM close prices starting from start_price.
    drift and vol are annualized; dt = 1/252.
    Returns array of length n_days (does NOT include start_price).
    """
    dt = 1 / 252
    z = rng.standard_normal(n_days)
    log_returns = (drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
    prices = start_price * np.exp(np.cumsum(log_returns))
    return prices


def _add_intraday(closes: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Reconstruct OHLCV from close prices."""
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0] * (1 + rng.normal(0, 0.002))
    opens[1:] = closes[:-1]

    noise_hi = np.abs(rng.normal(0, 0.005, n))
    noise_lo = np.abs(rng.normal(0, 0.005, n))
    highs = np.maximum(opens, closes) * (1 + noise_hi)
    lows  = np.minimum(opens, closes) * (1 - noise_lo)
    lows  = np.minimum(lows, np.minimum(opens, closes))

    volumes = rng.integers(100_000, 10_000_001, n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                         "Close": closes, "Volume": volumes.astype(float)})


def generate_synthetic_prices(scenario: dict, seed: int) -> pd.DataFrame:
    """
    Generate continuous synthetic OHLCV price series for a multi-regime scenario.

    scenario: dict with keys id, name, regimes (list of (regime_type, start_date, end_date))
    seed: int — random seed for reproducibility

    Returns pd.DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by business-day DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    start_price = 100.0
    segments = []
    dates_segments = []

    for regime_type, seg_start, seg_end in scenario["regimes"]:
        params = GBM_REGIME_PARAMS[regime_type]
        dates = _business_days(seg_start, seg_end)
        if len(dates) == 0:
            continue
        closes = _simulate_gbm_segment(
            start_price=start_price,
            drift=params["drift"],
            vol=params["vol"],
            n_days=len(dates),
            rng=rng,
        )
        start_price = closes[-1]
        segments.append(closes)
        dates_segments.append(dates)

    all_closes = np.concatenate(segments)
    # Use list concatenation to avoid deprecated DatetimeIndex.append
    all_dates_list = []
    for d in dates_segments:
        all_dates_list.extend(d)
    all_dates = pd.DatetimeIndex(all_dates_list)

    # Remove duplicate dates from regime boundary overlaps
    unique_mask = ~pd.Series(all_dates).duplicated().values
    all_closes = all_closes[unique_mask]
    all_dates  = all_dates[unique_mask]

    ohlcv = _add_intraday(all_closes, rng)
    ohlcv.index = all_dates
    ohlcv.index.name = "Date"
    return ohlcv
