# cpcv_analysis/data.py
import numpy as np
import pandas as pd
import yfinance as yf
from cpcv_analysis.config import (
    TICKER, START, END, FORWARD_HORIZON,
    CRASH_START, CRASH_DURATION, CRASH_MAGNITUDE,
)


def download_prices() -> pd.DataFrame:
    """Download SPY daily OHLCV via yfinance."""
    df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def inject_crash(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Inject a synthetic crash: multiply Close prices starting at CRASH_START
    by a linear ramp that reaches (1 + CRASH_MAGNITUDE) after CRASH_DURATION days,
    then holds that level for the rest of the series.
    """
    prices = prices.copy()
    crash_idx = prices.index.searchsorted(pd.Timestamp(CRASH_START))
    end_idx   = min(crash_idx + CRASH_DURATION, len(prices))

    # Linear drawdown over CRASH_DURATION bars
    ramp = np.linspace(1.0, 1.0 + CRASH_MAGNITUDE, end_idx - crash_idx)
    multiplier = np.ones(len(prices))
    multiplier[crash_idx:end_idx] = ramp
    multiplier[end_idx:]          = 1.0 + CRASH_MAGNITUDE

    # Apply to Close (and propagate to Open/High/Low for consistency)
    for col in ["Open", "High", "Low", "Close"]:
        prices[col] = prices[col] * multiplier

    print(f"[data] Crash injected: {CRASH_START} → {prices.index[end_idx-1].date()}, "
          f"magnitude={CRASH_MAGNITUDE:.0%}")
    return prices


def build_features(prices: pd.DataFrame):
    """
    Build X (features), y (binary label), t1 (label end dates) from prices.

    Features:
        rsi_14        — RSI with 14-day window
        momentum_20   — price ratio Close / Close.shift(20)
        rvol_20       — realized volatility (rolling 20d std of log returns)
        ret_lag1      — lagged 1d log return
        ret_lag5      — lagged 5d log return
        vol_norm      — volume / rolling 20d mean volume

    Label: 1 if forward 5d return > 0, else 0
    t1:    index of the last bar used to construct each label
    """
    df = prices.copy()
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # ── RSI-14 ──────────────────────────────────────────────────────────────
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Other features ──────────────────────────────────────────────────────
    df["momentum_20"] = df["Close"] / df["Close"].shift(20)
    df["rvol_20"]     = log_ret.rolling(20).std()
    df["ret_lag1"]    = log_ret.shift(1)
    df["ret_lag5"]    = log_ret.rolling(5).sum().shift(1)
    df["vol_norm"]    = df["Volume"] / df["Volume"].rolling(20).mean()

    # ── Label & t1 ──────────────────────────────────────────────────────────
    fwd_ret = np.log(df["Close"].shift(-FORWARD_HORIZON) / df["Close"])
    df["target"] = (fwd_ret > 0).astype(int)

    # t1: the timestamp FORWARD_HORIZON bars ahead of each observation
    t1 = pd.Series(
        [df.index[i + FORWARD_HORIZON] if i + FORWARD_HORIZON < len(df) else df.index[-1]
         for i in range(len(df))],
        index=df.index,
    )

    feature_cols = ["rsi_14", "momentum_20", "rvol_20", "ret_lag1", "ret_lag5", "vol_norm"]
    df = df.dropna(subset=feature_cols + ["target"])
    t1 = t1.loc[df.index]

    X = df[feature_cols]
    y = df["target"]

    print(f"[data] Observations: {len(X)}  |  Class balance: {y.mean():.1%} up")
    print(f"[data] Feature shape: {X.shape}  |  Date range: {X.index[0].date()} → {X.index[-1].date()}")
    return X, y, t1, prices


def load_data():
    """Full pipeline: download → crash → features. Returns (X, y, t1, prices)."""
    prices = download_prices()
    prices = inject_crash(prices)
    return build_features(prices)
