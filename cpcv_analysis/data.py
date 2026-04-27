# cpcv_analysis/data.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
from cpcv_analysis.config import (
    TICKER, START, END, FORWARD_HORIZON,
    CRASH_START, CRASH_DURATION, CRASH_MAGNITUDE,
)

_CACHE_DIR = os.path.join("data_cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, f"{TICKER}_{START}_{END}.csv")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to a clean OHLCV DataFrame."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df.sort_index()


def _save_prices_cache(df: pd.DataFrame) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_csv(_CACHE_PATH, index_label="Date")


def _load_prices_cache() -> pd.DataFrame:
    if not os.path.exists(_CACHE_PATH):
        raise FileNotFoundError(_CACHE_PATH)
    df = pd.read_csv(_CACHE_PATH, parse_dates=["Date"], index_col="Date")
    if df.empty:
        raise RuntimeError(f"Cached price file is empty: {_CACHE_PATH}")
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def download_prices() -> pd.DataFrame:
    """Download SPY daily OHLCV via yfinance, falling back to local cache if needed."""
    try:
        raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError("yfinance returned an empty DataFrame")
        df = _normalize_ohlcv(raw)
        if df.empty:
            raise RuntimeError("downloaded OHLCV is empty after normalization")
        _save_prices_cache(df)
        print(f"[data] Downloaded {TICKER} prices and refreshed cache → {_CACHE_PATH}")
        return df
    except Exception as exc:
        print(f"[data] Download failed for {TICKER} ({exc}). Trying local cache...")
        try:
            df = _load_prices_cache()
            print(f"[data] Loaded cached {TICKER} prices → {_CACHE_PATH}")
            return df
        except Exception as cache_exc:
            raise RuntimeError(
                f"Failed to download {TICKER} prices for {START} to {END}, "
                f"and no usable local cache was found at {_CACHE_PATH}."
            ) from cache_exc


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

    # t1: end of the label window for each observation (t + FORWARD_HORIZON).
    # Used by getTrainTimes to purge train obs whose label overlaps with test.
    t1 = pd.Series(
        [df.index[i + FORWARD_HORIZON] if i + FORWARD_HORIZON < len(df) else df.index[-1]
         for i in range(len(df))],
        index=df.index,
    )

    feature_cols = ["rsi_14", "momentum_20", "rvol_20", "ret_lag1", "ret_lag5", "vol_norm"]
    df["fwd_ret"] = fwd_ret
    df = df.dropna(subset=feature_cols + ["target", "fwd_ret"])
    t1 = t1.loc[df.index]

    X      = df[feature_cols]
    y      = df["target"]
    fwd_r  = df["fwd_ret"]   # actual forward log-return for PnL calculation

    print(f"[data] Observations: {len(X)}  |  Class balance: {y.mean():.1%} up")
    print(f"[data] Feature shape: {X.shape}  |  Date range: {X.index[0].date()} → {X.index[-1].date()}")
    return X, y, t1, prices, fwd_r


def inject_leakage(X: pd.DataFrame, y: pd.Series,
                   feature_name: str = "future_label") -> pd.DataFrame:
    """
    Inject perfect future-label leakage into X.

    Adds a column `feature_name` = y.shift(-1) (the next period's label).
    This simulates a common data-preparation mistake where a derived feature
    accidentally includes information from the future.

    The last observation gets 0 (no future label available).
    """
    X = X.copy()
    future = y.shift(-1).fillna(0).astype(float)
    X[feature_name] = future.values
    print(f"[data] Leakage injected: column '{feature_name}' = y.shift(-1)")
    return X


def load_data():
    """Full pipeline: download → crash → features. Returns (X, y, t1, prices, fwd_ret)."""
    prices = download_prices()
    prices = inject_crash(prices)
    return build_features(prices)


def load_data_range(start: str = None, end: str = None, use_crash: bool = False):
    """
    Pipeline: download → (crash opcional) → features.
    Usa START/END del config si no se pasan argumentos.
    Returns (X, y, t1, prices, fwd_ret).
    """
    from cpcv_analysis.config import START as CFG_START, END as CFG_END
    _start = start or CFG_START
    _end   = end   or CFG_END

    try:
        raw = yf.download(TICKER, start=_start, end=_end,
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError("yfinance returned empty DataFrame")
        prices = _normalize_ohlcv(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {TICKER} {_start}→{_end}: {exc}") from exc

    if use_crash:
        prices = inject_crash(prices)

    return build_features(prices)


def load_asset(ticker: str, start: str, end: str, use_crash: bool = False):
    """
    Descarga OHLCV de `ticker` en [start, end), construye features.
    Retorna (X, y, t1, prices, fwd_ret).
    BTC-USD: yfinance lo maneja directamente igual que equities.

    CSV cache: raw normalized prices are cached in data_cache/{ticker}_{start}_{end}.csv
    to avoid redundant yfinance downloads. build_features() and inject_crash() are always
    applied after loading from cache (never cached themselves).
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")

    if os.path.exists(cache_file):
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        prices.index.name = "Date"
        print(f"[data] Loaded {ticker} prices from cache → {cache_file}")
    else:
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                raise RuntimeError(f"yfinance vacío para {ticker}")
            prices = _normalize_ohlcv(raw)
        except Exception as exc:
            raise RuntimeError(f"Falló descarga {ticker} {start}→{end}: {exc}") from exc
        prices.to_csv(cache_file)
        print(f"[data] Downloaded {ticker} prices and cached → {cache_file}")

    if use_crash:
        prices = inject_crash(prices)

    X, y, t1, _, fwd_ret = build_features(prices)
    return X, y, t1, prices, fwd_ret
