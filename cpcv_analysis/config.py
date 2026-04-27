# cpcv_analysis/config.py

# ── Data ──────────────────────────────────────────────────────────────────
TICKER          = "SPY"
START           = "2023-10-01"   # Q4-2023 warm-up para WF
END             = "2026-01-01"   # cubre 2025 completo
FORWARD_HORIZON = 5

# ── Experiment date boundaries (Phase 1: SPY) ──────────────────────────────
DEV_START      = "2024-01-01"
DEV_END        = "2025-01-01"
RETRAIN_START  = "2025-01-01"
RETRAIN_END    = "2025-09-01"
HOLDOUT_START  = "2025-09-01"
HOLDOUT_END    = "2026-01-01"

# ── Multi-asset tickers (Phase 2) ──────────────────────────────────────────
ASSETS = {
    "SPY":     "Broad Market ETF",
    "AAPL":    "Large Cap Stock",
    "IWM":     "Small Cap ETF",
    "BTC-USD": "Crypto",
}
ASSET_START = "2022-10-01"   # Q4-2022: warm-up para WF en 3-year study
ASSET_END   = "2026-01-01"   # cubre 2023+2024+2025

# ── Phase 2/3 date splits (3-year study) ───────────────────────────────────
# Development: 2023+2024 (24 meses). Hold-out: 2025 (12 meses)
ASSET_DEV_START     = "2023-01-01"
ASSET_DEV_END       = "2025-01-01"
ASSET_RETRAIN_START = "2025-01-01"
ASSET_RETRAIN_END   = "2025-09-01"
ASSET_HOLDOUT_START = "2025-09-01"
ASSET_HOLDOUT_END   = "2026-01-01"

# ── Synthetic crash ────────────────────────────────────────────────────────
CRASH_START     = "2024-06-03"   # first trading day of crash window
CRASH_DURATION  = 21             # trading days (~1 calendar month)
CRASH_MAGNITUDE = -0.20          # total return during window (−20 %)

# ── Leakage scenario ───────────────────────────────────────────────────────
LEAKAGE_FEATURE_NAME = "future_label"   # name of the injected leakage column

# ── Monte Carlo Overfit ────────────────────────────────────────────────────
IS_FRAC = 0.70           # in-sample fraction for MC permutation tests

# ── CPCV ───────────────────────────────────────────────────────────────────
N_GROUPS    = 6
K_TEST      = 2
PCT_EMBARGO = 0.02

# ── Model ──────────────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators  = 100,
    max_depth     = 3,
    learning_rate = 0.01,
    eval_metric   = "logloss",
    random_state  = 42,
)

# ── Plots ──────────────────────────────────────────────────────────────────
PLOT_DIR = "plots/"
FIGSIZE  = (12, 5)
