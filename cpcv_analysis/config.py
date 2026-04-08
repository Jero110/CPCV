# cpcv_analysis/config.py

# ── Data ──────────────────────────────────────────────────────────────────
TICKER         = "SPY"
START          = "2024-01-01"
END            = "2025-01-01"
FORWARD_HORIZON = 5          # trading days for label / t1 construction

# ── Synthetic crash ────────────────────────────────────────────────────────
CRASH_START     = "2024-06-03"   # first trading day of crash window
CRASH_DURATION  = 21             # trading days (~1 calendar month)
CRASH_MAGNITUDE = -0.20          # total return during window (−20 %)

# ── CPCV ───────────────────────────────────────────────────────────────────
N_GROUPS    = 6
K_TEST      = 2
PCT_EMBARGO = 0.01

# ── Model ──────────────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators  = 100,
    max_depth     = 3,
    learning_rate = 0.01,
    use_label_encoder = False,
    eval_metric   = "logloss",
    random_state  = 42,
)

# ── Plots ──────────────────────────────────────────────────────────────────
PLOT_DIR = "plots/"
FIGSIZE  = (12, 5)
