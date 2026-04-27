# cpcv_analysis/config.py

# ── Data ──────────────────────────────────────────────────────────────────
TICKER          = "SPY"
START           = "2023-05-01"   # 8 meses antes de DEV_START para warm-up WF rolling
END             = "2026-01-01"   # cubre 2025 completo
FORWARD_HORIZON = 5

# ── Experiment date boundaries (Phase 1: SPY) ──────────────────────────────
WF_START       = "2023-05-01"   # inicio warm-up WF rolling (8 meses antes de dev)
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

# ── CPCV ───────────────────────────────────────────────────────────────────
N_GROUPS    = 6
K_TEST      = 2
PCT_EMBARGO = 0.01

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

# ── Timeline B: COVID 2017-2020 ───────────────────────────────────────────────
COVID_DOWNLOAD_START  = "2017-05-01"
COVID_DOWNLOAD_END    = "2020-05-01"
COVID_WF_START        = "2017-05-01"
COVID_DEV_START       = "2018-09-01"
COVID_DEV_END         = "2019-09-01"
COVID_RETRAIN_START   = "2019-09-01"
COVID_RETRAIN_END     = "2020-01-01"
COVID_HOLDOUT_START   = "2020-01-01"
COVID_HOLDOUT_END     = "2020-05-01"

# ── Timeline configs (for experiment.py) ─────────────────────────────────────
TIMELINE_A = dict(
    name="2023-2026",
    download_start=START,
    download_end=END,
    wf_start=WF_START,
    dev_start=DEV_START,
    dev_end=DEV_END,
    retrain_start=RETRAIN_START,
    retrain_end=RETRAIN_END,
    holdout_start=HOLDOUT_START,
    holdout_end=HOLDOUT_END,
)

TIMELINE_B = dict(
    name="COVID-2017-2020",
    download_start=COVID_DOWNLOAD_START,
    download_end=COVID_DOWNLOAD_END,
    wf_start=COVID_WF_START,
    dev_start=COVID_DEV_START,
    dev_end=COVID_DEV_END,
    retrain_start=COVID_RETRAIN_START,
    retrain_end=COVID_RETRAIN_END,
    holdout_start=COVID_HOLDOUT_START,
    holdout_end=COVID_HOLDOUT_END,
)

TIMELINES = [TIMELINE_A, TIMELINE_B]

# ── GBM regime parameters (annualized drift and vol) ─────────────────────────
GBM_REGIME_PARAMS = {
    "bull":     dict(drift=0.12,  vol=0.12),
    "bear":     dict(drift=-0.15, vol=0.20),
    "stagnant": dict(drift=0.00,  vol=0.08),
    "crash":    dict(drift=-0.80, vol=0.60),
    "recovery": dict(drift=0.40,  vol=0.25),
}

# ── 20 pre-defined synthetic scenarios ───────────────────────────────────────
# Each scenario: list of (regime_type, start_date, end_date) tuples
# All use COVID Timeline B calendar
SYNTHETIC_SCENARIOS = [
    # 01 bull_bull_bull
    dict(id="01", name="bull_bull_bull", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 02 bear_bear_bear
    dict(id="02", name="bear_bear_bear", regimes=[
        ("bear",     COVID_DEV_START,     COVID_DEV_END),
        ("bear",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bear",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 03 stagnant_all
    dict(id="03", name="stagnant_all", regimes=[
        ("stagnant", COVID_DEV_START,     COVID_DEV_END),
        ("stagnant", COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("stagnant", COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 04 crash_in_dev_start
    dict(id="04", name="crash_in_dev_start", regimes=[
        ("crash",    COVID_DEV_START,     "2018-10-01"),
        ("bull",     "2018-10-01",        COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 05 crash_in_dev_end
    dict(id="05", name="crash_in_dev_end", regimes=[
        ("bull",     COVID_DEV_START,     "2019-07-01"),
        ("crash",    "2019-07-01",        COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 06 crash_in_retrain
    dict(id="06", name="crash_in_retrain", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("crash",    COVID_RETRAIN_START, "2019-10-15"),
        ("recovery", "2019-10-15",        COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 07 crash_in_holdout_start
    dict(id="07", name="crash_in_holdout_start", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("crash",    COVID_HOLDOUT_START, "2020-02-01"),
        ("stagnant", "2020-02-01",        COVID_HOLDOUT_END),
    ]),
    # 08 crash_in_holdout_end
    dict(id="08", name="crash_in_holdout_end", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, "2020-03-01"),
        ("crash",    "2020-03-01",        COVID_HOLDOUT_END),
    ]),
    # 09 bull_then_crash_holdout
    dict(id="09", name="bull_then_crash_holdout", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, "2020-02-15"),
        ("crash",    "2020-02-15",        COVID_HOLDOUT_END),
    ]),
    # 10 bear_then_recovery_holdout
    dict(id="10", name="bear_then_recovery_holdout", regimes=[
        ("bear",     COVID_DEV_START,     COVID_DEV_END),
        ("bear",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("recovery", COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 11 crash_dev_recovery_holdout
    dict(id="11", name="crash_dev_recovery_holdout", regimes=[
        ("crash",    COVID_DEV_START,     "2018-10-01"),
        ("recovery", "2018-10-01",        COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("recovery", COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 12 stagnant_crash_holdout
    dict(id="12", name="stagnant_crash_holdout", regimes=[
        ("stagnant", COVID_DEV_START,     COVID_DEV_END),
        ("stagnant", COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("crash",    COVID_HOLDOUT_START, "2020-02-15"),
        ("stagnant", "2020-02-15",        COVID_HOLDOUT_END),
    ]),
    # 13 bull_crash_retrain_recovery
    dict(id="13", name="bull_crash_retrain_recovery", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("crash",    COVID_RETRAIN_START, "2019-11-01"),
        ("recovery", "2019-11-01",        COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 14 double_crash_dev_holdout
    dict(id="14", name="double_crash_dev_holdout", regimes=[
        ("crash",    COVID_DEV_START,     "2018-10-01"),
        ("recovery", "2018-10-01",        COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("crash",    COVID_HOLDOUT_START, "2020-02-15"),
        ("stagnant", "2020-02-15",        COVID_HOLDOUT_END),
    ]),
    # 15 recovery_after_crash_dev
    dict(id="15", name="recovery_after_crash_dev", regimes=[
        ("bull",     COVID_DEV_START,     "2019-03-01"),
        ("crash",    "2019-03-01",        "2019-05-01"),
        ("recovery", "2019-05-01",        COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 16 crash_start_warmup
    dict(id="16", name="crash_start_warmup", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 17 long_bull_short_crash_holdout
    dict(id="17", name="long_bull_short_crash_holdout", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, "2020-03-15"),
        ("crash",    "2020-03-15",        COVID_HOLDOUT_END),
    ]),
    # 18 volatile_bull
    dict(id="18", name="volatile_bull", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 19 bear_crash_recovery
    dict(id="19", name="bear_crash_recovery", regimes=[
        ("bear",     COVID_DEV_START,     "2019-03-01"),
        ("crash",    "2019-03-01",        "2019-05-01"),
        ("recovery", "2019-05-01",        COVID_DEV_END),
        ("bear",     COVID_RETRAIN_START, "2019-11-01"),
        ("crash",    "2019-11-01",        COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, COVID_HOLDOUT_END),
    ]),
    # 20 realistic_covid_analog
    dict(id="20", name="realistic_covid_analog", regimes=[
        ("bull",     COVID_DEV_START,     COVID_DEV_END),
        ("bull",     COVID_RETRAIN_START, COVID_RETRAIN_END),
        ("bull",     COVID_HOLDOUT_START, "2020-02-20"),
        ("crash",    "2020-02-20",        "2020-03-23"),
        ("recovery", "2020-03-23",        COVID_HOLDOUT_END),
    ]),
]
