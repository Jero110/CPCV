# CPCV — Combinatorial Purged Cross-Validation for Financial ML

Implementation and empirical analysis of Combinatorial Purged Cross-Validation (CPCV) as described in *Advances in Financial Machine Learning* by Marcos López de Prado (2018), Chapters 7 and 11–12.

Applied to SPY 2024 daily data across three scenarios: clean, synthetic crash, and feature leakage detection.

---

## Repo structure

```
CPCV/
├── cpcv_analysis/
│   ├── config.py            # Global parameters (N_GROUPS, K_TEST, embargo, XGB)
│   ├── data.py              # Download, feature engineering, crash/leakage injection
│   ├── splitters.py         # getTrainTimes, getEmbargoTimes, PurgedKFold, CPCV, WalkForward
│   ├── cv_runner.py         # cvScore, get_paths, run_cpcv, performance metrics
│   ├── comparison.py        # 9-method comparison (KFold, WalkForward, CCV, CPCV variants)
│   ├── advanced_analysis.py # OOS degradation, rank logits, Prob[Overfit]
│   └── plots.py             # All figures (12 plots across 3 scenarios)
├── tests/
│   ├── test_splitters.py    # Purge/embargo correctness, CPCV split count, path properties
│   ├── test_leakage.py      # Leakage detection: KFold exploits it, purge removes boundary overlap
│   ├── audit_theory.py      # Manual theory checks (not pytest)
│   └── conftest.py          # Shared fixtures
├── docs/
│   ├── cpcv_report.tex      # Full LaTeX analysis report
│   ├── cpcv_report.pdf      # Compiled PDF (see below)
│   └── de_prado_image_guide.md  # Index of De Prado reference images
├── fotos/                   # De Prado book screenshots (snippets, figures, formulas)
├── CPCV_LP_analysis.ipynb   # Exploratory notebook
└── .gitignore
```

---

## Quickstart

All commands use the `rappi` conda environment.

### Run full analysis (Scenarios A, B, C)

```bash
conda run -n rappi python3 -m cpcv_analysis.main
```

Outputs plots to `plots/A_clean/`, `plots/B_crash/`, `plots/C_leakage/`.

### Run tests

```bash
conda run -n rappi python3 -m pytest tests/ -v
```

Individual test files:

```bash
# Splitter correctness (purge, embargo, CPCV structure, path properties)
conda run -n rappi python3 -m pytest tests/test_splitters.py -v

# Leakage detection behavior
conda run -n rappi python3 -m pytest tests/test_leakage.py -v
```

### Run a single scenario

```python
from cpcv_analysis.data import download_prices, build_features
from cpcv_analysis.splitters import CombinatorialPurgedKFold
from cpcv_analysis.cv_runner import run_cpcv
from cpcv_analysis.config import N_GROUPS, K_TEST, PCT_EMBARGO, XGB_PARAMS
from xgboost import XGBClassifier

prices = download_prices()
X, y, t1, _, fwd_ret = build_features(prices)
cpcv = CombinatorialPurgedKFold(N_GROUPS, K_TEST, t1, PCT_EMBARGO)
clf  = XGBClassifier(**XGB_PARAMS)
fold_results, path_results, oos_by_split = run_cpcv(clf, X, y, t1, cpcv, fwd_ret=fwd_ret)
```

---

## Three scenarios

| Scenario | Data | Purpose |
|----------|------|---------|
| **A — Clean** | SPY 2024 as downloaded | Baseline CPCV analysis and method comparison |
| **B — Crash** | SPY 2024 + synthetic −20% crash (Jun 2024) | Regime change stress test |
| **C — Leakage** | SPY 2024 + injected future-label feature | Feature leakage detection across all CV methods |

---

## Method comparison

Nine CV schemes are compared, varying splitter type and whether purging/embargo are applied:

| Method | Splitter | Purge | Embargo |
|--------|----------|-------|---------|
| KFold | Sequential KFold | No | No |
| KFold+Purge | PurgedKFold | Yes | No |
| KFold+Purge+Embargo | PurgedKFold | Yes | Yes |
| WalkForward | Expanding window | No | No |
| WalkForward+Purge | Expanding window | Yes | No |
| WalkForward+Purge+Embargo | Expanding window | Yes | Yes |
| CCV | Combinatorial (raw) | No | No |
| CCV+Purge | Combinatorial | Yes | No |
| **CPCV** | **Combinatorial** | **Yes** | **Yes** |

---

## Key results (Scenario A — clean SPY 2024)

- **IS Sharpe:** ~8 across all methods (structural in-sample overfit)
- **OOS Sharpe (CPCV paths):** 0.85 – 3.64 across 5 paths
- **Mean ΔSR:** 5.71 (IS − OOS degradation)
- **Prob[Overfit]:** ~0.49 (estimated from rank logit distribution)
- **φ = 5 paths**, each a complete simulated backtest covering all 6 time groups once

---

## Configuration (`cpcv_analysis/config.py`)

```python
N_GROUPS    = 6       # time groups → C(6,2) = 15 splits, φ = 5 paths
K_TEST      = 2       # test groups per split
PCT_EMBARGO = 0.01    # 1% embargo (~2 obs on 227-obs dataset)
FORWARD_HORIZON = 5   # days for label and t1 construction
```

---

## Analysis report

The full write-up — method descriptions, formulas, code snippets, result tables, and all figures — is in:

**[`docs/cpcv_report.pdf`](docs/cpcv_report.pdf)**

Covers: purging & embargo implementation, CPCV path construction, OOS degradation analysis, rank logits and PBO estimation, walk-forward vs CPCV comparison, and the feature leakage experiment.

---

## Reference

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
