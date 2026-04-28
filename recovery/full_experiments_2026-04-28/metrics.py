"""
metrics.py
Calibration metrics for comparing a validation distribution against a true hold-out.
"""
import numpy as np
import pandas as pd


def _max_drawdown(pnl: pd.Series) -> float:
    """Maximum drawdown anchored at zero initial wealth. Returns value <= 0."""
    if len(pnl) == 0:
        return 0.0
    cum = pd.concat([pd.Series([0.0]), pnl]).cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return float(dd.min())


def compute_metrics(
    val_sharpes: pd.Series,
    ho_sr: float,
    val_pnl_concat: pd.Series,
    ho_pnl: pd.Series,
) -> dict:
    """
    Compute 6 calibration metrics for one experiment run.

    Parameters
    ----------
    val_sharpes    : Sharpe ratios for each validation fold/path
    ho_sr          : Sharpe ratio on the true hold-out set
    val_pnl_concat : Concatenated PnL series across all validation folds/paths
    ho_pnl         : PnL series on the true hold-out set

    Returns dict with keys: delta_median, coverage_90, rank_pct,
                             dispersion, delta_maxDD, pct_positive
    """
    if len(val_sharpes) == 0:
        raise ValueError("val_sharpes must not be empty")
    sharpes = np.asarray(val_sharpes)

    p5  = float(np.percentile(sharpes, 5))
    p95 = float(np.percentile(sharpes, 95))

    delta_median = float(np.median(sharpes)) - float(ho_sr)
    coverage_90  = int(p5 <= ho_sr <= p95)
    rank_pct     = float(np.mean(sharpes <= ho_sr))
    dispersion   = float(np.std(sharpes))
    delta_maxDD  = _max_drawdown(val_pnl_concat) - _max_drawdown(ho_pnl)
    pct_positive = float(np.mean(sharpes > 0))

    return dict(
        delta_median=delta_median,
        coverage_90=coverage_90,
        rank_pct=rank_pct,
        dispersion=dispersion,
        delta_maxDD=delta_maxDD,
        pct_positive=pct_positive,
    )
