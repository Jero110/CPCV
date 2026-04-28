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
    Calibration metrics comparing a validation distribution against the true hold-out.

    Returns dict with keys:
        delta        : mean(val_SRs) - ho_sr
        bias         : delta / |ho_sr|  (NaN when ho_sr == 0)
        z_score      : (ho_sr - mean(val_SRs)) / std(val_SRs)  (NaN when n < 2)
        coverage_90  : 1 if ho_sr in [P5, P95] of val_SRs
        rank_pct     : fraction of val_SRs <= ho_sr
        dispersion   : std(val_SRs)
        delta_maxDD  : maxDD(val_pnl) - maxDD(ho_pnl)
        pct_positive : fraction of val_SRs > 0
    """
    if len(val_sharpes) == 0:
        raise ValueError("val_sharpes must not be empty")
    sharpes = np.asarray(val_sharpes, dtype=float)

    p5  = float(np.percentile(sharpes, 5))
    p95 = float(np.percentile(sharpes, 95))

    mu_val = float(np.mean(sharpes))
    std_val = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else float("nan")

    delta = mu_val - float(ho_sr)
    bias = delta / abs(float(ho_sr)) if abs(float(ho_sr)) > 1e-10 else float("nan")
    z_score = (
        (float(ho_sr) - mu_val) / std_val
        if not np.isnan(std_val) and std_val > 0 else float("nan")
    )
    coverage_90  = int(p5 <= ho_sr <= p95)
    rank_pct     = float(np.mean(sharpes <= ho_sr))
    dispersion   = std_val if not np.isnan(std_val) else 0.0
    delta_maxDD  = _max_drawdown(val_pnl_concat) - _max_drawdown(ho_pnl)
    pct_positive = float(np.mean(sharpes > 0))

    return dict(
        delta=delta,
        bias=bias,
        z_score=z_score,
        coverage_90=coverage_90,
        rank_pct=rank_pct,
        dispersion=dispersion,
        delta_maxDD=delta_maxDD,
        pct_positive=pct_positive,
    )
