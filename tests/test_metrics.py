import pytest
import numpy as np
import pandas as pd


def test_compute_metrics_returns_all_keys():
    from cpcv_analysis.metrics import compute_metrics
    val_sharpes = pd.Series([0.5, 0.8, 1.2, -0.3, 0.6])
    ho_sr = 0.7
    val_pnl = pd.Series(np.ones(50) * 0.01)
    ho_pnl  = pd.Series(np.ones(20) * 0.01)
    result = compute_metrics(val_sharpes, ho_sr, val_pnl, ho_pnl)
    expected_keys = {
        "delta_median", "coverage_90", "rank_pct",
        "dispersion", "delta_maxDD", "pct_positive",
    }
    assert expected_keys == set(result.keys())


def test_compute_metrics_delta_median():
    from cpcv_analysis.metrics import compute_metrics
    val_sharpes = pd.Series([1.0, 1.0, 1.0])
    ho_sr = 0.5
    result = compute_metrics(val_sharpes, ho_sr,
                             pd.Series([0.01] * 30), pd.Series([0.01] * 10))
    assert abs(result["delta_median"] - 0.5) < 1e-10


def test_compute_metrics_coverage_90_inside():
    from cpcv_analysis.metrics import compute_metrics
    val_sharpes = pd.Series(np.linspace(0, 2, 100))
    ho_sr = 1.0
    result = compute_metrics(val_sharpes, ho_sr,
                             pd.Series([0.0] * 50), pd.Series([0.0] * 10))
    assert result["coverage_90"] == 1


def test_compute_metrics_coverage_90_outside():
    from cpcv_analysis.metrics import compute_metrics
    val_sharpes = pd.Series(np.linspace(0, 1, 100))
    ho_sr = 5.0
    result = compute_metrics(val_sharpes, ho_sr,
                             pd.Series([0.0] * 50), pd.Series([0.0] * 10))
    assert result["coverage_90"] == 0


def test_compute_metrics_pct_positive():
    from cpcv_analysis.metrics import compute_metrics
    val_sharpes = pd.Series([1.0, -1.0, 2.0, -2.0])
    result = compute_metrics(val_sharpes, 0.0,
                             pd.Series([0.0] * 20), pd.Series([0.0] * 5))
    assert abs(result["pct_positive"] - 0.5) < 1e-10


def test_compute_metrics_max_dd():
    from cpcv_analysis.metrics import compute_metrics
    val_pnl = pd.Series([0.01] * 50)
    ho_pnl  = pd.Series([-0.01] * 20)
    result = compute_metrics(pd.Series([0.5]), 0.5, val_pnl, ho_pnl)
    # val_pnl monotone up → maxDD ≈ 0; ho_pnl monotone down → maxDD < 0
    # delta_maxDD = maxDD(val_pnl) - maxDD(ho_pnl) > 0
    assert result["delta_maxDD"] > 0
