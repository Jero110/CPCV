import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


@pytest.mark.slow
def test_run_experiment_returns_metrics_and_figure():
    from xgboost import XGBClassifier
    from cpcv_analysis.config import XGB_PARAMS, TIMELINE_B
    from cpcv_analysis.experiment import run_experiment

    clf = XGBClassifier(**XGB_PARAMS)
    metrics, fig = run_experiment("SPY", TIMELINE_B, clf, method="cpcv")

    expected_keys = {
        "delta_median", "coverage_90", "rank_pct",
        "dispersion", "delta_maxDD", "pct_positive",
    }
    assert expected_keys.issubset(set(metrics.keys()))
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


@pytest.mark.slow
def test_run_experiment_all_methods():
    from xgboost import XGBClassifier
    from cpcv_analysis.config import XGB_PARAMS, TIMELINE_B
    from cpcv_analysis.experiment import run_experiment

    clf = XGBClassifier(**XGB_PARAMS)
    for method in ["cpcv", "wf", "kfold"]:
        metrics, fig = run_experiment("SPY", TIMELINE_B, clf, method=method)
        assert "delta_median" in metrics, f"method={method} missing delta_median"
        plt.close("all")
