"""
tests/test_backtest_engine_kfold.py
TDD tests for _method_vs_holdout_plot and kfold_vs_holdout_plot.
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier

from cpcv_analysis.backtest_engine import (
    _method_vs_holdout_plot,
    kfold_vs_holdout_plot,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_synthetic(n=100, seed=42):
    """Return X, y, t1, fwd_ret with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    X = pd.DataFrame(rng.standard_normal((n, 4)), index=dates,
                     columns=["f1", "f2", "f3", "f4"])
    y = pd.Series((rng.standard_normal(n) > 0).astype(int), index=dates)
    # t1: each label ends 2 days later
    t1 = pd.Series(dates + pd.Timedelta(days=2), index=dates)
    fwd_ret = pd.Series(rng.standard_normal(n) * 0.01, index=dates)
    return X, y, t1, fwd_ret


def _make_clf():
    return DummyClassifier(strategy="stratified", random_state=0)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestMethodVsHoldoutPlot:

    def test_returns_figure(self):
        """_method_vs_holdout_plot returns a matplotlib Figure."""
        val_sharpes = pd.Series([0.3, -0.1, 0.5, 0.2, 0.4], name="test")
        fig = _method_vs_holdout_plot(
            val_sharpes=val_sharpes,
            ho_sr=0.35,
            prices_full=None,
            timeline_cfg={},
            method_label="TestMethod",
            fig_title="Test Figure",
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_no_prices_no_raise(self):
        """With prices_full=None and timeline_cfg={} it must not raise."""
        val_sharpes = pd.Series([0.1, 0.2, -0.1, 0.4, 0.3])
        try:
            fig = _method_vs_holdout_plot(
                val_sharpes=val_sharpes,
                ho_sr=0.25,
                prices_full=None,
                timeline_cfg={},
                method_label="KFold",
                fig_title="No prices test",
            )
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close("all")

    def test_with_prices_and_timeline(self):
        """When prices_full and timeline_cfg are provided, figure is still returned."""
        dates = pd.date_range("2023-01-02", periods=200, freq="B")
        prices = pd.DataFrame({"Close": np.linspace(100, 150, 200)}, index=dates)
        timeline_cfg = {
            "wf_start":       "2023-01-02",
            "dev_start":      "2023-03-01",
            "dev_end":        "2023-06-01",
            "retrain_start":  "2023-06-01",
            "retrain_end":    "2023-07-01",
            "holdout_start":  "2023-07-01",
            "holdout_end":    "2023-09-01",
        }
        val_sharpes = pd.Series([0.1, 0.3, -0.2, 0.5, 0.4])
        fig = _method_vs_holdout_plot(
            val_sharpes=val_sharpes,
            ho_sr=0.20,
            prices_full=prices,
            timeline_cfg=timeline_cfg,
            method_label="KFold",
            fig_title="With prices",
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_figsize(self):
        """Figure size must be (14, 5)."""
        val_sharpes = pd.Series([0.1, 0.2, 0.3])
        fig = _method_vs_holdout_plot(
            val_sharpes=val_sharpes,
            ho_sr=0.15,
            prices_full=None,
            timeline_cfg={},
            method_label="KFold",
            fig_title="Figsize test",
        )
        w, h = fig.get_size_inches()
        assert abs(w - 14) < 0.1
        assert abs(h - 5) < 0.1
        plt.close("all")

    def test_does_not_call_show(self, monkeypatch):
        """_method_vs_holdout_plot must NOT call plt.show() internally."""
        show_called = []
        monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
        val_sharpes = pd.Series([0.1, 0.3])
        _method_vs_holdout_plot(
            val_sharpes=val_sharpes,
            ho_sr=0.2,
            prices_full=None,
            timeline_cfg={},
            method_label="KFold",
            fig_title="No show",
        )
        assert show_called == [], "_method_vs_holdout_plot must not call plt.show()"
        plt.close("all")


class TestKfoldVsHoldoutPlot:

    def test_returns_tuple(self):
        """kfold_vs_holdout_plot returns a tuple (pd.Series, float)."""
        X, y, t1, fwd_ret = _make_synthetic(n=120)
        clf = _make_clf()

        # dev: first 80, retrain: 80-100, holdout: 100-120
        X_dev = X.iloc[:80]
        y_dev = y.iloc[:80]
        t1_dev = t1.iloc[:80]
        fwd_ret_dev = fwd_ret.iloc[:80]

        X_retrain = X.iloc[80:100]
        y_retrain = y.iloc[80:100]

        X_holdout = X.iloc[100:]
        fwd_ret_holdout = fwd_ret.iloc[100:]

        result = kfold_vs_holdout_plot(
            clf=clf,
            X_dev=X_dev, y_dev=y_dev, t1_dev=t1_dev, fwd_ret_dev=fwd_ret_dev,
            X_retrain=X_retrain, y_retrain=y_retrain,
            X_holdout=X_holdout, fwd_ret_holdout=fwd_ret_holdout,
            prices_full=None,
            n_splits=4,
        )
        plt.close("all")

        assert isinstance(result, tuple), "Must return a tuple"
        assert len(result) == 2, "Tuple must have 2 elements"
        kf_sharpes, ho_sr = result
        assert isinstance(kf_sharpes, pd.Series), "First element must be pd.Series"
        assert isinstance(ho_sr, float), "Second element must be float"

    def test_returns_non_empty_series(self):
        """kfold_vs_holdout_plot returns a non-empty Series of Sharpes."""
        X, y, t1, fwd_ret = _make_synthetic(n=120)
        clf = _make_clf()

        X_dev = X.iloc[:80]
        y_dev = y.iloc[:80]
        t1_dev = t1.iloc[:80]
        fwd_ret_dev = fwd_ret.iloc[:80]
        X_retrain = X.iloc[80:100]
        y_retrain = y.iloc[80:100]
        X_holdout = X.iloc[100:]
        fwd_ret_holdout = fwd_ret.iloc[100:]

        kf_sharpes, ho_sr = kfold_vs_holdout_plot(
            clf=clf,
            X_dev=X_dev, y_dev=y_dev, t1_dev=t1_dev, fwd_ret_dev=fwd_ret_dev,
            X_retrain=X_retrain, y_retrain=y_retrain,
            X_holdout=X_holdout, fwd_ret_holdout=fwd_ret_holdout,
            n_splits=4,
        )
        plt.close("all")

        assert len(kf_sharpes) > 0, "Sharpe series must not be empty"

    def test_no_prices_no_raise(self):
        """kfold_vs_holdout_plot with prices_full=None should not raise."""
        X, y, t1, fwd_ret = _make_synthetic(n=100)
        clf = _make_clf()

        X_dev = X.iloc[:60]
        y_dev = y.iloc[:60]
        t1_dev = t1.iloc[:60]
        fwd_ret_dev = fwd_ret.iloc[:60]
        X_retrain = X.iloc[60:80]
        y_retrain = y.iloc[60:80]
        X_holdout = X.iloc[80:]
        fwd_ret_holdout = fwd_ret.iloc[80:]

        try:
            result = kfold_vs_holdout_plot(
                clf=clf,
                X_dev=X_dev, y_dev=y_dev, t1_dev=t1_dev, fwd_ret_dev=fwd_ret_dev,
                X_retrain=X_retrain, y_retrain=y_retrain,
                X_holdout=X_holdout, fwd_ret_holdout=fwd_ret_holdout,
                prices_full=None,
                n_splits=4,
            )
        finally:
            plt.close("all")
