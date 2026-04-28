# cpcv_analysis/experiment.py
"""
experiment.py
Single entry-point for running one (ticker × timeline × method) experiment.

Usage:
    from cpcv_analysis.experiment import run_experiment
    metrics, fig = run_experiment("SPY", TIMELINE_B, clf, method="cpcv")
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import KFold

from cpcv_analysis.data import load_asset
from cpcv_analysis.backtest_engine import (
    slice_by_dates,
    holdout_sharpe,
    cpcv_sharpe_dist,
    wf_rolling_sharpe_dist,
    kfold_sharpe_dist,
    _method_vs_holdout_plot,
    _build_cpcv_splits_table,
    _wf_rolling_fold_dates,
    _pnl_from_split,
)
from cpcv_analysis.metrics import compute_metrics
from cpcv_analysis.config import N_GROUPS, K_TEST, PCT_EMBARGO
from cpcv_analysis.splitters import PurgedKFold, RollingWalkForwardCV
from cpcv_analysis.cv_runner import get_paths


# ── Internal helpers ──────────────────────────────────────────────────────────

def _holdout_pnl(clf, X_retrain, y_retrain, X_holdout, fwd_ret_holdout) -> pd.Series:
    """Retrain clf on (X_retrain, y_retrain), predict on X_holdout, compute PnL."""
    clf_final = clone(clf)
    clf_final.fit(X_retrain, y_retrain)
    y_pred = clf_final.predict(X_holdout)
    signs  = (2 * y_pred - 1).astype(float)
    return pd.Series(
        signs * fwd_ret_holdout.values,
        index=X_holdout.index,
        dtype=float,
    )


def _try_cpcv_val_pnl(clf, X_dev, y_dev, t1_dev, fwd_ret_dev) -> pd.Series:
    """
    Concatenate all OOS PnL segments from CPCV splits (one observation per day).
    Falls back to empty Series if anything fails.
    """
    try:
        _, oos_by_split, _, _ = _build_cpcv_splits_table(
            clf, X_dev, y_dev, t1_dev, fwd_ret_dev,
            n_groups=N_GROUPS, k_test=K_TEST, pct_embargo=PCT_EMBARGO,
        )
        paths = get_paths(N_GROUPS, K_TEST)
        pieces = []
        for split_ids in paths:
            for sid in split_ids:
                if sid in oos_by_split:
                    pieces.append(oos_by_split[sid])
        if pieces:
            return pd.concat(pieces).sort_index().groupby(level=0).mean()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def _try_wf_val_pnl(clf, X_dev, y_dev, t1_dev, fwd_ret_dev, timeline_cfg) -> pd.Series:
    """
    Collect OOS PnL across WF rolling folds.
    Falls back to empty Series if anything fails.
    """
    try:
        fold_dates = _wf_rolling_fold_dates(
            wf_start=timeline_cfg["wf_start"],
            dev_start=timeline_cfg["dev_start"],
            dev_end=timeline_cfg["dev_end"],
        )
        splitter = RollingWalkForwardCV(fold_dates, t1=t1_dev, pctEmbargo=PCT_EMBARGO)
        pieces = []
        for train_idx, test_idx in splitter.split(X_dev):
            train_idx = np.array(train_idx)
            test_idx  = np.array(test_idx)
            if len(train_idx) < 5 or len(test_idx) < 2:
                continue
            if len(np.unique(y_dev.iloc[train_idx])) < 2:
                continue
            _, oos_pnl, _, _ = _pnl_from_split(
                clf, X_dev, y_dev, t1_dev, fwd_ret_dev, train_idx, test_idx)
            pieces.append(oos_pnl)
        if pieces:
            return pd.concat(pieces).sort_index()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def _try_kfold_val_pnl(clf, X_dev, y_dev, t1_dev, fwd_ret_dev) -> pd.Series:
    """
    Collect OOS PnL across KFold purge+embargo folds.
    Falls back to empty Series if anything fails.
    """
    try:
        splitter = PurgedKFold(n_splits=6, t1=t1_dev, pctEmbargo=PCT_EMBARGO)
        pieces = []
        for train_idx, test_idx in splitter.split(X_dev):
            train_idx = np.array(train_idx)
            test_idx  = np.array(test_idx)
            if len(train_idx) < 5 or len(test_idx) < 2:
                continue
            if len(np.unique(y_dev.iloc[train_idx])) < 2:
                continue
            _, oos_pnl, _, _ = _pnl_from_split(
                clf, X_dev, y_dev, t1_dev, fwd_ret_dev, train_idx, test_idx)
            pieces.append(oos_pnl)
        if pieces:
            return pd.concat(pieces).sort_index()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


# ── Public API ────────────────────────────────────────────────────────────────

def run_experiment(ticker: str, timeline_cfg: dict, clf, method: str) -> tuple:
    """
    Run a single (ticker × timeline × method) experiment.

    Parameters
    ----------
    ticker       : Yahoo Finance ticker symbol, e.g. "SPY"
    timeline_cfg : dict with keys: download_start, download_end, wf_start,
                   dev_start, dev_end, retrain_start, retrain_end,
                   holdout_start, holdout_end, [name]
    clf          : sklearn-compatible classifier (will be cloned internally)
    method       : one of "cpcv" | "wf" | "kfold"

    Returns
    -------
    metrics : dict  (output of compute_metrics)
    fig     : matplotlib.figure.Figure
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y, t1, prices, fwd_ret = load_asset(
        ticker,
        timeline_cfg["download_start"],
        timeline_cfg["download_end"],
    )

    # ── 2. Slice into dev / retrain / holdout windows ─────────────────────────
    X_dev, y_dev, t1_dev, fwd_ret_dev = slice_by_dates(
        X, y, t1, fwd_ret,
        timeline_cfg["dev_start"], timeline_cfg["dev_end"])

    X_ret, y_ret, _, fwd_ret_ret = slice_by_dates(
        X, y, t1, fwd_ret,
        timeline_cfg["retrain_start"], timeline_cfg["retrain_end"])

    X_ho, y_ho, _, fwd_ret_ho = slice_by_dates(
        X, y, t1, fwd_ret,
        timeline_cfg["holdout_start"], timeline_cfg["holdout_end"])

    clf = clone(clf)

    # ── 3. Hold-out metrics ───────────────────────────────────────────────────
    ho_sr  = holdout_sharpe(clf, X_ret, y_ret, X_ho, fwd_ret_ho)
    ho_pnl = _holdout_pnl(clf, X_ret, y_ret, X_ho, fwd_ret_ho)

    # ── 4. Validation Sharpe distribution + val PnL per method ───────────────
    if method == "cpcv":
        val_sharpes = cpcv_sharpe_dist(
            clf, X_dev, y_dev, t1_dev, fwd_ret_dev,
            n_groups=N_GROUPS, k_test=K_TEST,
            pct_embargo=PCT_EMBARGO, variant="purge_embargo",
        )
        val_pnl     = _try_cpcv_val_pnl(clf, X_dev, y_dev, t1_dev, fwd_ret_dev)
        method_label = "CPCV"

    elif method == "wf":
        # WF Fold 1 train window starts at wf_start (8m before dev_start).
        # We must pass the full dataset (from wf_start onward) so the splitter
        # finds data in that pre-dev window; X_dev only starts at dev_start.
        X_wf, y_wf, t1_wf, fwd_ret_wf = slice_by_dates(
            X, y, t1, fwd_ret,
            timeline_cfg["wf_start"], timeline_cfg["dev_end"])
        val_sharpes = wf_rolling_sharpe_dist(
            clf, X_wf, y_wf, t1_wf, fwd_ret_wf,
            wf_start=timeline_cfg["wf_start"],
            dev_start=timeline_cfg["dev_start"],
            dev_end=timeline_cfg["dev_end"],
            pct_embargo=PCT_EMBARGO,
        )
        val_pnl     = _try_wf_val_pnl(clf, X_wf, y_wf, t1_wf, fwd_ret_wf, timeline_cfg)
        method_label = "WF rolling 8:4"

    elif method == "kfold":
        val_sharpes = kfold_sharpe_dist(
            clf, X_dev, y_dev, t1_dev, fwd_ret_dev,
            variant="purge_embargo", n_splits=6, pct_embargo=PCT_EMBARGO,
        )
        val_pnl     = _try_kfold_val_pnl(clf, X_dev, y_dev, t1_dev, fwd_ret_dev)
        method_label = "KFold purged+embargo"

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: cpcv | wf | kfold")

    # ── 5. Compute metrics and plot ───────────────────────────────────────────
    metrics = compute_metrics(val_sharpes, ho_sr, val_pnl, ho_pnl)

    tl_name = timeline_cfg.get("name", "")
    fig = _method_vs_holdout_plot(
        val_sharpes=val_sharpes,
        ho_sr=ho_sr,
        prices_full=prices,
        timeline_cfg=timeline_cfg,
        method_label=method_label,
        fig_title=f"{ticker} | {tl_name} | {method_label}",
    )

    return metrics, fig


def _run_experiment_from_arrays(
    X, y, t1, prices, fwd_ret,
    timeline_cfg: dict,
    clf,
    method: str,
) -> tuple:
    """Same as run_experiment but accepts pre-loaded arrays instead of ticker+download."""
    from sklearn.base import clone as sk_clone

    X_dev, y_dev, t1_dev, fwd_ret_dev = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["dev_start"], timeline_cfg["dev_end"])
    X_ret, y_ret, t1_ret, fwd_ret_ret = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["retrain_start"], timeline_cfg["retrain_end"])
    X_ho, y_ho, t1_ho, fwd_ret_ho = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["holdout_start"], timeline_cfg["holdout_end"])

    clf = sk_clone(clf)
    ho_sr  = holdout_sharpe(clf, X_ret, y_ret, X_ho, fwd_ret_ho)
    ho_pnl = _holdout_pnl(clf, X_ret, y_ret, X_ho, fwd_ret_ho)

    if method == "cpcv":
        val_sharpes = cpcv_sharpe_dist(
            clf, X_dev, y_dev, t1_dev, fwd_ret_dev,
            n_groups=N_GROUPS, k_test=K_TEST, pct_embargo=PCT_EMBARGO, variant="purge_embargo")
        val_pnl = _try_cpcv_val_pnl(clf, X_dev, y_dev, t1_dev, fwd_ret_dev)
        method_label = "CPCV"
    elif method == "wf":
        X_wf, y_wf, t1_wf, fwd_ret_wf = slice_by_dates(
            X, y, t1, fwd_ret,
            timeline_cfg["wf_start"], timeline_cfg["dev_end"])
        val_sharpes = wf_rolling_sharpe_dist(
            clf, X_wf, y_wf, t1_wf, fwd_ret_wf,
            wf_start=timeline_cfg["wf_start"], dev_start=timeline_cfg["dev_start"],
            dev_end=timeline_cfg["dev_end"], pct_embargo=PCT_EMBARGO)
        val_pnl = _try_wf_val_pnl(clf, X_wf, y_wf, t1_wf, fwd_ret_wf, timeline_cfg)
        method_label = "WF rolling 8:4"
    elif method == "kfold":
        val_sharpes = kfold_sharpe_dist(
            clf, X_dev, y_dev, t1_dev, fwd_ret_dev,
            variant="purge_embargo", n_splits=6, pct_embargo=PCT_EMBARGO)
        val_pnl = _try_kfold_val_pnl(clf, X_dev, y_dev, t1_dev, fwd_ret_dev)
        method_label = "KFold purged+embargo"
    else:
        raise ValueError(f"Unknown method '{method}'")

    metrics = compute_metrics(val_sharpes, ho_sr, val_pnl, ho_pnl)
    tl_name = timeline_cfg.get("name", "synthetic")
    fig = _method_vs_holdout_plot(
        val_sharpes=val_sharpes, ho_sr=ho_sr, prices_full=prices,
        timeline_cfg=timeline_cfg, method_label=method_label,
        fig_title=f"Synthetic | {tl_name} | {method_label}")
    return metrics, fig


def run_experiment_full(ticker: str, timeline_cfg: dict, clf) -> dict:
    """
    Run all three methods for one real dataset and return fold/path summaries.
    """
    from sklearn.base import clone as sk_clone

    X, y, t1, prices, fwd_ret = load_asset(
        ticker, timeline_cfg["download_start"], timeline_cfg["download_end"])

    X_dev, y_dev, t1_dev, fwd_ret_dev = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["dev_start"], timeline_cfg["dev_end"])
    X_ret, y_ret, _, fwd_ret_ret = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["retrain_start"], timeline_cfg["retrain_end"])
    X_ho, y_ho, _, fwd_ret_ho = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["holdout_start"], timeline_cfg["holdout_end"])
    X_wf, y_wf, t1_wf, fwd_ret_wf = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["wf_start"], timeline_cfg["dev_end"])

    clf = sk_clone(clf)
    ho_sr = holdout_sharpe(clf, X_ret, y_ret, X_ho, fwd_ret_ho)
    ho_pnl = _holdout_pnl(clf, X_ret, y_ret, X_ho, fwd_ret_ho)

    _, oos_by_split, _, _ = _build_cpcv_splits_table(
        clf, X_dev, y_dev, t1_dev, fwd_ret_dev)
    cpcv_fold_srs = []
    for sid in sorted(oos_by_split.keys()):
        pnl = oos_by_split[sid]
        if len(pnl) >= 2 and pnl.std() > 0:
            cpcv_fold_srs.append(float(np.sqrt(252) * pnl.mean() / pnl.std()))
        else:
            cpcv_fold_srs.append(0.0)

    paths = get_paths(N_GROUPS, K_TEST)
    cpcv_path_srs = []
    for split_ids in paths:
        valid = [sid for sid in split_ids if sid in oos_by_split]
        if not valid:
            continue
        path_pnl = pd.concat([oos_by_split[sid] for sid in valid]).sort_index()
        if len(path_pnl) >= 2 and path_pnl.std() > 0:
            cpcv_path_srs.append(float(np.sqrt(252) * path_pnl.mean() / path_pnl.std()))
        else:
            cpcv_path_srs.append(0.0)

    cpcv_val_sharpes = pd.Series(cpcv_path_srs, name="cpcv_paths")
    cpcv_val_pnl = _try_cpcv_val_pnl(clf, X_dev, y_dev, t1_dev, fwd_ret_dev)
    metrics_cpcv = compute_metrics(cpcv_val_sharpes, ho_sr, cpcv_val_pnl, ho_pnl)

    fold_dates = _wf_rolling_fold_dates(
        wf_start=timeline_cfg["wf_start"],
        dev_start=timeline_cfg["dev_start"],
        dev_end=timeline_cfg["dev_end"],
    )
    splitter_wf = RollingWalkForwardCV(fold_dates, t1=None, pctEmbargo=0.0)
    wf_fold_srs = []
    wf_pieces = []
    for train_idx, test_idx in splitter_wf.split(X_wf):
        _, oos_pnl, _, _ = _pnl_from_split(
            clf, X_wf, y_wf, t1_wf, fwd_ret_wf, train_idx, test_idx)
        wf_pieces.append(oos_pnl)
        if len(oos_pnl) >= 2 and oos_pnl.std() > 0:
            wf_fold_srs.append(float(np.sqrt(252) * oos_pnl.mean() / oos_pnl.std()))
        else:
            wf_fold_srs.append(0.0)
    wf_concat = pd.concat(wf_pieces).sort_index() if wf_pieces else pd.Series(dtype=float)
    wf_sr = (
        float(np.sqrt(252) * wf_concat.mean() / wf_concat.std())
        if len(wf_concat) >= 2 and wf_concat.std() > 0 else 0.0
    )
    metrics_wf = compute_metrics(pd.Series([wf_sr], name="wf_concat"), ho_sr, wf_concat, ho_pnl)

    splitter_kf = KFold(n_splits=3, shuffle=False)
    kfold_fold_srs = []
    kf_pieces = []
    for train_idx, test_idx in splitter_kf.split(X_dev):
        _, oos_pnl, _, _ = _pnl_from_split(
            clf, X_dev, y_dev, t1_dev, fwd_ret_dev, train_idx, test_idx)
        kf_pieces.append(oos_pnl)
        if len(oos_pnl) >= 2 and oos_pnl.std() > 0:
            kfold_fold_srs.append(float(np.sqrt(252) * oos_pnl.mean() / oos_pnl.std()))
        else:
            kfold_fold_srs.append(0.0)
    kf_concat = pd.concat(kf_pieces).sort_index() if kf_pieces else pd.Series(dtype=float)
    kfold_sr = (
        float(np.sqrt(252) * kf_concat.mean() / kf_concat.std())
        if len(kf_concat) >= 2 and kf_concat.std() > 0 else 0.0
    )
    metrics_kfold = compute_metrics(
        pd.Series([kfold_sr], name="kfold_concat"), ho_sr, kf_concat, ho_pnl)

    return dict(
        cpcv_fold_srs=cpcv_fold_srs,
        wf_fold_srs=wf_fold_srs,
        kfold_fold_srs=kfold_fold_srs,
        cpcv_path_srs=cpcv_path_srs,
        wf_sr=wf_sr,
        kfold_sr=kfold_sr,
        holdout_sr=ho_sr,
        metrics_cpcv=metrics_cpcv,
        metrics_wf=metrics_wf,
        metrics_kfold=metrics_kfold,
    )


def run_experiment_full_from_arrays(X, y, t1, prices, fwd_ret, timeline_cfg: dict, clf) -> dict:
    """
    Same as run_experiment_full but accepts pre-loaded arrays.
    """
    from sklearn.base import clone as sk_clone

    X_dev, y_dev, t1_dev, fwd_ret_dev = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["dev_start"], timeline_cfg["dev_end"])
    X_ret, y_ret, _, fwd_ret_ret = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["retrain_start"], timeline_cfg["retrain_end"])
    X_ho, y_ho, _, fwd_ret_ho = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["holdout_start"], timeline_cfg["holdout_end"])
    X_wf, y_wf, t1_wf, fwd_ret_wf = slice_by_dates(
        X, y, t1, fwd_ret, timeline_cfg["wf_start"], timeline_cfg["dev_end"])

    clf = sk_clone(clf)
    ho_sr = holdout_sharpe(clf, X_ret, y_ret, X_ho, fwd_ret_ho)
    ho_pnl = _holdout_pnl(clf, X_ret, y_ret, X_ho, fwd_ret_ho)

    _, oos_by_split, _, _ = _build_cpcv_splits_table(
        clf, X_dev, y_dev, t1_dev, fwd_ret_dev)
    cpcv_fold_srs = []
    for sid in sorted(oos_by_split.keys()):
        pnl = oos_by_split[sid]
        if len(pnl) >= 2 and pnl.std() > 0:
            cpcv_fold_srs.append(float(np.sqrt(252) * pnl.mean() / pnl.std()))
        else:
            cpcv_fold_srs.append(0.0)

    paths = get_paths(N_GROUPS, K_TEST)
    cpcv_path_srs = []
    for split_ids in paths:
        valid = [sid for sid in split_ids if sid in oos_by_split]
        if not valid:
            continue
        path_pnl = pd.concat([oos_by_split[sid] for sid in valid]).sort_index()
        if len(path_pnl) >= 2 and path_pnl.std() > 0:
            cpcv_path_srs.append(float(np.sqrt(252) * path_pnl.mean() / path_pnl.std()))
        else:
            cpcv_path_srs.append(0.0)

    cpcv_val_sharpes = pd.Series(cpcv_path_srs, name="cpcv_paths")
    cpcv_val_pnl = _try_cpcv_val_pnl(clf, X_dev, y_dev, t1_dev, fwd_ret_dev)
    metrics_cpcv = compute_metrics(cpcv_val_sharpes, ho_sr, cpcv_val_pnl, ho_pnl)

    fold_dates = _wf_rolling_fold_dates(
        wf_start=timeline_cfg["wf_start"],
        dev_start=timeline_cfg["dev_start"],
        dev_end=timeline_cfg["dev_end"],
    )
    splitter_wf = RollingWalkForwardCV(fold_dates, t1=None, pctEmbargo=0.0)
    wf_fold_srs = []
    wf_pieces = []
    for train_idx, test_idx in splitter_wf.split(X_wf):
        _, oos_pnl, _, _ = _pnl_from_split(
            clf, X_wf, y_wf, t1_wf, fwd_ret_wf, train_idx, test_idx)
        wf_pieces.append(oos_pnl)
        if len(oos_pnl) >= 2 and oos_pnl.std() > 0:
            wf_fold_srs.append(float(np.sqrt(252) * oos_pnl.mean() / oos_pnl.std()))
        else:
            wf_fold_srs.append(0.0)
    wf_concat = pd.concat(wf_pieces).sort_index() if wf_pieces else pd.Series(dtype=float)
    wf_sr = (
        float(np.sqrt(252) * wf_concat.mean() / wf_concat.std())
        if len(wf_concat) >= 2 and wf_concat.std() > 0 else 0.0
    )
    metrics_wf = compute_metrics(pd.Series([wf_sr], name="wf_concat"), ho_sr, wf_concat, ho_pnl)

    splitter_kf = KFold(n_splits=3, shuffle=False)
    kfold_fold_srs = []
    kf_pieces = []
    for train_idx, test_idx in splitter_kf.split(X_dev):
        _, oos_pnl, _, _ = _pnl_from_split(
            clf, X_dev, y_dev, t1_dev, fwd_ret_dev, train_idx, test_idx)
        kf_pieces.append(oos_pnl)
        if len(oos_pnl) >= 2 and oos_pnl.std() > 0:
            kfold_fold_srs.append(float(np.sqrt(252) * oos_pnl.mean() / oos_pnl.std()))
        else:
            kfold_fold_srs.append(0.0)
    kf_concat = pd.concat(kf_pieces).sort_index() if kf_pieces else pd.Series(dtype=float)
    kfold_sr = (
        float(np.sqrt(252) * kf_concat.mean() / kf_concat.std())
        if len(kf_concat) >= 2 and kf_concat.std() > 0 else 0.0
    )
    metrics_kfold = compute_metrics(
        pd.Series([kfold_sr], name="kfold_concat"), ho_sr, kf_concat, ho_pnl)

    return dict(
        cpcv_fold_srs=cpcv_fold_srs,
        wf_fold_srs=wf_fold_srs,
        kfold_fold_srs=kfold_fold_srs,
        cpcv_path_srs=cpcv_path_srs,
        wf_sr=wf_sr,
        kfold_sr=kfold_sr,
        holdout_sr=ho_sr,
        metrics_cpcv=metrics_cpcv,
        metrics_wf=metrics_wf,
        metrics_kfold=metrics_kfold,
    )
