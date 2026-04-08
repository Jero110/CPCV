# cpcv_analysis/comparison.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold as SklearnKFold
from xgboost import XGBClassifier
from cpcv_analysis.config import N_GROUPS, K_TEST, PCT_EMBARGO, XGB_PARAMS
from cpcv_analysis.splitters import PurgedKFold, CombinatorialPurgedKFold, WalkForwardCV
from cpcv_analysis.cv_runner import cvScore, run_cpcv, _metrics_from_pnl


def _fresh_clf():
    return XGBClassifier(**XGB_PARAMS)


def _summarize(fold_results: list, method_name: str) -> dict:
    """Aggregate fold results into a single comparison row."""
    folds = [f for f in fold_results if "accuracy" in f]
    if not folds:
        return {"method": method_name, "IS_SR": np.nan, "OOS_SR": np.nan,
                "Delta_SR": np.nan, "accuracy": np.nan, "f1": np.nan,
                "return_pct": np.nan, "mean_return_pct": np.nan, "ann_return_pct": np.nan,
                "max_drawdown_pct": np.nan, "calmar": np.nan,
                "hit_ratio": np.nan, "profit_factor": np.nan,
                "volatility_pct": np.nan}
    return {
        "method":     method_name,
        "IS_SR":      float(np.mean([f["is_sharpe"]               for f in folds])),
        "OOS_SR":     float(np.mean([f["sharpe"]                  for f in folds])),
        "Delta_SR":   float(np.mean([f["is_sharpe"] - f["sharpe"] for f in folds])),
        "accuracy":   float(np.mean([f["accuracy"]                for f in folds])),
        "f1":         float(np.mean([f["f1"]                      for f in folds])),
        "return_pct": float(np.mean([f["return_pct"]              for f in folds])),
        "mean_return_pct": float(np.mean([f["mean_return_pct"]    for f in folds])),
        "ann_return_pct": float(np.mean([f["ann_return_pct"]      for f in folds])),
        "max_drawdown_pct": float(np.mean([f["max_drawdown_pct"]  for f in folds])),
        "calmar": float(np.mean([f["calmar"]                      for f in folds])),
        "hit_ratio": float(np.mean([f["hit_ratio"]                for f in folds])),
        "profit_factor": float(np.mean([f["profit_factor"]        for f in folds])),
        "volatility_pct": float(np.mean([f["volatility_pct"]      for f in folds])),
    }


def _run_kfold_nopurge(X, y, n_splits, fwd_ret=None):
    """Standard KFold (no purge, no embargo) using sklearn directly."""
    clf = _fresh_clf()
    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(SklearnKFold(n_splits=n_splits, shuffle=False).split(X)):
        if len(train_idx) < 10 or len(test_idx) < 5:
            continue
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]
        clf.fit(X_tr, y_tr)
        y_hat_tr = clf.predict(X_tr)
        y_hat    = clf.predict(X_te)
        ret_tr   = fwd_ret.iloc[train_idx].values if fwd_ret is not None else y_tr.values
        ret_te   = fwd_ret.iloc[test_idx].values  if fwd_ret is not None else y_te.values
        pnl_tr   = pd.Series((2*y_hat_tr-1)*ret_tr, index=X_tr.index).astype(float)
        pnl      = pd.Series((2*y_hat-1)*ret_te,    index=X_te.index).astype(float)
        is_metrics = _metrics_from_pnl(pnl_tr)
        oos_metrics = _metrics_from_pnl(pnl)
        folds.append({
            "fold_id": fold_id,
            "accuracy": float(accuracy_score(y_te, y_hat)),
            "f1": float(f1_score(y_te, y_hat, zero_division=0)),
            "return_pct": oos_metrics["return_pct"],
            "mean_return_pct": oos_metrics["mean_return_pct"],
            "ann_return_pct": oos_metrics["ann_return_pct"],
            "sharpe": oos_metrics["sharpe"],
            "max_drawdown_pct": oos_metrics["max_drawdown_pct"],
            "calmar": oos_metrics["calmar"],
            "hit_ratio": oos_metrics["hit_ratio"],
            "profit_factor": oos_metrics["profit_factor"],
            "volatility_pct": oos_metrics["volatility_pct"],
            "is_sharpe": is_metrics["sharpe"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })
    return folds


def _run_ccv_nopurge(X, y, n_groups, k, fwd_ret=None):
    """CCV: combinatorial splits but using raw_train_idx (no purge)."""
    dummy_t1 = pd.Series(X.index, index=X.index)
    splitter = CombinatorialPurgedKFold(n_groups, k, dummy_t1, pctEmbargo=0.0)
    clf = _fresh_clf()
    folds = []
    for split_id, (raw_tr, test_idx, final_tr, test_groups) in enumerate(splitter.split(X)):
        use_tr = raw_tr
        if len(use_tr) < 10 or len(test_idx) < 5:
            continue
        X_tr, y_tr = X.iloc[use_tr], y.iloc[use_tr]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        clf.fit(X_tr, y_tr)
        y_hat_tr = clf.predict(X_tr)
        y_hat    = clf.predict(X_te)
        ret_tr   = fwd_ret.iloc[use_tr].values   if fwd_ret is not None else y_tr.values
        ret_te   = fwd_ret.iloc[test_idx].values if fwd_ret is not None else y_te.values
        pnl_tr   = pd.Series((2*y_hat_tr-1)*ret_tr, index=X_tr.index).astype(float)
        pnl      = pd.Series((2*y_hat-1)*ret_te,    index=X_te.index).astype(float)
        is_metrics = _metrics_from_pnl(pnl_tr)
        oos_metrics = _metrics_from_pnl(pnl)
        folds.append({
            "fold_id": split_id,
            "accuracy": float(accuracy_score(y_te, y_hat)),
            "f1": float(f1_score(y_te, y_hat, zero_division=0)),
            "return_pct": oos_metrics["return_pct"],
            "mean_return_pct": oos_metrics["mean_return_pct"],
            "ann_return_pct": oos_metrics["ann_return_pct"],
            "sharpe": oos_metrics["sharpe"],
            "max_drawdown_pct": oos_metrics["max_drawdown_pct"],
            "calmar": oos_metrics["calmar"],
            "hit_ratio": oos_metrics["hit_ratio"],
            "profit_factor": oos_metrics["profit_factor"],
            "volatility_pct": oos_metrics["volatility_pct"],
            "is_sharpe": is_metrics["sharpe"],
            "n_train": len(use_tr),
            "n_test": len(test_idx),
        })
    return folds


def run_all_methods(X, y, t1, fwd_ret=None):
    """
    Run all 9 validation methods on (X, y, t1).
    fwd_ret: actual forward log-returns; if provided, PnL is economically meaningful.
    Returns:
        comparison_df  — one row per method (IS_SR, OOS_SR, Delta_SR, accuracy, f1, return_pct)
        all_folds_df   — one row per (method, fold) with fold-level metrics; used for rank logits
    """
    rows = []
    all_folds = []

    def _collect(folds, method):
        rows.append(_summarize(folds, method))
        for enum_id, f in enumerate(folds):
            if "accuracy" in f:
                # Use fold_id (temporal index) as trial_id so that folds from
                # different methods with the same trial_id correspond to the
                # same time period — required for correct rank logit computation.
                fold_id = f.get("fold_id", enum_id)
                all_folds.append({
                    "method":   method,
                    "fold_id":  fold_id,
                    "trial_id": fold_id,
                    "IS_SR":    f["is_sharpe"],
                    "OOS_SR":   f["sharpe"],
                })

    # ── KFold variants ──────────────────────────────────────────────────────
    print("[comparison] Running KFold (no purge)...")
    _collect(_run_kfold_nopurge(X, y, N_GROUPS, fwd_ret), "KFold")

    print("[comparison] Running KFold+Purge...")
    _collect(cvScore(_fresh_clf(), X, y, t1,
                     PurgedKFold(n_splits=N_GROUPS, t1=t1, pctEmbargo=0.0),
                     fwd_ret=fwd_ret), "KFold+Purge")

    print("[comparison] Running KFold+Purge+Embargo...")
    _collect(cvScore(_fresh_clf(), X, y, t1,
                     PurgedKFold(n_splits=N_GROUPS, t1=t1, pctEmbargo=PCT_EMBARGO),
                     fwd_ret=fwd_ret), "KFold+Purge+Embargo")

    # ── WalkForward variants ────────────────────────────────────────────────
    print("[comparison] Running WalkForward (no purge)...")
    _collect(cvScore(_fresh_clf(), X, y, t1,
                     WalkForwardCV(n_splits=N_GROUPS, t1=None, pctEmbargo=0.0),
                     fwd_ret=fwd_ret), "WalkForward")

    print("[comparison] Running WalkForward+Purge...")
    _collect(cvScore(_fresh_clf(), X, y, t1,
                     WalkForwardCV(n_splits=N_GROUPS, t1=t1, pctEmbargo=0.0),
                     fwd_ret=fwd_ret), "WalkForward+Purge")

    print("[comparison] Running WalkForward+Purge+Embargo...")
    _collect(cvScore(_fresh_clf(), X, y, t1,
                     WalkForwardCV(n_splits=N_GROUPS, t1=t1, pctEmbargo=PCT_EMBARGO),
                     fwd_ret=fwd_ret), "WalkForward+Purge+Embargo")

    # ── CCV variants ────────────────────────────────────────────────────────
    print("[comparison] Running CCV (no purge)...")
    _collect(_run_ccv_nopurge(X, y, N_GROUPS, K_TEST, fwd_ret), "CCV")

    print("[comparison] Running CCV+Purge...")
    cpcv_purge = CombinatorialPurgedKFold(N_GROUPS, K_TEST, t1, pctEmbargo=0.0)
    fold_res_purge, _, _ = run_cpcv(_fresh_clf(), X, y, t1, cpcv_purge, verbose=False, fwd_ret=fwd_ret)
    _collect(fold_res_purge, "CCV+Purge")

    print("[comparison] Running CPCV (purge + embargo)...")
    cpcv_full = CombinatorialPurgedKFold(N_GROUPS, K_TEST, t1, pctEmbargo=PCT_EMBARGO)
    fold_res_full, _, _ = run_cpcv(_fresh_clf(), X, y, t1, cpcv_full, verbose=False, fwd_ret=fwd_ret)
    _collect(fold_res_full, "CPCV")

    comparison_df = pd.DataFrame(rows).set_index("method")
    all_folds_df  = pd.DataFrame(all_folds)
    print("\n[comparison] Done.")
    print(comparison_df.round(4).to_string())
    return comparison_df, all_folds_df
