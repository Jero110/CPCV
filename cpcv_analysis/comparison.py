# cpcv_analysis/comparison.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold as SklearnKFold
from xgboost import XGBClassifier
from cpcv_analysis.config import N_GROUPS, K_TEST, PCT_EMBARGO, XGB_PARAMS
from cpcv_analysis.splitters import PurgedKFold, CombinatorialPurgedKFold, WalkForwardCV
from cpcv_analysis.cv_runner import cvScore, run_cpcv, _annualized_sr


def _fresh_clf():
    return XGBClassifier(**XGB_PARAMS)


def _summarize(fold_results: list, method_name: str) -> dict:
    """Aggregate fold results into a single comparison row."""
    folds = [f for f in fold_results if "accuracy" in f]
    if not folds:
        return {"method": method_name, "IS_SR": np.nan, "OOS_SR": np.nan,
                "Delta_SR": np.nan, "accuracy": np.nan, "f1": np.nan, "return_pct": np.nan}
    return {
        "method":     method_name,
        "IS_SR":      float(np.mean([f["is_sharpe"]               for f in folds])),
        "OOS_SR":     float(np.mean([f["sharpe"]                  for f in folds])),
        "Delta_SR":   float(np.mean([f["is_sharpe"] - f["sharpe"] for f in folds])),
        "accuracy":   float(np.mean([f["accuracy"]                for f in folds])),
        "f1":         float(np.mean([f["f1"]                      for f in folds])),
        "return_pct": float(np.mean([f["return_pct"]              for f in folds])),
    }


def _run_kfold_nopurge(X, y, n_splits):
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
        pnl_tr   = pd.Series((2*y_hat_tr-1)*y_tr.values, index=X_tr.index).astype(float)
        pnl      = pd.Series((2*y_hat-1)*y_te.values,    index=X_te.index).astype(float)
        folds.append({
            "fold_id":    fold_id,
            "accuracy":   float(accuracy_score(y_te, y_hat)),
            "f1":         float(f1_score(y_te, y_hat, zero_division=0)),
            "return_pct": float(pnl.sum()),
            "sharpe":     _annualized_sr(pnl),
            "is_sharpe":  _annualized_sr(pnl_tr),
            "n_train":    len(train_idx),
            "n_test":     len(test_idx),
        })
    return folds


def _run_ccv_nopurge(X, y, n_groups, k):
    """CCV: combinatorial splits but using raw_train_idx (no purge)."""
    # Use a dummy t1 (just X.index itself) so splitter initializes, but we'll use raw_train_idx
    dummy_t1 = pd.Series(X.index, index=X.index)
    splitter = CombinatorialPurgedKFold(n_groups, k, dummy_t1, pctEmbargo=0.0)
    clf = _fresh_clf()
    folds = []
    for split_id, (raw_tr, test_idx, final_tr, test_groups) in enumerate(splitter.split(X)):
        # Use raw_train_idx — no purge
        use_tr = raw_tr
        if len(use_tr) < 10 or len(test_idx) < 5:
            continue
        X_tr, y_tr = X.iloc[use_tr], y.iloc[use_tr]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        clf.fit(X_tr, y_tr)
        y_hat_tr = clf.predict(X_tr)
        y_hat    = clf.predict(X_te)
        pnl_tr   = pd.Series((2*y_hat_tr-1)*y_tr.values, index=X_tr.index).astype(float)
        pnl      = pd.Series((2*y_hat-1)*y_te.values,    index=X_te.index).astype(float)
        folds.append({
            "fold_id":    split_id,
            "accuracy":   float(accuracy_score(y_te, y_hat)),
            "f1":         float(f1_score(y_te, y_hat, zero_division=0)),
            "return_pct": float(pnl.sum()),
            "sharpe":     _annualized_sr(pnl),
            "is_sharpe":  _annualized_sr(pnl_tr),
            "n_train":    len(use_tr),
            "n_test":     len(test_idx),
        })
    return folds


def run_all_methods(X, y, t1, fwd_ret=None) -> pd.DataFrame:
    """
    Run all 9 validation methods on (X, y, t1).
    Returns comparison_df: one row per method, columns = IS_SR, OOS_SR, Delta_SR, accuracy, f1, return_pct.
    """
    rows = []

    # ── KFold variants ──────────────────────────────────────────────────────
    print("[comparison] Running KFold (no purge)...")
    rows.append(_summarize(_run_kfold_nopurge(X, y, N_GROUPS), "KFold"))

    print("[comparison] Running KFold+Purge...")
    rows.append(_summarize(
        cvScore(_fresh_clf(), X, y, t1,
                PurgedKFold(n_splits=N_GROUPS, t1=t1, pctEmbargo=0.0)),
        "KFold+Purge"))

    print("[comparison] Running KFold+Purge+Embargo...")
    rows.append(_summarize(
        cvScore(_fresh_clf(), X, y, t1,
                PurgedKFold(n_splits=N_GROUPS, t1=t1, pctEmbargo=PCT_EMBARGO)),
        "KFold+Purge+Embargo"))

    # ── WalkForward variants ────────────────────────────────────────────────
    print("[comparison] Running WalkForward (no purge)...")
    rows.append(_summarize(
        cvScore(_fresh_clf(), X, y, t1,
                WalkForwardCV(n_splits=N_GROUPS, t1=None, pctEmbargo=0.0)),
        "WalkForward"))

    print("[comparison] Running WalkForward+Purge...")
    rows.append(_summarize(
        cvScore(_fresh_clf(), X, y, t1,
                WalkForwardCV(n_splits=N_GROUPS, t1=t1, pctEmbargo=0.0)),
        "WalkForward+Purge"))

    print("[comparison] Running WalkForward+Purge+Embargo...")
    rows.append(_summarize(
        cvScore(_fresh_clf(), X, y, t1,
                WalkForwardCV(n_splits=N_GROUPS, t1=t1, pctEmbargo=PCT_EMBARGO)),
        "WalkForward+Purge+Embargo"))

    # ── CCV variants ────────────────────────────────────────────────────────
    print("[comparison] Running CCV (no purge)...")
    rows.append(_summarize(_run_ccv_nopurge(X, y, N_GROUPS, K_TEST), "CCV"))

    print("[comparison] Running CCV+Purge...")
    cpcv_purge = CombinatorialPurgedKFold(N_GROUPS, K_TEST, t1, pctEmbargo=0.0)
    fold_res_purge, _, _ = run_cpcv(_fresh_clf(), X, y, t1, cpcv_purge)
    rows.append(_summarize(fold_res_purge, "CCV+Purge"))

    print("[comparison] Running CPCV (purge + embargo)...")
    cpcv_full = CombinatorialPurgedKFold(N_GROUPS, K_TEST, t1, pctEmbargo=PCT_EMBARGO)
    fold_res_full, _, _ = run_cpcv(_fresh_clf(), X, y, t1, cpcv_full)
    rows.append(_summarize(fold_res_full, "CPCV"))

    comparison_df = pd.DataFrame(rows).set_index("method")
    print("\n[comparison] Done.")
    print(comparison_df.round(4).to_string())
    return comparison_df, comparison_df
