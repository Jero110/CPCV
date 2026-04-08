# cpcv_analysis/cv_runner.py
"""
cvScore: extended from De Prado (2018) base.
get_paths: reconstructs CPCV paths from split assignments.
run_cpcv: full CPCV run returning fold_results + path_results.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from math import comb
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier
from cpcv_analysis.config import N_GROUPS, K_TEST, XGB_PARAMS


def _annualized_sr(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe ratio from a return series."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * returns.mean() / returns.std())


def cvScore(clf, X: pd.DataFrame, y: pd.Series, t1: pd.Series,
            splitter, sample_weight: pd.Series = None) -> list:
    """
    Extended cvScore based on De Prado (2018).
    Works with PurgedKFold, WalkForwardCV (yields train_idx, test_idx).

    Returns list of dicts, one per fold:
        fold_id, accuracy, f1, return_pct, sharpe, is_sharpe, n_train, n_test
    """
    results = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X)):
        if len(train_idx) < 10 or len(test_idx) < 5:
            print(f"  [cvScore] fold {fold_id}: skipped (too few obs)")
            continue

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]
        sw_tr = sample_weight.iloc[train_idx] if sample_weight is not None else None

        clf.fit(X_tr, y_tr, sample_weight=sw_tr)

        # IS Sharpe (simple: sign prediction on train)
        y_hat_tr = clf.predict(X_tr)
        pnl_tr   = pd.Series((2 * y_hat_tr - 1) * y_tr.values,
                              index=X_tr.index).astype(float)
        is_sr    = _annualized_sr(pnl_tr)

        # OOS predictions
        y_hat = clf.predict(X_te)
        pnl   = pd.Series((2 * y_hat - 1) * y_te.values,
                          index=X_te.index).astype(float)

        results.append({
            "fold_id":    fold_id,
            "accuracy":   float(accuracy_score(y_te, y_hat)),
            "f1":         float(f1_score(y_te, y_hat, zero_division=0)),
            "return_pct": float(pnl.sum()),
            "sharpe":     _annualized_sr(pnl),
            "is_sharpe":  is_sr,
            "n_train":    len(train_idx),
            "n_test":     len(test_idx),
            # store OOS pnl series for path reconstruction
            "_oos_pnl":   pnl,
        })

    return results


def get_paths(N: int = N_GROUPS, k: int = K_TEST) -> list:
    """
    Returns phi = C(N,k)*k/N paths where:
      - Each path is a list of N/k split_ids covering all N groups exactly once
      - No split_id is shared between paths (paths are disjoint)

    Algorithm:
      1. Enumerate all valid single-path partitions: sets of N/k combos
         (splits) whose groups cover {0..N-1} exactly once.
      2. From those, find a collection of n_paths partitions that are mutually
         disjoint (no shared split index).
    """
    all_combos  = list(combinations(range(N), k))
    combo_to_id = {c: i for i, c in enumerate(all_combos)}
    n_paths     = len(all_combos) * k // N   # phi
    splits_per_path = N // k

    # Step 1: enumerate all valid single-path partitions
    # A valid partition = a set of splits_per_path combos whose groups partition {0..N-1}
    def find_partitions(remaining_groups, min_combo_idx):
        """Yield lists of combo-indices that partition remaining_groups."""
        if not remaining_groups:
            yield []
            return
        for i in range(min_combo_idx, len(all_combos)):
            combo = all_combos[i]
            if all(g in remaining_groups for g in combo):
                for rest in find_partitions(remaining_groups - set(combo), i + 1):
                    yield [i] + rest

    all_partitions = list(find_partitions(set(range(N)), 0))

    # Step 2: greedily select n_paths disjoint partitions
    selected    = []
    used_splits = set()
    for partition in all_partitions:
        p_set = frozenset(partition)
        if not (p_set & used_splits):
            selected.append(sorted(partition))
            used_splits |= p_set
            if len(selected) == n_paths:
                break

    return selected


def run_cpcv(clf, X: pd.DataFrame, y: pd.Series, t1: pd.Series,
             cpcv_splitter, verbose: bool = True, fwd_ret=None) -> tuple:
    """
    Full CPCV run.
    Returns:
        fold_results  — list of per-fold dicts (same schema as cvScore)
        path_results  — list of per-path dicts
        oos_by_split  — dict {split_id: pnl Series} for equity curves
    """
    fold_results  = []
    oos_by_split  = {}
    split_groups  = []  # track test_groups per split for path assignment

    for split_id, (raw_tr, test_idx, final_tr, test_groups) in enumerate(
            cpcv_splitter.split(X)):

        if len(final_tr) < 10 or len(test_idx) < 5:
            if verbose:
                print(f"  [CPCV] split {split_id}: skipped (too few obs)")
            continue

        X_tr, y_tr = X.iloc[final_tr], y.iloc[final_tr]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

        clf.fit(X_tr, y_tr)

        y_hat_tr = clf.predict(X_tr)
        pnl_tr   = pd.Series((2 * y_hat_tr - 1) * y_tr.values,
                              index=X_tr.index).astype(float)
        is_sr    = _annualized_sr(pnl_tr)

        y_hat = clf.predict(X_te)
        pnl   = pd.Series((2 * y_hat - 1) * y_te.values,
                          index=X_te.index).astype(float)

        fold_results.append({
            "fold_id":     split_id,
            "test_groups": test_groups,
            "accuracy":    float(accuracy_score(y_te, y_hat)),
            "f1":          float(f1_score(y_te, y_hat, zero_division=0)),
            "return_pct":  float(pnl.sum()),
            "sharpe":      _annualized_sr(pnl),
            "is_sharpe":   is_sr,
            "n_train":     len(final_tr),
            "n_test":      len(test_idx),
            "_oos_pnl":    pnl,
        })
        oos_by_split[split_id] = pnl
        split_groups.append((split_id, test_groups))

    # ── Build paths ──────────────────────────────────────────────────────────
    paths = get_paths(N_GROUPS, K_TEST)
    path_results = []

    if verbose:
        print("\n[CPCV] Path assignment:")
    for pid, split_ids in enumerate(paths):
        # Map split_ids to actual fold results (by fold_id)
        fold_map = {fr["fold_id"]: fr for fr in fold_results}
        valid_ids = [sid for sid in split_ids if sid in fold_map]

        if not valid_ids:
            continue

        # Concatenate OOS returns for this path
        path_pnl = pd.concat([oos_by_split[sid] for sid in valid_ids]).sort_index()

        # Print path composition for debuggability
        if verbose:
            for sid in valid_ids:
                fr = fold_map[sid]
                pnl_s = oos_by_split[sid]
                print(f"  Path {pid} | split {sid} | groups {fr['test_groups']} | "
                      f"dates {pnl_s.index[0].date()} → {pnl_s.index[-1].date()} | "
                      f"sharpe={fr['sharpe']:.3f}")

        path_results.append({
            "path_id":    pid,
            "split_ids":  valid_ids,
            "accuracy":   float(np.mean([fold_map[s]["accuracy"]   for s in valid_ids])),
            "f1":         float(np.mean([fold_map[s]["f1"]         for s in valid_ids])),
            "return_pct": float(np.mean([fold_map[s]["return_pct"] for s in valid_ids])),
            "sharpe":     _annualized_sr(path_pnl),   # De Prado: SR of full path series
            "_path_pnl":  path_pnl,
        })

    return fold_results, path_results, oos_by_split
