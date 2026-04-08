# cpcv_analysis/cv_runner.py
"""
cvScore: extended from De Prado (2018) base.
get_paths: reconstructs CPCV paths from split assignments.
run_cpcv: full CPCV run returning fold_results + path_results.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score
from cpcv_analysis.config import N_GROUPS, K_TEST


def _annualized_sr(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe ratio from a return series."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * returns.mean() / returns.std())


def _annualized_return_pct(returns: pd.Series, periods: int = 252) -> float:
    """Annualized return (%) from additive log-return series."""
    if len(returns) == 0:
        return 0.0
    mean_log_ret = float(returns.mean())
    return float((np.exp(mean_log_ret * periods) - 1.0) * 100.0)


def _cumulative_return_pct(returns: pd.Series) -> float:
    """Cumulative return (%) from additive log-return series."""
    if len(returns) == 0:
        return 0.0
    return float((np.exp(float(returns.sum())) - 1.0) * 100.0)


def _mean_return_pct(returns: pd.Series) -> float:
    """Mean return (%) per observation from additive log-return series."""
    if len(returns) == 0:
        return 0.0
    return float(returns.mean() * 100.0)


def _max_drawdown_pct(returns: pd.Series) -> float:
    """Max drawdown (%) from additive log-return series."""
    if len(returns) == 0:
        return 0.0
    equity = np.exp(returns.cumsum())
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min() * 100.0)


def _profit_factor(returns: pd.Series) -> float:
    """Profit factor = gross profits / gross losses."""
    if len(returns) == 0:
        return 0.0
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(-returns[returns < 0].sum())
    if gross_loss == 0:
        return 10.0 if gross_profit > 0 else 0.0
    return float(min(gross_profit / gross_loss, 10.0))


def _hit_ratio(returns: pd.Series) -> float:
    """Fraction of positive strategy returns."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def _volatility_pct(returns: pd.Series, periods: int = 252) -> float:
    """Annualized volatility (%) from additive log-return series."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(periods) * 100.0)


def _metrics_from_pnl(pnl: pd.Series) -> dict:
    """Core performance metrics used throughout the analysis."""
    ann_return = _annualized_return_pct(pnl)
    max_dd = _max_drawdown_pct(pnl)
    return {
        "return_pct": _cumulative_return_pct(pnl),
        "mean_return_pct": _mean_return_pct(pnl),
        "ann_return_pct": ann_return,
        "sharpe": _annualized_sr(pnl),
        "max_drawdown_pct": max_dd,
        "calmar": float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0,
        "hit_ratio": _hit_ratio(pnl),
        "profit_factor": _profit_factor(pnl),
        "volatility_pct": _volatility_pct(pnl),
    }


def cvScore(clf, X: pd.DataFrame, y: pd.Series, t1: pd.Series,
            splitter, sample_weight: pd.Series = None,
            fwd_ret: pd.Series = None) -> list:
    """
    Extended cvScore based on De Prado (2018).
    Works with PurgedKFold, WalkForwardCV (yields train_idx, test_idx).

    fwd_ret: actual forward log-returns aligned with X.index.
             If provided, PnL = sign(prediction) * fwd_ret (economically meaningful).
             If None, PnL = sign(prediction) * label (±1, unit-less).

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

        # IS PnL
        y_hat_tr  = clf.predict(X_tr)
        signs_tr  = (2 * y_hat_tr - 1)
        ret_tr    = fwd_ret.iloc[train_idx].values if fwd_ret is not None else y_tr.values
        pnl_tr    = pd.Series(signs_tr * ret_tr, index=X_tr.index).astype(float)
        is_acc    = float(accuracy_score(y_tr, y_hat_tr))
        is_f1     = float(f1_score(y_tr, y_hat_tr, zero_division=0))
        is_metrics = _metrics_from_pnl(pnl_tr)

        # OOS PnL
        y_hat   = clf.predict(X_te)
        signs   = (2 * y_hat - 1)
        ret_te  = fwd_ret.iloc[test_idx].values if fwd_ret is not None else y_te.values
        pnl     = pd.Series(signs * ret_te, index=X_te.index).astype(float)

        # Uncomment for verbose per-fold debug:
        # for i, (pred, true, r) in enumerate(zip(y_hat, y_te.values, pnl.values)):
        #     print(f"    fold {fold_id} obs {i}: pred={pred} true={true} pnl={r:.4f}")

        oos_metrics = _metrics_from_pnl(pnl)
        results.append({
            "fold_id": fold_id,
            "is_accuracy": is_acc,
            "is_f1": is_f1,
            "is_return_pct": is_metrics["return_pct"],
            "is_mean_return_pct": is_metrics["mean_return_pct"],
            "is_ann_return_pct": is_metrics["ann_return_pct"],
            "is_sharpe": is_metrics["sharpe"],
            "is_max_drawdown_pct": is_metrics["max_drawdown_pct"],
            "is_calmar": is_metrics["calmar"],
            "is_hit_ratio": is_metrics["hit_ratio"],
            "is_profit_factor": is_metrics["profit_factor"],
            "is_volatility_pct": is_metrics["volatility_pct"],
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
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "_oos_pnl": pnl,
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
             cpcv_splitter, verbose: bool = True,
             fwd_ret: pd.Series = None) -> tuple:
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
            print(f"  [CPCV] split {split_id}: skipped (too few obs)")
            continue

        X_tr, y_tr = X.iloc[final_tr], y.iloc[final_tr]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

        clf.fit(X_tr, y_tr)

        y_hat_tr  = clf.predict(X_tr)
        signs_tr  = (2 * y_hat_tr - 1)
        ret_tr    = fwd_ret.iloc[final_tr].values if fwd_ret is not None else y_tr.values
        pnl_tr    = pd.Series(signs_tr * ret_tr, index=X_tr.index).astype(float)
        is_acc    = float(accuracy_score(y_tr, y_hat_tr))
        is_f1     = float(f1_score(y_tr, y_hat_tr, zero_division=0))
        is_metrics = _metrics_from_pnl(pnl_tr)

        y_hat   = clf.predict(X_te)
        signs   = (2 * y_hat - 1)
        ret_te  = fwd_ret.iloc[test_idx].values if fwd_ret is not None else y_te.values
        pnl     = pd.Series(signs * ret_te, index=X_te.index).astype(float)

        # Verbose per-observation debug (uncomment to see every prediction):
        # for i, (pred, true, r) in enumerate(zip(y_hat, y_te.values, pnl.values)):
        #     print(f"    split {split_id} | {X_te.index[i].date()} | pred={pred} true={true} pnl={r:+.4f}")

        oos_metrics = _metrics_from_pnl(pnl)
        fold_results.append({
            "fold_id": split_id,
            "test_groups": test_groups,
            "is_accuracy": is_acc,
            "is_f1": is_f1,
            "is_return_pct": is_metrics["return_pct"],
            "is_mean_return_pct": is_metrics["mean_return_pct"],
            "is_ann_return_pct": is_metrics["ann_return_pct"],
            "is_sharpe": is_metrics["sharpe"],
            "is_max_drawdown_pct": is_metrics["max_drawdown_pct"],
            "is_calmar": is_metrics["calmar"],
            "is_hit_ratio": is_metrics["hit_ratio"],
            "is_profit_factor": is_metrics["profit_factor"],
            "is_volatility_pct": is_metrics["volatility_pct"],
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
            "n_train": len(final_tr),
            "n_test": len(test_idx),
            "_oos_pnl": pnl,
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
            "path_id": pid,
            "split_ids": valid_ids,
            "is_accuracy": float(np.mean([fold_map[s]["is_accuracy"] for s in valid_ids])),
            "is_f1": float(np.mean([fold_map[s]["is_f1"] for s in valid_ids])),
            "is_return_pct": float(np.mean([fold_map[s]["is_return_pct"] for s in valid_ids])),
            "is_mean_return_pct": float(np.mean([fold_map[s]["is_mean_return_pct"] for s in valid_ids])),
            "is_ann_return_pct": float(np.mean([fold_map[s]["is_ann_return_pct"] for s in valid_ids])),
            "is_sharpe": float(np.mean([fold_map[s]["is_sharpe"] for s in valid_ids])),
            "is_max_drawdown_pct": float(np.mean([fold_map[s]["is_max_drawdown_pct"] for s in valid_ids])),
            "is_calmar": float(np.mean([fold_map[s]["is_calmar"] for s in valid_ids])),
            "is_hit_ratio": float(np.mean([fold_map[s]["is_hit_ratio"] for s in valid_ids])),
            "is_profit_factor": float(np.mean([fold_map[s]["is_profit_factor"] for s in valid_ids])),
            "is_volatility_pct": float(np.mean([fold_map[s]["is_volatility_pct"] for s in valid_ids])),
            "accuracy": float(np.mean([fold_map[s]["accuracy"] for s in valid_ids])),
            "f1": float(np.mean([fold_map[s]["f1"] for s in valid_ids])),
            "return_pct": _cumulative_return_pct(path_pnl),
            "mean_return_pct": _mean_return_pct(path_pnl),
            "ann_return_pct": _annualized_return_pct(path_pnl),
            "sharpe": _annualized_sr(path_pnl),   # De Prado: SR of full path series
            "max_drawdown_pct": _max_drawdown_pct(path_pnl),
            "calmar": float(_annualized_return_pct(path_pnl) / abs(_max_drawdown_pct(path_pnl)))
                      if _max_drawdown_pct(path_pnl) < 0 else 0.0,
            "hit_ratio": _hit_ratio(path_pnl),
            "profit_factor": _profit_factor(path_pnl),
            "volatility_pct": _volatility_pct(path_pnl),
            "_path_pnl": path_pnl,
        })

    if verbose:
        # ── Split summary table ───────────────────────────────────────────────
        print("\n[CPCV] ── SPLIT METRICS ──────────────────────────────────────────────────────────")
        print(f"  {'Split':>5}  {'Groups':<10}  {'n_train':>7}  {'n_test':>6}  "
              f"{'IS_SR':>7}  {'OOS_SR':>7}  {'IS_MDD':>8}  {'OOS_MDD':>8}  "
              f"{'IS_Hit':>7}  {'OOS_Hit':>7}  {'IS_PF':>6}  {'OOS_PF':>6}  "
              f"{'Date range'}")
        print("  " + "-" * 134)
        for fr in fold_results:
            pnl_s = oos_by_split[fr["fold_id"]]
            print(f"  {fr['fold_id']:>5}  {str(fr['test_groups']):<10}  "
                  f"{fr['n_train']:>7}  {fr['n_test']:>6}  "
                  f"{fr['is_sharpe']:>7.3f}  {fr['sharpe']:>7.3f}  "
                  f"{fr['is_max_drawdown_pct']:>7.2f}%  {fr['max_drawdown_pct']:>8.2f}%  "
                  f"{fr['is_hit_ratio']:>7.2%}  {fr['hit_ratio']:>7.2%}  "
                  f"{fr['is_profit_factor']:>6.2f}  {fr['profit_factor']:>6.2f}  "
                  f"{pnl_s.index[0].date()} → {pnl_s.index[-1].date()}")

        # ── Path summary table ────────────────────────────────────────────────
        print("\n[CPCV] ── PATH METRICS ───────────────────────────────────────────────────────────")
        print(f"  {'Path':>4}  {'Splits':<14}  {'IS_SR':>7}  {'OOS_SR':>7}  "
              f"{'IS_MDD':>8}  {'OOS_MDD':>8}  {'IS_Hit':>7}  {'OOS_Hit':>7}  "
              f"{'IS_PF':>6}  {'OOS_PF':>6}")
        print("  " + "-" * 96)
        for pr in path_results:
            print(f"  {pr['path_id']:>4}  {str(pr['split_ids']):<14}  "
                  f"{pr['is_sharpe']:>7.3f}  {pr['sharpe']:>7.3f}  "
                  f"{pr['is_max_drawdown_pct']:>7.2f}%  {pr['max_drawdown_pct']:>8.2f}%  "
                  f"{pr['is_hit_ratio']:>7.2%}  {pr['hit_ratio']:>7.2%}  "
                  f"{pr['is_profit_factor']:>6.2f}  {pr['profit_factor']:>6.2f}")
        print()

    return fold_results, path_results, oos_by_split
