# cpcv_analysis/advanced_analysis.py
"""
Advanced analyses from De Prado (2018) cap. 11-12, implemented from scratch.

  oos_degradation(fold_results)   → DataFrame with IS_SR, OOS_SR per fold
  rank_logits(comparison_df)      → (logits array, prob_overfit float, method_names list)
"""
import numpy as np
import pandas as pd
from scipy import stats


def oos_degradation(fold_results: list) -> pd.DataFrame:
    """
    Build IS vs OOS Sharpe DataFrame from fold_results list.
    Works with output of cvScore or run_cpcv fold_results.
    """
    rows = []
    for f in fold_results:
        rows.append({
            "fold_id":  f.get("fold_id", len(rows)),
            "IS_SR":    f["is_sharpe"],
            "OOS_SR":   f["sharpe"],
            "Delta_SR": f["is_sharpe"] - f["sharpe"],
        })
    return pd.DataFrame(rows)


def rank_logits(comparison_df: pd.DataFrame) -> tuple:
    """
    Compute rank logits and Prob[Overfit] from comparison_df.

    For each method, rank its OOS_SR among all methods (1 = worst, N = best).
    Logit = log(rank / (N - rank)).
    Prob[Overfit] = fraction of logits < 0 (area to left of 0 under fitted normal).

    Returns:
        logits        — np.array of logit values (one per method)
        prob_overfit  — float, estimated from fitted normal CDF
        method_names  — list of method names aligned with logits
    """
    oos_srs      = comparison_df["OOS_SR"].values
    n            = len(oos_srs)
    ranks        = stats.rankdata(oos_srs)  # 1 = worst

    # Clip to avoid log(0)
    ranks_clipped = np.clip(ranks, 0.5, n - 0.5)
    logits        = np.log(ranks_clipped / (n - ranks_clipped))

    # Fit normal to logits
    mu, sigma    = stats.norm.fit(logits)
    prob_overfit = float(stats.norm.cdf(0, loc=mu, scale=sigma))

    print(f"[advanced] Rank logits: mu={mu:.3f}, sigma={sigma:.3f}")
    print(f"[advanced] Prob[Overfit] = {prob_overfit:.3f}")

    return logits, prob_overfit, list(comparison_df.index)
