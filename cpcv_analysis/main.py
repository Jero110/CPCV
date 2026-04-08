"""
Orchestrates the full CPCV analysis. Run with:
    conda run -n rappi python3 -m cpcv_analysis.main
"""
import os
import pandas as pd
from xgboost import XGBClassifier

from cpcv_analysis.config import (
    N_GROUPS, K_TEST, PCT_EMBARGO, XGB_PARAMS, CRASH_START, CRASH_DURATION,
    LEAKAGE_FEATURE_NAME,
)
from cpcv_analysis.data import load_data, inject_leakage, build_features
from cpcv_analysis.splitters import CombinatorialPurgedKFold
from cpcv_analysis.cv_runner import run_cpcv
from cpcv_analysis.comparison import run_all_methods
from cpcv_analysis import plots


def _run_scenario_leakage(label: str, prices, plot_subdir: str):
    """
    Scenario C: run comparison of all methods under clean vs leaked features.
    Leaked: one feature column = next period's label (perfect future leakage).
    """
    os.makedirs(plot_subdir, exist_ok=True)

    X, y, t1, _ = build_features(prices)
    fwd_ret = None
    X_leaked = inject_leakage(X, y, feature_name=LEAKAGE_FEATURE_NAME)

    print(f"\n[{label}] Running all methods on CLEAN features...")
    comparison_clean, _ = run_all_methods(X, y, t1, fwd_ret=fwd_ret)

    print(f"\n[{label}] Running all methods on LEAKED features...")
    comparison_leaked, _ = run_all_methods(X_leaked, y, t1, fwd_ret=fwd_ret)

    print(f"\n[{label}] Generating leakage comparison plot → {plot_subdir}")
    plots.plot_leakage_comparison(comparison_clean, comparison_leaked, out_dir=plot_subdir)

    # Summary
    print(f"\n[{label}] Leakage impact summary:")
    print(f"  {'Method':<28}  {'OOS_SR clean':>12}  {'OOS_SR leaked':>13}  {'Delta':>7}")
    print("  " + "-" * 68)
    for method in comparison_clean.index:
        sr_c = comparison_clean.loc[method, "OOS_SR"]
        sr_l = comparison_leaked.loc[method, "OOS_SR"]
        delta = sr_l - sr_c
        flag = " ← EXPLOITS LEAKAGE" if delta > 0.1 else ""
        print(f"  {method:<28}  {sr_c:>12.3f}  {sr_l:>13.3f}  {delta:>+7.3f}{flag}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  CPCV Tesis Analysis")
    print("=" * 60)

    # ── 1. Data ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    X, y, t1, prices = load_data()

    # ── 2. CPCV run ────────────────────────────────────────────────────────
    print("\n[2/5] Running CPCV...")
    clf  = XGBClassifier(**XGB_PARAMS)
    cpcv = CombinatorialPurgedKFold(N_GROUPS, K_TEST, t1, PCT_EMBARGO)
    fold_results, path_results, oos_by_split = run_cpcv(clf, X, y, t1, cpcv)

    # Build split_table for split matrix plot
    split_table = pd.DataFrame([
        {"split_id": fr["fold_id"], "test_groups": fr["test_groups"]}
        for fr in fold_results
    ])

    # ── 3. Comparison ──────────────────────────────────────────────────────
    print("\n[3/5] Running all 9 methods comparison...")
    comparison_df, _ = run_all_methods(X, y, t1)

    # ── 4. Plots ───────────────────────────────────────────────────────────
    print("\n[4/5] Generating plots...")

    # Crash window end index for shading
    crash_idx = prices.index.searchsorted(pd.Timestamp(CRASH_START))
    crash_end = min(crash_idx + CRASH_DURATION, len(prices) - 1)

    plots.plot_spy_prices(prices, CRASH_START, crash_end)
    plots.plot_split_matrix(split_table, N_GROUPS)
    plots.plot_is_oos_per_split(fold_results)
    plots.plot_equity_curves(path_results)
    plots.plot_comparison_sharpe(comparison_df)
    plots.plot_comparison_delta(comparison_df)
    plots.plot_comparison_accuracy_f1(comparison_df)
    plots.plot_comparison_return_pct(comparison_df)
    plots.plot_comparison_heatmap(comparison_df)
    plots.plot_oos_degradation(fold_results)
    plots.plot_rank_logits(comparison_df)

    # ── 5. Scenario C: leakage detection ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  SCENARIO C: Leakage detection (clean vs leaked features)")
    print("=" * 60)
    _run_scenario_leakage("Scenario C", prices, "plots/C_leakage")

    # ── 6. Summary ─────────────────────────────────────────────────────────
    print("\n[6/6] Summary")
    print("=" * 60)
    cpcv_is  = pd.DataFrame(fold_results)["is_sharpe"].mean()
    cpcv_oos = pd.DataFrame(path_results)["sharpe"].mean()
    print(f"  CPCV IS  SR (mean splits): {cpcv_is:.4f}")
    print(f"  CPCV OOS SR (mean paths):  {cpcv_oos:.4f}")
    print(f"  Delta SR:                  {cpcv_is - cpcv_oos:.4f}")
    print("\nAll plots saved to plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
