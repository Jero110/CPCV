"""
Orchestrates the full CPCV analysis. Run with:
    conda run -n rappi python3 -m cpcv_analysis.main

Two scenarios:
  Scenario A — SPY 2024 clean (no synthetic crash)
  Scenario B — SPY 2024 with synthetic 20% crash injected
  Scenario C — Feature leakage detection (clean vs leaked features)
"""
import os
import pandas as pd
from xgboost import XGBClassifier

from cpcv_analysis.config import (
    N_GROUPS, K_TEST, PCT_EMBARGO, XGB_PARAMS, CRASH_START, CRASH_DURATION,
    LEAKAGE_FEATURE_NAME,
)
from cpcv_analysis.data import download_prices, inject_crash, build_features, inject_leakage
from cpcv_analysis.splitters import CombinatorialPurgedKFold
from cpcv_analysis.cv_runner import run_cpcv
from cpcv_analysis.comparison import run_all_methods
from cpcv_analysis import plots


def _run_scenario(label: str, prices, plot_subdir: str):
    """Run full analysis for one price scenario. Saves plots to plot_subdir."""
    os.makedirs(plot_subdir, exist_ok=True)

    X, y, t1, _, fwd_ret = build_features(prices)

    # ── CPCV ──────────────────────────────────────────────────────────────────
    print(f"\n[{label}] Running CPCV...")
    clf  = XGBClassifier(**XGB_PARAMS)
    cpcv = CombinatorialPurgedKFold(N_GROUPS, K_TEST, t1, PCT_EMBARGO)
    fold_results, path_results, oos_by_split = run_cpcv(clf, X, y, t1, cpcv, fwd_ret=fwd_ret)

    split_table = pd.DataFrame([
        {"split_id": fr["fold_id"], "test_groups": fr["test_groups"]}
        for fr in fold_results
    ])

    # ── Comparison ────────────────────────────────────────────────────────────
    print(f"\n[{label}] Running all 9 methods comparison...")
    comparison_df, all_folds_df, method_folds = run_all_methods(
        X, y, t1, fwd_ret=fwd_ret, return_method_folds=True
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n[{label}] Generating plots → {plot_subdir}")

    crash_idx = prices.index.searchsorted(pd.Timestamp(CRASH_START))
    crash_end = min(crash_idx + CRASH_DURATION, len(prices) - 1)

    plots.plot_spy_prices(prices, CRASH_START, crash_end,
                          X=X, N_groups=N_GROUPS, out_dir=plot_subdir,
                          highlight_groups_on_price=True)
    plots.plot_split_matrix(split_table, N_GROUPS, out_dir=plot_subdir)
    plots.plot_path_example(fold_results, path_results, N_GROUPS, path_id=0, out_dir=plot_subdir)
    plots.plot_is_oos_per_split(fold_results, out_dir=plot_subdir)
    plots.plot_metrics_per_path(path_results, out_dir=plot_subdir)
    plots.plot_equity_curves(path_results, out_dir=plot_subdir)
    plots.plot_comparison_metrics(comparison_df, out_dir=plot_subdir)
    plots.plot_comparison_delta(comparison_df, out_dir=plot_subdir)
    plots.plot_comparison_heatmap(comparison_df, out_dir=plot_subdir)
    plots.plot_method_distributions(all_folds_df, method_folds=method_folds, out_dir=plot_subdir)
    plots.plot_model_validation_map(comparison_df, out_dir=plot_subdir)
    plots.plot_walkforward_purge_debug(prices, X, t1, fold_id=2, n_splits=N_GROUPS, out_dir=plot_subdir)
    plots.plot_oos_degradation(fold_results, all_folds_df, out_dir=plot_subdir)
    plots.plot_rank_logits(all_folds_df, out_dir=plot_subdir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[{label}] Summary")
    print("=" * 60)
    cpcv_is  = pd.DataFrame(fold_results)["is_sharpe"].mean()
    cpcv_oos = pd.DataFrame(path_results)["sharpe"].mean()
    print(f"  CPCV IS  SR (mean splits): {cpcv_is:.4f}")
    print(f"  CPCV OOS SR (mean paths):  {cpcv_oos:.4f}")
    print(f"  Delta SR:                  {cpcv_is - cpcv_oos:.4f}")
    print(f"  Plots saved to: {plot_subdir}")
    print("=" * 60)


def _run_scenario_leakage(label: str, prices, plot_subdir: str):
    """
    Scenario C: run comparison of all methods under clean vs leaked features.
    Leaked: one feature column = next period's label (perfect future leakage).
    """
    os.makedirs(plot_subdir, exist_ok=True)

    X, y, t1, _, fwd_ret = build_features(prices)
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
        flag = " ← LEAKAGE INFLATES OOS" if delta > 1.0 else ""
        print(f"  {method:<28}  {sr_c:>12.3f}  {sr_l:>13.3f}  {delta:>+7.3f}{flag}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  CPCV Tesis Analysis — Scenarios A, B & C")
    print("=" * 60)

    # Download once
    print("\n[0] Downloading SPY prices...")
    prices_clean = download_prices()

    # Scenario A: clean SPY
    print("\n" + "=" * 60)
    print("  SCENARIO A: SPY 2024 (no crash)")
    print("=" * 60)
    _run_scenario("Scenario A", prices_clean, "plots/A_clean")

    # Scenario B: with synthetic crash
    print("\n" + "=" * 60)
    print("  SCENARIO B: SPY 2024 + synthetic crash (−20%)")
    print("=" * 60)
    prices_crash = inject_crash(prices_clean)
    _run_scenario("Scenario B", prices_crash, "plots/B_crash")

    # Scenario C: leakage detection
    print("\n" + "=" * 60)
    print("  SCENARIO C: Leakage detection (clean vs leaked features)")
    print("=" * 60)
    _run_scenario_leakage("Scenario C", prices_clean, "plots/C_leakage")

    print("\nDone. All plots saved to plots/A_clean/, plots/B_crash/, plots/C_leakage/")


if __name__ == "__main__":
    main()
