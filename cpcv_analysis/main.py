"""
Orchestrates the full CPCV analysis. Run with:
    conda run -n rappi python3 -m cpcv_analysis.main
"""
import pandas as pd
from xgboost import XGBClassifier

from cpcv_analysis.config import N_GROUPS, K_TEST, PCT_EMBARGO, XGB_PARAMS, CRASH_START, CRASH_DURATION
from cpcv_analysis.data import load_data
from cpcv_analysis.splitters import CombinatorialPurgedKFold
from cpcv_analysis.cv_runner import run_cpcv
from cpcv_analysis.comparison import run_all_methods
from cpcv_analysis import plots


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
    comparison_df = run_all_methods(X, y, t1)

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

    # ── 5. Summary ─────────────────────────────────────────────────────────
    print("\n[5/5] Summary")
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
