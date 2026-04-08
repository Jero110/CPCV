# cpcv_analysis/plots.py
"""
All plotting functions. Minimalist style: light gray grid, single accent, clean type.
Each function saves a PNG and returns the fig.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from cpcv_analysis.config import PLOT_DIR, FIGSIZE

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
ACCENT  = "#2171b5"
GRAY    = "#aaaaaa"
RED     = "#e45756"
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#eeeeee",
    "grid.linewidth":     0.6,
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     10,
})


def _save(fig, name: str):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plots] Saved → {path}")
    plt.close(fig)


def plot_spy_prices(prices: pd.DataFrame, crash_start: str, crash_end_idx: int):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(prices.index, prices["Close"], color=ACCENT, lw=1.2)
    ax.axvspan(pd.Timestamp(crash_start), prices.index[crash_end_idx],
               alpha=0.15, color=RED, label="Synthetic crash")
    ax.set_title("SPY Close — with synthetic crash")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    _save(fig, "01_spy_prices.png")


def plot_split_matrix(split_table: pd.DataFrame, N: int):
    """Heatmap of CPCV split assignments — preserved from original notebook."""
    matrix = np.zeros((len(split_table), N), dtype=int)
    for i, tg in enumerate(split_table["test_groups"]):
        for g in tg:
            matrix[i, g] = 1

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"G{g}" for g in range(N)])
    ax.set_yticks(range(len(split_table)))
    ax.set_yticklabels([f"Split {i}" for i in range(len(split_table))])
    ax.set_title("CPCV — Split × Group matrix (blue = test)")
    plt.colorbar(im, ax=ax, shrink=0.6)
    _save(fig, "02_split_matrix.png")


def plot_is_oos_per_split(fold_results: list):
    """IS vs OOS Sharpe bars per split — preserved from original notebook."""
    df  = pd.DataFrame([{"split": f["fold_id"], "IS": f["is_sharpe"],
                          "OOS": f["sharpe"]} for f in fold_results])
    x   = np.arange(len(df))
    w   = 0.35
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - w/2, df["IS"],  w, label="IS",  color=ACCENT, alpha=0.8)
    ax.bar(x + w/2, df["OOS"], w, label="OOS", color=RED,    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Split {i}" for i in df["split"]], rotation=45, ha="right")
    ax.set_title("IS vs OOS Sharpe per CPCV split")
    ax.set_ylabel("Annualized Sharpe")
    ax.legend()
    _save(fig, "03_is_oos_per_split.png")


def plot_equity_curves(path_results: list):
    """Cumulative equity curves per path."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors  = plt.cm.tab10(np.linspace(0, 1, len(path_results)))
    for pr, col in zip(path_results, colors):
        eq = np.exp(pr["_path_pnl"].sort_index().cumsum())
        ax.plot(eq.index, eq.values, lw=1.3,
                label=f"Path {pr['path_id']} (SR={pr['sharpe']:.2f})", color=col)
    ax.axhline(1.0, color=GRAY, ls="--", lw=0.8)
    ax.set_title("CPCV — Cumulative equity curves by path (OOS)")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=9)
    _save(fig, "04_equity_curves.png")


def plot_comparison_sharpe(comparison_df: pd.DataFrame):
    methods = comparison_df.index.tolist()
    x       = np.arange(len(methods))
    w       = 0.35
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - w/2, comparison_df["IS_SR"],  w, label="IS",  color=ACCENT, alpha=0.8)
    ax.bar(x + w/2, comparison_df["OOS_SR"], w, label="OOS", color=RED,    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title("IS vs OOS Sharpe — all methods")
    ax.set_ylabel("Annualized Sharpe")
    ax.legend()
    _save(fig, "05_comparison_sharpe.png")


def plot_comparison_delta(comparison_df: pd.DataFrame):
    methods = comparison_df.index.tolist()
    colors  = [RED if v > 0 else ACCENT for v in comparison_df["Delta_SR"]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(methods, comparison_df["Delta_SR"], color=colors)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title("Delta SR (IS − OOS) — higher = more overfit")
    ax.set_ylabel("Delta Sharpe")
    _save(fig, "06_comparison_delta_sr.png")


def plot_comparison_accuracy_f1(comparison_df: pd.DataFrame):
    methods = comparison_df.index.tolist()
    x       = np.arange(len(methods))
    w       = 0.35
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - w/2, comparison_df["accuracy"], w, label="Accuracy", color=ACCENT, alpha=0.8)
    ax.bar(x + w/2, comparison_df["f1"],       w, label="F1",       color=RED,    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title("OOS Accuracy and F1 — all methods")
    ax.legend()
    _save(fig, "07_comparison_accuracy_f1.png")


def plot_comparison_return_pct(comparison_df: pd.DataFrame):
    methods = comparison_df.index.tolist()
    colors  = [ACCENT if v >= 0 else RED for v in comparison_df["return_pct"]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(methods, comparison_df["return_pct"], color=colors)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title("OOS Return % — all methods")
    ax.set_ylabel("Return %")
    _save(fig, "08_comparison_return_pct.png")


def plot_comparison_heatmap(comparison_df: pd.DataFrame):
    cols   = ["IS_SR", "OOS_SR", "Delta_SR", "accuracy", "f1", "return_pct"]
    data   = comparison_df[cols].astype(float)
    norm   = (data - data.min()) / (data.max() - data.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(norm.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)
    for i in range(len(data)):
        for j, col in enumerate(cols):
            ax.text(j, i, f"{data.iloc[i][col]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if norm.values[i, j] > 0.6 else "black")
    ax.set_title("Methods × Metrics heatmap (normalized color scale)")
    plt.colorbar(im, ax=ax, shrink=0.5)
    _save(fig, "09_comparison_heatmap.png")


def plot_oos_degradation(fold_results: list):
    from cpcv_analysis.advanced_analysis import oos_degradation
    df  = oos_degradation(fold_results)
    x, y = df["IS_SR"].values, df["OOS_SR"].values

    if len(x) < 2:
        print("[plots] Not enough folds for degradation plot, skipping.")
        return

    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    y_ci   = se * 1.96 * np.sqrt(1/len(x) + (x_line - x.mean())**2 / np.sum((x - x.mean())**2))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, color=ACCENT, alpha=0.7, zorder=3)
    ax.plot(x_line, y_line, color=RED, lw=1.5, label=f"Regression (R²={r**2:.2f})")
    ax.fill_between(x_line, y_line - y_ci, y_line + y_ci, alpha=0.15, color=RED)
    ax.axline((0, 0), slope=1, color=GRAY, ls="--", lw=0.8, label="IS=OOS line")
    ax.set_xlabel("IS Sharpe")
    ax.set_ylabel("OOS Sharpe")
    ax.set_title("OOS Performance Degradation")
    ax.legend()
    _save(fig, "10_oos_degradation.png")


def plot_rank_logits(comparison_df: pd.DataFrame):
    from cpcv_analysis.advanced_analysis import rank_logits
    logits, prob_overfit, names = rank_logits(comparison_df)

    mu, sigma = stats.norm.fit(logits)
    x_curve   = np.linspace(logits.min() - 1, logits.max() + 1, 200)
    y_curve   = stats.norm.pdf(x_curve, mu, sigma)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(logits, bins=min(10, len(logits)), color=ACCENT, alpha=0.7,
            density=True, label="Rank logits")
    ax.plot(x_curve, y_curve, color=RED, lw=2, label="Fitted normal")
    ax.axvline(0, color=GRAY, ls="--", lw=1)
    ax.fill_between(x_curve, y_curve, where=(x_curve < 0),
                    alpha=0.2, color=RED, label=f"Prob[Overfit] = {prob_overfit:.2f}")
    ax.set_xlabel("Logit of rank")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Rank Logits — Probability of Backtest Overfit")
    ax.legend()
    _save(fig, "11_rank_logits.png")


# ── 12 Leakage comparison ─────────────────────────────────────────────────────
def plot_leakage_comparison(comparison_clean: pd.DataFrame,
                            comparison_leaked: pd.DataFrame,
                            out_dir: str = "plots/"):
    """
    Side-by-side bar chart: OOS Sharpe and OOS Accuracy for each method,
    clean features vs leaked features.

    comparison_clean, comparison_leaked: DataFrames with index=method,
    columns including OOS_SR and accuracy (output of run_all_methods).
    """
    methods = comparison_clean.index.tolist()
    x = np.arange(len(methods))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Panel 1: OOS Sharpe
    ax1.bar(x - w/2, comparison_clean["OOS_SR"].values,  w,
            label="Clean features", color=ACCENT, alpha=0.85, zorder=3)
    ax1.bar(x + w/2, comparison_leaked["OOS_SR"].values, w,
            label="Leaked features", color=RED,  alpha=0.85, zorder=3)
    ax1.axhline(0, color=GRAY, lw=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax1.set_title("OOS Sharpe — Clean vs Leaked Features\n"
                  "(KFold should jump up; CPCV should stay flat)")
    ax1.set_ylabel("Ann. Sharpe Ratio (OOS)")
    ax1.legend(fontsize=8)

    # Panel 2: OOS Accuracy
    ax2.bar(x - w/2, comparison_clean["accuracy"].values,  w,
            label="Clean features", color=ACCENT, alpha=0.85, zorder=3)
    ax2.bar(x + w/2, comparison_leaked["accuracy"].values, w,
            label="Leaked features", color=RED,  alpha=0.85, zorder=3)
    ax2.axhline(0.5, color=GRAY, ls="--", lw=0.8, label="Random baseline (50%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax2.set_title("OOS Accuracy — Clean vs Leaked Features\n"
                  "(methods without purge should exploit leakage)")
    ax2.set_ylabel("OOS Accuracy")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.legend(fontsize=8)

    fig.suptitle("Scenario C — Feature Leakage Detection\n"
                 "KFold exploits future label; CPCV is robust via purge + embargo",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    name = "12_leakage_comparison.png"
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plots] Saved → {path}")
    plt.close(fig)
