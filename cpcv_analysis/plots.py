# cpcv_analysis/plots.py
"""
All plotting functions. Thesis-level minimalist style.
Each function accepts out_dir and saves a numbered PNG there.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from cpcv_analysis.config import FIGSIZE

# ── Global style ───────────────────────────────────────────────────────────────
BLUE   = "#1a6faf"
RED    = "#c0392b"
GRAY   = "#888888"
LGRAY  = "#e8e8e8"
BLACK  = "#1a1a1a"

plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    True,
    "axes.spines.bottom":  True,
    "axes.edgecolor":      "#cccccc",
    "axes.linewidth":      0.8,
    "axes.grid":           True,
    "grid.color":          LGRAY,
    "grid.linewidth":      0.5,
    "grid.alpha":          1.0,
    "font.family":         "sans-serif",
    "font.size":           11,
    "axes.titlesize":      13,
    "axes.titleweight":    "bold",
    "axes.labelsize":      10,
    "axes.labelcolor":     BLACK,
    "xtick.color":         GRAY,
    "ytick.color":         GRAY,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.frameon":      False,
    "legend.fontsize":     9,
})


def _save(fig, name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[plots] Saved → {path}")
    plt.close(fig)


# ── 01 SPY Prices + CPCV group overlay ───────────────────────────────────────
def plot_spy_prices(prices, crash_start: str, crash_end_idx: int,
                   X=None, N_groups: int = 6, out_dir="plots/",
                   highlight_groups_on_price: bool = False):
    """
    SPY Close with synthetic crash window shaded and optional CPCV fold overlay.
    """
    has_groups = X is not None

    fig, ax_price = plt.subplots(figsize=FIGSIZE if not has_groups else (13, 6))

    # ── Top: price ────────────────────────────────────────────────────────────
    ax_price.plot(prices.index, prices["Close"], color=BLUE, lw=1.5, zorder=4)
    ax_price.axvspan(pd.Timestamp(crash_start), prices.index[crash_end_idx],
                     alpha=0.13, color=RED, label="Synthetic crash window", zorder=2)
    ax_price.set_title("SPY Daily Close Price — CPCV Group Structure", pad=10)
    ax_price.set_ylabel("Price (USD)")
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax_price.legend(loc="upper left")

    if has_groups:
        indices   = np.arange(len(X))
        groups    = np.array_split(indices, N_groups)

        if highlight_groups_on_price:
            overlay_colors = ["#eef4fa", "#f8fbfd"]
            for gid, grp in enumerate(groups):
                start = X.index[grp[0]]
                end = X.index[grp[-1]]
                ax_price.axvspan(
                    start,
                    end,
                    color=overlay_colors[gid % len(overlay_colors)],
                    alpha=0.9,
                    zorder=0,
                )
                midpoint = start + (end - start) / 2
                ax_price.text(
                    midpoint,
                    0.985,
                    f"Fold {gid + 1}",
                    transform=ax_price.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=GRAY,
                )

        # Vertical dividers between groups
        for gid in range(1, N_groups):
            boundary = X.index[groups[gid][0]]
            ax_price.axvline(boundary, color="#b9c3cc", lw=0.8, ls="-", alpha=0.75, zorder=2)

    fig.tight_layout()
    _save(fig, "01_spy_prices.png", out_dir)


# ── 02 Split Matrix ───────────────────────────────────────────────────────────
def plot_split_matrix(split_table: pd.DataFrame, N: int, out_dir="plots/"):
    n_splits = len(split_table)
    matrix   = np.zeros((n_splits, N), dtype=int)
    for i, tg in enumerate(split_table["test_groups"]):
        for g in tg:
            matrix[i, g] = 1

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    # Grid lines between cells
    for x in range(N + 1):
        ax.axvline(x - 0.5, color="white", lw=1.5)
    for y in range(n_splits + 1):
        ax.axhline(y - 0.5, color="white", lw=1.5)

    ax.set_xticks(range(N))
    ax.set_xticklabels([f"G{g + 1}" for g in range(N)], fontsize=10)
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Split {i + 1}" for i in range(n_splits)], fontsize=8)
    ax.set_title(f"CPCV — Split × Group Assignment  (blue = test,  {n_splits} splits total)")
    ax.set_xlabel("Time group")
    ax.set_ylabel("CV split")

    # Annotation: "TEST" in blue cells
    for i in range(n_splits):
        for j in range(N):
            if matrix[i, j]:
                ax.text(j, i, "TEST", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "02_split_matrix.png", out_dir)


def plot_path_example(fold_results: list, path_results: list, N: int,
                      path_id: int = 0, out_dir="plots/"):
    """Show the split×group submatrix for one CPCV path example."""
    if not path_results:
        return

    path = next((p for p in path_results if p["path_id"] == path_id), None)
    if path is None:
        path = path_results[0]
        path_id = path["path_id"]

    fold_map = {f["fold_id"]: f for f in fold_results}
    split_ids = path["split_ids"]
    matrix = np.zeros((len(split_ids), N), dtype=int)
    row_labels = []

    for i, split_id in enumerate(split_ids):
        groups = fold_map[split_id]["test_groups"]
        for g in groups:
            matrix[i, g] = 1
        row_labels.append(f"Split {split_id + 1}")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    for x in range(N + 1):
        ax.axvline(x - 0.5, color="white", lw=1.5)
    for y in range(len(split_ids) + 1):
        ax.axhline(y - 0.5, color="white", lw=1.5)

    ax.set_xticks(range(N))
    ax.set_xticklabels([f"G{g + 1}" for g in range(N)], fontsize=10)
    ax.set_yticks(range(len(split_ids)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Time group")
    ax.set_ylabel("Splits in path")
    ax.set_title(
        f"CPCV Path {path_id + 1} — Split × Group Submatrix\n"
        f"(Path {path_id + 1} = {', '.join(row_labels)})"
    )

    for i in range(len(split_ids)):
        for j in range(N):
            if matrix[i, j]:
                ax.text(j, i, "TEST", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

    ax.grid(False)
    fig.tight_layout()
    _save(fig, "02b_path_example.png", out_dir)


# ── Shared metrics grid ───────────────────────────────────────────────────────
def _plot_metrics_grid(df: pd.DataFrame, labels: list, filename: str,
                       suptitle: str, out_dir: str):
    """
    3×3 grid of IS vs OOS bar charts. Reused for splits, paths, and comparison.
    df must have columns: is_sharpe/sharpe, is_accuracy/accuracy, is_f1/f1,
    is_mean_return_pct/mean_return_pct, is_ann_return_pct/ann_return_pct,
    is_max_drawdown_pct/max_drawdown_pct, is_hit_ratio/hit_ratio,
    is_profit_factor/profit_factor, is_volatility_pct/volatility_pct.
    """
    panels = [
        ("is_sharpe",           "sharpe",           "Sharpe Ratio",          "Ann. Sharpe",   False, None),
        ("is_accuracy",         "accuracy",         "Accuracy",              "Accuracy",       True,  0.5),
        ("is_f1",               "f1",               "F1 Score",              "F1",             True,  None),
        ("is_mean_return_pct",  "mean_return_pct",  "Mean Return",           "Mean Ret (%)",  False, None),
        ("is_ann_return_pct",   "ann_return_pct",   "Annualized Return",     "Ann Ret (%)",   False, None),
        ("is_max_drawdown_pct", "max_drawdown_pct", "Max Drawdown",          "MDD (%)",       False, None),
        ("is_hit_ratio",        "hit_ratio",        "Hit Ratio",             "Hit Ratio",      True,  0.5),
        ("is_profit_factor",    "profit_factor",    "Profit Factor (OOS)",   "Profit Factor", False, 1.0),
        ("is_volatility_pct",   "volatility_pct",   "Annualized Volatility", "Volatility (%)",False, None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.ravel()

    for ax, (is_col, oos_col, title, ylabel, pct, baseline) in zip(axes, panels):
        if is_col == "is_profit_factor":
            x = np.arange(len(df))
            w = 0.55
            ax.bar(x, df[oos_col], w, label="OOS", color=RED, alpha=0.85, zorder=3)
            ax.axhline(1.0, color=GRAY, ls="--", lw=0.8, zorder=2, label="Breakeven (PF=1)")
            ax.axhline(0, color=BLACK, lw=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel)
            ax.text(0.98, 0.97, "IS always = 10\n(model memorises train)",
                    transform=ax.transAxes, fontsize=7, ha="right", va="top", color=GRAY,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=LGRAY))
            ax.legend(fontsize=8)
        else:
            _plot_is_oos_bars(ax, df, labels, is_col, oos_col, title, ylabel,
                              percent_axis=pct, baseline=baseline)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(suptitle, y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename, out_dir)


# ── 03 IS vs OOS per split ────────────────────────────────────────────────────
def _plot_is_oos_bars(ax, df: pd.DataFrame, x_labels, is_col: str, oos_col: str,
                      title: str, ylabel: str, percent_axis: bool = False,
                      baseline: float = None):
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w/2, df[is_col],  w, label="IS",  color=BLUE, alpha=0.85, zorder=3)
    ax.bar(x + w/2, df[oos_col], w, label="OOS", color=RED,  alpha=0.85, zorder=3)
    ax.axhline(0, color=BLACK, lw=0.7, zorder=2)
    if baseline is not None:
        ax.axhline(baseline, color=GRAY, ls="--", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    if percent_axis:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))


def plot_is_oos_per_split(fold_results: list, out_dir="plots/"):
    df = pd.DataFrame([{
        "is_sharpe":           f["is_sharpe"],
        "sharpe":              f["sharpe"],
        "is_accuracy":         f["is_accuracy"],
        "accuracy":            f["accuracy"],
        "is_f1":               f["is_f1"],
        "f1":                  f["f1"],
        "is_mean_return_pct":  f["is_mean_return_pct"],
        "mean_return_pct":     f["mean_return_pct"],
        "is_ann_return_pct":   f["is_ann_return_pct"],
        "ann_return_pct":      f["ann_return_pct"],
        "is_max_drawdown_pct": f["is_max_drawdown_pct"],
        "max_drawdown_pct":    f["max_drawdown_pct"],
        "is_hit_ratio":        f["is_hit_ratio"],
        "hit_ratio":           f["hit_ratio"],
        "is_profit_factor":    f["is_profit_factor"],
        "profit_factor":       f["profit_factor"],
        "is_volatility_pct":   f["is_volatility_pct"],
        "volatility_pct":      f["volatility_pct"],
    } for f in fold_results])
    labels = [f"S{f['fold_id'] + 1}" for f in fold_results]
    _plot_metrics_grid(df, labels, "03_is_oos_per_split.png",
                       "CPCV — Metrics per Split: IS vs OOS", out_dir)


def plot_metrics_per_path(path_results: list, out_dir="plots/"):
    df = pd.DataFrame([{
        "path":                p["path_id"],
        "is_sharpe":           p["is_sharpe"],
        "sharpe":              p["sharpe"],
        "is_accuracy":         p["is_accuracy"],
        "accuracy":            p["accuracy"],
        "is_f1":               p["is_f1"],
        "f1":                  p["f1"],
        "is_mean_return_pct":  p["is_mean_return_pct"],
        "mean_return_pct":     p["mean_return_pct"],
        "is_ann_return_pct":   p["is_ann_return_pct"],
        "ann_return_pct":      p["ann_return_pct"],
        "is_max_drawdown_pct": p["is_max_drawdown_pct"],
        "max_drawdown_pct":    p["max_drawdown_pct"],
        "is_hit_ratio":        p["is_hit_ratio"],
        "hit_ratio":           p["hit_ratio"],
        "is_profit_factor":    p["is_profit_factor"],
        "profit_factor":       p["profit_factor"],
        "is_volatility_pct":   p["is_volatility_pct"],
        "volatility_pct":      p["volatility_pct"],
    } for p in path_results])
    labels = [f"P{p['path_id'] + 1}" for p in path_results]
    _plot_metrics_grid(df, labels, "04_metrics_per_path.png",
                       "CPCV — Metrics per Path: IS vs OOS", out_dir)


# ── 04 Equity curves per path ─────────────────────────────────────────────────
def plot_equity_curves(path_results: list, out_dir="plots/"):
    """Cumulative OOS PnL per path (additive log-return cumsum)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    palette = [BLUE, RED, "#27ae60", "#8e44ad", "#e67e22",
               "#2980b9", "#c0392b", "#16a085", "#d35400"]
    for pr, col in zip(path_results, palette):
        pnl  = pr["_path_pnl"].sort_index()
        cum  = pnl.cumsum()          # additive cumulative log-return (no exp blow-up)
        cum_pct = cum * 100          # express as %
        ax.plot(cum_pct.index, cum_pct.values, lw=1.5,
                label=f"Path {pr['path_id'] + 1}  SR={pr['sharpe']:.2f}", color=col)
    ax.axhline(0, color=GRAY, ls="--", lw=0.8)
    ax.set_title("CPCV — OOS Equity Curves by Path\n"
                 "(each path = disjoint time segments covering all groups once)")
    ax.set_ylabel("Cumulative Log-Return (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    _save(fig, "04_equity_curves.png", out_dir)


# ── 05 Comparison metrics grid ────────────────────────────────────────────────
def plot_comparison_metrics(comparison_df: pd.DataFrame, out_dir="plots/"):
    """
    Same 3×3 grid as splits/paths but for method-level comparison.
    comparison_df index = method name; columns include IS_SR, OOS_SR, Delta_SR,
    accuracy, f1, mean_return_pct, ann_return_pct, max_drawdown_pct,
    hit_ratio, profit_factor, volatility_pct.
    """
    df = comparison_df.reset_index().rename(columns={
        "IS_SR":            "is_sharpe",
        "OOS_SR":           "sharpe",
        "accuracy":         "accuracy",
        "f1":               "f1",
        "mean_return_pct":  "mean_return_pct",
        "ann_return_pct":   "ann_return_pct",
        "max_drawdown_pct": "max_drawdown_pct",
        "hit_ratio":        "hit_ratio",
        "profit_factor":    "profit_factor",
        "volatility_pct":   "volatility_pct",
    })
    # For comparison, IS columns are the same as OOS columns (method-level averages)
    # except Sharpe where we have both IS_SR and OOS_SR.
    for col in ["accuracy", "f1", "mean_return_pct", "ann_return_pct",
                "max_drawdown_pct", "hit_ratio", "profit_factor", "volatility_pct"]:
        df[f"is_{col}"] = df[col]   # no separate IS for these — show same bar

    labels = df["method"].tolist()
    _plot_metrics_grid(df, labels, "05_comparison_metrics.png",
                       "All Methods — OOS Metrics Comparison", out_dir)


# ── 06 Delta SR ───────────────────────────────────────────────────────────────
def plot_comparison_delta(comparison_df: pd.DataFrame, out_dir="plots/"):
    methods = comparison_df.index.tolist()
    deltas  = comparison_df["Delta_SR"].values
    colors  = [RED if v > 0 else BLUE for v in deltas]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(x, deltas, color=colors, alpha=0.85, zorder=3)
    ax.axhline(0, color=BLACK, lw=0.8)
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=9)
    ax.set_title("Sharpe Degradation: IS − OOS by Method\n"
                 "(higher bar = more optimistic bias / overfitting risk)")
    ax.set_ylabel("ΔSharpe  (IS − OOS)")
    fig.tight_layout()
    _save(fig, "06_comparison_delta_sr.png", out_dir)


# ── 09 Heatmap ────────────────────────────────────────────────────────────────
def plot_comparison_heatmap(comparison_df: pd.DataFrame, out_dir="plots/"):
    cols  = ["IS_SR", "OOS_SR", "Delta_SR", "mean_return_pct", "accuracy", "f1"]
    labels = ["IS SR", "OOS SR", "ΔSR", "Mean Ret", "Accuracy", "F1"]
    data  = comparison_df[cols].astype(float)
    # Normalize per column for color scale (0=worst, 1=best within column)
    norm  = (data - data.min()) / (data.max() - data.min() + 1e-9)
    # For Delta_SR lower is better.
    norm["Delta_SR"] = 1 - norm["Delta_SR"]

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(norm.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)

    for i in range(len(data)):
        for j, col in enumerate(cols):
            val = data.iloc[i][col]
            txt = f"{val:.3f}" if col in ["IS_SR", "OOS_SR", "Delta_SR"] else \
                  f"{val:.1%}" if col in ["accuracy", "f1"] else f"{val:.2f}%"
            color = "white" if norm.values[i, j] > 0.55 else BLACK
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Methods × Metrics — Normalized Heatmap\n"
                 "(color = normalized score; for ΔSR, lower is better → inverted)")
    plt.colorbar(im, ax=ax, shrink=0.5, label="Normalized score (higher = better)")
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "09_comparison_heatmap.png", out_dir)


# ── 10 OOS Degradation ────────────────────────────────────────────────────────
def plot_oos_degradation(fold_results: list, all_folds_df: pd.DataFrame = None,
                         out_dir="plots/"):
    """
    IS vs OOS Sharpe scatter with regression line.
    Uses all_folds_df (all methods × folds) for density like De Prado Fig 11.1.
    Falls back to fold_results (CPCV only) if all_folds_df not provided.
    """
    if all_folds_df is not None and len(all_folds_df) > 5:
        df = all_folds_df.copy()
        # Clip extreme IS_SR outliers (beyond 99th percentile) that arise from
        # near-zero-variance training windows in combinatorial splits.
        p99_is = np.percentile(df["IS_SR"].abs(), 99)
        df = df[df["IS_SR"].abs() <= p99_is].copy()
        x, y = df["IS_SR"].values, df["OOS_SR"].values
        source = "all methods × common trials"
    else:
        from cpcv_analysis.advanced_analysis import oos_degradation
        df = oos_degradation(fold_results)
        x, y = df["IS_SR"].values, df["OOS_SR"].values
        source = "CPCV folds only"

    if len(x) < 3:
        print("[plots] Not enough data for degradation plot, skipping.")
        return

    slope, intercept, r, p, se = stats.linregress(x, y)
    n      = len(x)
    prob_neg = float(np.mean(y < 0))
    adj_r2   = max(0.0, r**2 - (1 - r**2) / max(n - 2, 1))

    # Single panel. IS_SR >> OOS_SR in all folds (structural overfitting).
    # The identity line y=x cannot be shown at scale — instead we draw a
    # horizontal reference at OOS=IS_mean to make the gap legible, plus
    # annotate the mean IS vs mean OOS for context.
    x_margin = (x.max() - x.min()) * 0.1 + 0.5
    y_margin = (y.max() - y.min()) * 0.15 + 0.5
    x_lo, x_hi = x.min() - x_margin, x.max() + x_margin
    y_lo, y_hi = y.min() - y_margin, y.max() + y_margin

    x_line  = np.linspace(x_lo, x_hi, 300)
    y_line  = slope * x_line + intercept
    se_pred = se * np.sqrt(1/n + (x_line - x.mean())**2 / np.sum((x - x.mean())**2))
    ci      = 1.96 * se_pred

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color=BLUE, alpha=0.6, s=40, zorder=4, label=f"Folds ({source})")
    ax.plot(x_line, y_line, color=RED, lw=2, zorder=5,
            label=f"Regression  R²={r**2:.2f}")
    ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.12, color=RED)

    # Horizontal reference: OOS = mean(IS) — shows how far OOS falls below IS
    ax.axhline(x.mean(), color="#27ae60", ls="--", lw=1.1, zorder=3,
               label=f"IS mean = {x.mean():.1f}  (identity reference)")
    ax.axhline(0, color=GRAY, ls=":", lw=0.8, zorder=2, label="OOS = 0")

    # Annotation
    eq = (f"SR_OOS = {intercept:+.2f} + {slope:.2f}·SR_IS\n"
          f"adj R² = {adj_r2:.2f}   n = {n}\n"
          f"Mean IS SR = {x.mean():.2f}  |  Mean OOS SR = {y.mean():.2f}\n"
          f"Prob[SR_OOS < 0] = {prob_neg:.2f}")
    ax.text(0.04, 0.97, eq, transform=ax.transAxes, fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=LGRAY))

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Sharpe Ratio — In-Sample (IS)")
    ax.set_ylabel("Sharpe Ratio — Out-of-Sample (OOS)")
    ax.set_title("OOS Performance Degradation — IS vs OOS Sharpe Ratio\n"
                 "(each dot = one method × fold; green line = IS mean = identity would be at y≈IS)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, "10_oos_degradation.png", out_dir)


# ── 11 Rank Logits ────────────────────────────────────────────────────────────
def plot_rank_logits(all_folds_df, out_dir="plots/"):
    from cpcv_analysis.advanced_analysis import rank_logits
    logits, prob_overfit, method_names = rank_logits(all_folds_df)

    mu, sigma = stats.norm.fit(logits)
    x_curve   = np.linspace(logits.min() - 1, logits.max() + 1, 300)
    y_curve   = stats.norm.pdf(x_curve, mu, sigma)

    fig, ax = plt.subplots(figsize=(9, 5))
    n_bins = max(15, len(logits) // 5)
    ax.hist(logits, bins=n_bins, color=BLUE, alpha=0.75, density=True,
            label="Rank logits", zorder=3)
    ax.plot(x_curve, y_curve, color=RED, lw=2, label="Fitted normal", zorder=4)
    ax.axvline(0, color=BLACK, ls="--", lw=1, zorder=5)

    # Shade area to left of 0 (overfit region)
    mask = x_curve < 0
    ax.fill_between(x_curve[mask], y_curve[mask], alpha=0.20, color=RED,
                    label=f"Prob[Overfit] = {prob_overfit:.2f}", zorder=2)

    ax.text(0.97, 0.95, f"Prob[Overfit] = {prob_overfit:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=LGRAY))

    ax.set_xlabel("Logit of Rank  λ = log( rank / (N − rank) )")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Rank Logits — Probability of Backtest Overfitting\n"
                 "(best IS method ranked by OOS in each common trial)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "11_rank_logits.png", out_dir)


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
            label="Clean features", color=BLUE, alpha=0.85, zorder=3)
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
            label="Clean features", color=BLUE, alpha=0.85, zorder=3)
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
    _save(fig, "12_leakage_comparison.png", out_dir)
