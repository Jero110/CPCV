# cpcv_analysis/plots.py
"""
Plot functions used by full_experiments.ipynb:
    - plot_split_matrix       — CPCV (N, k) split×group assignment
    - plot_fold_oos_violins   — OOS Sharpe per fold/split (3 violins)
    - plot_paths_vs_holdout   — CPCV path distribution + WF/KFold points + hold-out
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "font.family":         "sans-serif",
    "font.size":           11,
    "legend.frameon":      False,
})


def _save(fig, name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[plots] Saved → {path}")
    plt.close(fig)


def _strip_ax(ax):
    """Minimal academic axes: top/right spines off, thin remaining spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=3, width=0.6, labelsize=8)


def _bw_jitter_group(ax, pos, arr, rng, marker="o"):
    """Scatter a group of points with small jitter; filled black markers."""
    jit = rng.uniform(-0.10, 0.10, size=len(arr))
    ax.scatter(pos + jit, arr, color="black", s=18, zorder=5,
               linewidths=0, marker=marker)


# ── Split matrix ──────────────────────────────────────────────────────────────
def plot_split_matrix(split_table: pd.DataFrame, N: int, out_dir="plots/"):
    n_splits = len(split_table)
    matrix   = np.zeros((n_splits, N), dtype=int)
    for i, tg in enumerate(split_table["test_groups"]):
        for g in tg:
            matrix[i, g] = 1

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

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

    for i in range(n_splits):
        for j in range(N):
            if matrix[i, j]:
                ax.text(j, i, "TEST", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "02_split_matrix.png", out_dir)


# ── OOS Sharpe per fold/split (3 violins) ─────────────────────────────────────
def plot_fold_oos_violins(cpcv_fold_srs, wf_fold_srs, kfold_fold_srs,
                          label: str, out_dir: str = "plots/"):
    """
    OOS Sharpe ratio per fold/split for each method.
    CPCV: 15 splits, WF: 3 folds, KFold: 3 folds.
    Black-and-white publication style.
    """
    methods = [
        ("CPCV\n(15 splits)",       cpcv_fold_srs,  1),
        ("Walk-forward\n(3 folds)", wf_fold_srs,    2),
        ("KFold\n(3 folds)",        kfold_fold_srs, 3),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    rng = np.random.default_rng(0)

    for name, data, pos in methods:
        arr = np.asarray(data, dtype=float)
        if len(arr) == 0:
            continue
        if len(arr) >= 4:
            parts = ax.violinplot([arr], positions=[pos], showmedians=False,
                                  showextrema=False, widths=0.50)
            for pc in parts["bodies"]:
                pc.set_facecolor("#dddddd")
                pc.set_alpha(1.0)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.6)
        _bw_jitter_group(ax, pos, arr, rng)
        med = float(np.median(arr))
        ax.hlines(med, pos - 0.25, pos + 0.25, colors="black", lw=1.2, ls="--")

    ax.axhline(0, color="black", ls=":", lw=0.6, zorder=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([m[0] for m in methods], fontsize=8)
    ax.set_ylabel("OOS Sharpe ratio (annualised)", fontsize=9)
    ax.set_title(label, fontsize=8, pad=5)
    _strip_ax(ax)
    ax.grid(axis="y", lw=0.4, ls=":", color="#cccccc", zorder=0)
    fig.tight_layout(pad=0.8)
    _save(fig, "03_fold_oos_violins.png", out_dir)


# ── CPCV paths + WF/KFold points + hold-out reference ────────────────────────
def plot_paths_vs_holdout(cpcv_path_srs, wf_sr, kfold_sr, holdout_sr,
                          label: str, out_dir: str = "plots/"):
    """
    CPCV: distribution of path Sharpes (violin + jittered dots).
    WF and KFold: single point each — Sharpe of all OOS returns concatenated.
    Hold-out: horizontal reference line.
    Black-and-white publication style.
    """
    cpcv_arr = np.asarray(cpcv_path_srs, dtype=float)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    # CPCV: light violin + jittered dots
    if len(cpcv_arr) >= 4:
        parts = ax.violinplot([cpcv_arr], positions=[1], showmedians=False,
                              showextrema=False, widths=0.46)
        for pc in parts["bodies"]:
            pc.set_facecolor("#dddddd")
            pc.set_alpha(1.0)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.6)
    _bw_jitter_group(ax, 1, cpcv_arr, rng)
    med_c = float(np.median(cpcv_arr)) if len(cpcv_arr) else 0.0
    ax.hlines(med_c, 0.75, 1.27, colors="black", lw=1.2, ls="--")

    # WF: single point (concat SR)
    ax.scatter([2], [wf_sr], color="black", s=40, zorder=6, marker="s")

    # KFold: single point (concat SR)
    ax.scatter([3], [kfold_sr], color="black", s=40, zorder=6, marker="^")

    # Hold-out reference line
    ax.hlines(holdout_sr, 0.55, 3.65, colors="black", lw=1.6, ls="-", zorder=4)

    ax.axhline(0, color="black", ls=":", lw=0.6, zorder=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["CPCV\n(paths)", "Walk-forward\n(concat SR)",
                         "KFold\n(concat SR)"], fontsize=8)
    ax.set_ylabel("OOS Sharpe ratio (annualised)", fontsize=9)
    ax.set_title(label, fontsize=8, pad=5)

    handles = [
        Line2D([0], [0], color="black", lw=1.2, ls="--", label=f"CPCV median $= {med_c:.2f}$"),
        Line2D([0], [0], color="black", lw=1.6, ls="-",  label=f"Hold-out $= {holdout_sr:.2f}$"),
        Line2D([0], [0], color="black", lw=0, marker="s", markersize=5, label=f"WF $= {wf_sr:.2f}$"),
        Line2D([0], [0], color="black", lw=0, marker="^", markersize=5, label=f"KFold $= {kfold_sr:.2f}$"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", frameon=False)
    ax.set_xlim(0.45, 3.85)
    _strip_ax(ax)
    ax.grid(axis="y", lw=0.4, ls=":", color="#cccccc", zorder=0)
    fig.tight_layout(pad=0.8)
    _save(fig, "04_paths_vs_holdout.png", out_dir)
