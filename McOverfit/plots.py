import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_STYLE = {
    "figure.facecolor": "black",
    "axes.facecolor":   "black",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "legend.facecolor": "#111111",
    "legend.edgecolor": "#333333",
    "font.family":      "monospace",
}

def _apply_style():
    plt.rcParams.update(_STYLE)

def _fan_params(n):
    return np.clip(4.0 / n, 0.04, 0.25), (1.0 if n <= 200 else 0.5)

def _spine_style(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.tick_params(colors="white", labelsize=9)


def is_curves(dates_is, real_is_eq, perm_data, n_perm, save_path=None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 5), facecolor="black")
    ax.set_facecolor("black")

    alpha, lw = _fan_params(n_perm)
    for eq in perm_data["curves"]:
        ax.plot(dates_is, eq, color="white", lw=lw, alpha=alpha)

    median_eq = np.median(np.vstack(perm_data["curves"]), axis=0)
    ax.plot(dates_is, median_eq, color="white", lw=2.0, label="Permutation Optimized (median)")
    ax.plot(dates_is, real_is_eq, color="#e05c5c", lw=2.0, label="Real Optimized")

    _spine_style(ax)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Cumulative Log Return", fontsize=10)
    ax.set_title(f"In-Sample Permutation {n_perm}", fontsize=13, pad=10)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.show()


def is_histogram(perm_data, real_is_sr, pvalue_is, n_perm, save_path=None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="black")
    ax.set_facecolor("black")

    ax.hist(perm_data["sharpes"], bins=40, color="#555555", edgecolor="none")
    ax.axvline(real_is_sr, color="#e05c5c", lw=2.0, label="Real")
    ax.text(0.63, 0.82, f"p-value: {pvalue_is:.3f}", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    _spine_style(ax)
    ax.set_xlabel("IS Sharpe Ratio (best of grid search)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"In-Sample Permutation {n_perm} — Sharpe Distribution", fontsize=13, pad=10)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.show()


def wf_curves(dates_is, dates_oos, log_close_is, real_oos_eq, real_oos_sr,
              perm_data, pvalue_wf, n_perm, save_path=None):
    _apply_style()
    is_end_val = log_close_is.iloc[-1]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="black")
    ax.set_facecolor("black")

    ax.plot(dates_is, log_close_is.values, color="#888888", lw=1.4)

    alpha, lw = _fan_params(n_perm)
    for eq in perm_data["curves"]:
        ax.plot(dates_oos, is_end_val + eq, color="white", lw=lw, alpha=alpha)

    ax.plot(dates_oos, is_end_val + real_oos_eq, color="#e05c5c", lw=2.0,
            label=f"Real OOS (SR={real_oos_sr:.2f})")
    ax.axvline(dates_oos[0], color="#555555", lw=1.0, ls="--")

    _spine_style(ax)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Log Close  /  Cum. Log Return (OOS)", fontsize=10)
    ax.set_title(f"Walkforward Permutation {n_perm}", fontsize=13, pad=10)

    handles = [
        Line2D([0], [0], color="#888888", lw=1.4,  label="IS Log Price"),
        Line2D([0], [0], color="white",   lw=1.0, alpha=0.6, label="OOS Permuted"),
        Line2D([0], [0], color="#e05c5c", lw=2.0,  label=f"OOS Real (SR={real_oos_sr:.2f})"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    ax.text(0.55, 0.07, f"p-value (WF) = {pvalue_wf:.3f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.show()


def wf_histogram(perm_data, real_oos_sr, pvalue_wf, n_perm, save_path=None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="black")
    ax.set_facecolor("black")

    ax.hist(perm_data["sharpes"], bins=40, color="#555555", edgecolor="none")
    ax.axvline(real_oos_sr, color="#e05c5c", lw=2.0, label="Real OOS")
    ax.text(0.63, 0.82, f"p-value: {pvalue_wf:.3f}", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    _spine_style(ax)
    ax.set_xlabel("OOS Sharpe Ratio", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Walkforward Permutation {n_perm} — OOS Sharpe Distribution", fontsize=13, pad=10)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.show()
