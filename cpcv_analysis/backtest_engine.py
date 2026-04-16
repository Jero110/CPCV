# cpcv_analysis/backtest_engine.py
"""
backtest_engine.py
Funciones de debug y producción para backtesting CPCV + comparación de métodos.
Toda la lógica vive aquí. El notebook solo importa y llama.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations

from sklearn.base import clone
from sklearn.model_selection import KFold

from cpcv_analysis.splitters import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    WalkForwardCV,
)
from cpcv_analysis.cv_runner import get_paths, run_cpcv
from cpcv_analysis.config import N_GROUPS, K_TEST, PCT_EMBARGO
from IPython.display import display


# ── helpers ──────────────────────────────────────────────────────────────────

def get_last_n_days(X, y, t1, fwd_ret, n=100):
    """Slice the last n observations."""
    X = X.iloc[-n:]
    y = y.loc[X.index]
    t1 = t1.loc[X.index]
    fwd_ret = fwd_ret.loc[X.index]
    return X, y, t1, fwd_ret


def _fold_sharpe(pnl: pd.Series, periods: int = 252) -> float:
    """
    Annualized Sharpe ratio.
    Formula: SR = sqrt(periods) * mean(pnl) / std(pnl)
    Returns 0.0 if std == 0 or fewer than 2 observations.
    """
    if len(pnl) < 2 or pnl.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * pnl.mean() / pnl.std())


def _pnl_from_split(clf, X, y, t1, fwd_ret, final_tr, test_idx):
    """
    Fit clf on train, return (is_pnl, oos_pnl, y_hat_tr, y_hat_te).
    PnL = sign(y_pred_mapped) * fwd_ret  where sign maps {0→-1, 1→+1}.
    """
    X_tr, y_tr = X.iloc[final_tr], y.iloc[final_tr]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

    clf = clone(clf)
    clf.fit(X_tr, y_tr)

    y_hat_tr = clf.predict(X_tr)
    signs_tr = (2 * y_hat_tr - 1).astype(float)
    is_pnl = pd.Series(signs_tr * fwd_ret.iloc[final_tr].values,
                       index=X_tr.index, dtype=float)

    y_hat_te = clf.predict(X_te)
    signs_te = (2 * y_hat_te - 1).astype(float)
    oos_pnl = pd.Series(signs_te * fwd_ret.iloc[test_idx].values,
                        index=X_te.index, dtype=float)

    return is_pnl, oos_pnl, y_hat_tr, y_hat_te


def _build_cpcv_splits_table(clf, X, y, t1, fwd_ret,
                              n_groups=N_GROUPS, k_test=K_TEST,
                              pct_embargo=PCT_EMBARGO):
    """
    Corre todos los splits de CPCV y devuelve:
      - splits_info: lista de dicts con metadata por split
      - oos_by_split: dict {split_id: oos_pnl Series}
      - is_by_split:  dict {split_id: is_pnl Series}
      - preds_by_split: dict {split_id: (y_hat_tr, y_hat_te)}
    """
    cpcv = CombinatorialPurgedKFold(n_groups, k_test, t1, pct_embargo)
    splits_info = []
    oos_by_split = {}
    is_by_split = {}
    preds_by_split = {}

    for split_id, (raw_tr, test_idx, final_tr, test_groups) in enumerate(cpcv.split(X)):
        is_pnl, oos_pnl, y_hat_tr, y_hat_te = _pnl_from_split(
            clf, X, y, t1, fwd_ret, final_tr, test_idx)

        # Embargo: indices that were in raw_tr but removed by purge/embargo
        raw_tr_set = set(raw_tr.tolist())
        final_tr_set = set(final_tr.tolist())
        embargoed_idx = sorted(raw_tr_set - final_tr_set)

        splits_info.append({
            "split_id": split_id,
            "test_groups": test_groups,
            "train_start": X.index[final_tr[0]] if len(final_tr) else None,
            "train_end":   X.index[final_tr[-1]] if len(final_tr) else None,
            "test_start":  X.index[test_idx[0]],
            "test_end":    X.index[test_idx[-1]],
            "embargo_end": X.index[embargoed_idx[-1]] if embargoed_idx else None,
            "n_train": len(final_tr),
            "n_test":  len(test_idx),
            "n_embargoed": len(embargoed_idx),
            "_final_tr": final_tr,
            "_test_idx": test_idx,
            "_embargoed_idx": embargoed_idx,
        })
        oos_by_split[split_id] = oos_pnl
        is_by_split[split_id] = is_pnl
        preds_by_split[split_id] = (y_hat_tr, y_hat_te)

    return splits_info, oos_by_split, is_by_split, preds_by_split


def _get_split_detail(split_info, X, y, fwd_ret, is_pnl, oos_pnl,
                      y_hat_tr, y_hat_te):
    """
    Devuelve dos DataFrames: IS y OOS con columnas:
      date | y_pred | y_real | fwd_ret | pnl
    """
    final_tr = split_info["_final_tr"]
    test_idx = split_info["_test_idx"]

    is_df = pd.DataFrame({
        "y_pred": y_hat_tr,
        "y_real": y.iloc[final_tr].values,
        "fwd_ret": fwd_ret.iloc[final_tr].values,
        "pnl": is_pnl.values,
    }, index=X.index[final_tr])
    is_df.index.name = "date"

    oos_df = pd.DataFrame({
        "y_pred": y_hat_te,
        "y_real": y.iloc[test_idx].values,
        "fwd_ret": fwd_ret.iloc[test_idx].values,
        "pnl": oos_pnl.values,
    }, index=X.index[test_idx])
    oos_df.index.name = "date"

    return is_df, oos_df


def _plot_split_timeline(split_info, X):
    """Bar horizontal con train=azul, test=naranja, embargo=rojo."""
    n = len(X)
    colors = ["#3498db"] * n  # default train
    for idx in split_info["_test_idx"]:
        colors[idx] = "#e67e22"
    for idx in split_info["_embargoed_idx"]:
        colors[idx] = "#e74c3c"

    fig, ax = plt.subplots(figsize=(14, 1.5))
    ax.bar(range(n), [1] * n, color=colors, width=1.0, linewidth=0)
    ax.set_yticks([])
    ax.set_xlabel("Índice de observación")
    ax.set_title(
        f"Split {split_info['split_id']}  test_groups={split_info['test_groups']}  "
        f"n_train={split_info['n_train']}  n_test={split_info['n_test']}  "
        f"n_embargoed={split_info['n_embargoed']}"
    )
    legend = [
        mpatches.Patch(color="#3498db", label="Train"),
        mpatches.Patch(color="#e67e22", label="Test"),
        mpatches.Patch(color="#e74c3c", label="Embargo/Purged"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def _plot_sharpe_dist(sharpes: pd.Series, title: str = "Distribución de Sharpes"):
    """Histograma simple de una Serie de Sharpes."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sharpes.values, bins=max(5, len(sharpes) // 2), color="#3498db",
            edgecolor="white", alpha=0.85)
    ax.axvline(sharpes.mean(), color="#e74c3c", linewidth=2,
               label=f"Media={sharpes.mean():.3f}")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Sharpe anualizado")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def cpcv_debug(clf, X, y, t1, fwd_ret,
               split_to_inspect=0,
               n_groups=N_GROUPS, k_test=K_TEST,
               pct_embargo=PCT_EMBARGO):
    """
    Debug completo de CPCV. Muestra:
      1. Tabla de splits
      2. Timeline detallado del split `split_to_inspect`
      3. Tabla de predicciones IS y OOS por split
      4. Fórmulas explícitas con valores numéricos (verificable con calculadora)
      5. Returns de un path
      6. Sharpe del path con fórmula
      7. Distribución de Sharpes de paths
    """
    splits_info, oos_by_split, is_by_split, preds_by_split = _build_cpcv_splits_table(
        clf, X, y, t1, fwd_ret, n_groups, k_test, pct_embargo)

    # ── Sección 1: Tabla de splits ──────────────────────────────────────────
    print("=" * 70)
    print("SECCIÓN 1 — SPLITS OVERVIEW")
    print("=" * 70)
    rows = []
    for s in splits_info:
        rows.append({
            "split_id":    s["split_id"],
            "test_groups": str(s["test_groups"]),
            "train_start": s["train_start"].date() if s["train_start"] else None,
            "train_end":   s["train_end"].date()   if s["train_end"]   else None,
            "test_start":  s["test_start"].date(),
            "test_end":    s["test_end"].date(),
            "embargo_end": s["embargo_end"].date() if s["embargo_end"] else "-",
            "n_train":     s["n_train"],
            "n_test":      s["n_test"],
            "n_embargoed": s["n_embargoed"],
        })
    display(pd.DataFrame(rows).set_index("split_id"))

    # ── Sección 2: Timeline detallado ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"SECCIÓN 2 — TIMELINE DETALLADO (split {split_to_inspect})")
    print("=" * 70)
    _plot_split_timeline(splits_info[split_to_inspect], X)

    # ── Sección 3: Predicciones IS y OOS ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SECCIÓN 3 — PREDICCIONES POR SPLIT")
    print("=" * 70)
    for s in splits_info:
        sid = s["split_id"]
        y_hat_tr, y_hat_te = preds_by_split[sid]
        is_df, oos_df = _get_split_detail(
            s, X, y, fwd_ret, is_by_split[sid], oos_by_split[sid],
            y_hat_tr, y_hat_te)
        print(f"\n--- Split {sid}  test_groups={s['test_groups']} ---")
        print("  IS (train):")
        display(is_df.head(10).style.format({
            "fwd_ret": "{:.5f}", "pnl": "{:+.5f}"}))
        print("  OOS (test):")
        display(oos_df.style.format({
            "fwd_ret": "{:.5f}", "pnl": "{:+.5f}"}))

    # ── Sección 4: Fórmulas con valores numéricos ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("SECCIÓN 4 — FÓRMULAS IS vs OOS SHARPE (verificable con calculadora)")
    print("=" * 70)
    print("  PnL_i     = sign(y_pred_i) * fwd_ret_i   [sign: 0→-1, 1→+1]")
    print("  SR        = sqrt(252) * mean(pnl) / std(pnl)\n")
    for s in splits_info:
        sid = s["split_id"]
        is_pnl  = is_by_split[sid]
        oos_pnl = oos_by_split[sid]
        is_sr   = _fold_sharpe(is_pnl)
        oos_sr  = _fold_sharpe(oos_pnl)
        print(f"  Fold {sid}  test_groups={s['test_groups']}")
        print(f"    IS  SR = sqrt(252) * {is_pnl.mean():.6f} / {is_pnl.std():.6f}"
              f"  = {is_sr:+.4f}   (n={len(is_pnl)})")
        print(f"    OOS SR = sqrt(252) * {oos_pnl.mean():.6f} / {oos_pnl.std():.6f}"
              f"  = {oos_sr:+.4f}   (n={len(oos_pnl)})")
        print()

    # ── Secciones 5–7: paths ─────────────────────────────────────────────────
    paths = get_paths(n_groups, k_test)
    path_sharpes = []
    path_pnls = []
    for pid, split_ids in enumerate(paths):
        valid = [sid for sid in split_ids if sid in oos_by_split]
        if not valid:
            continue
        path_pnl = pd.concat([oos_by_split[sid] for sid in valid]).sort_index()
        path_pnls.append((pid, valid, path_pnl))
        path_sharpes.append(_fold_sharpe(path_pnl))

    # Sección 5: returns del primer path
    print(f"\n{'=' * 70}")
    print("SECCIÓN 5 — RETURNS DEL PATH 0")
    print("=" * 70)
    pid0, splits0, pnl0 = path_pnls[0]
    fig, ax = plt.subplots(figsize=(12, 3))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in pnl0]
    ax.bar(range(len(pnl0)), pnl0.values, color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Path {pid0} | splits={splits0} | OOS PnL por observación")
    ax.set_xlabel("Observación (orden temporal)")
    ax.set_ylabel("PnL (log-ret)")
    ax.set_xticks(range(0, len(pnl0), max(1, len(pnl0) // 10)))
    ax.set_xticklabels([str(d.date()) for d in pnl0.index[::max(1, len(pnl0) // 10)]],
                       rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.show()

    # Sección 6: Sharpe del path con fórmula
    print(f"\n{'=' * 70}")
    print("SECCIÓN 6 — SHARPE DEL PATH 0 (con fórmula)")
    print("=" * 70)
    sr0 = _fold_sharpe(pnl0)
    print(f"  Path {pid0}  splits={splits0}")
    print(f"  SR = sqrt(252) * mean(path_pnl) / std(path_pnl)")
    print(f"     = sqrt(252) * {pnl0.mean():.6f} / {pnl0.std():.6f}")
    print(f"     = {sr0:+.4f}")
    print(f"  n_obs={len(pnl0)}  rango={pnl0.index[0].date()} → {pnl0.index[-1].date()}")

    # Sección 7: distribución de Sharpes de paths
    print(f"\n{'=' * 70}")
    print("SECCIÓN 7 — DISTRIBUCIÓN DE SHARPES DE PATHS")
    print(f"  phi = C({n_groups},{k_test})*{k_test}/{n_groups} = {len(path_sharpes)} paths")
    print("=" * 70)
    _plot_sharpe_dist(pd.Series(path_sharpes, name="CPCV paths"),
                     title=f"Distribución Sharpes CPCV  (N={n_groups}, k={k_test}, "
                           f"phi={len(path_sharpes)})")

    return pd.Series(path_sharpes, name="cpcv_path_sharpes")


def cpcv_sharpe_dist(clf, X, y, t1, fwd_ret,
                     n_groups=N_GROUPS, k_test=K_TEST,
                     pct_embargo=PCT_EMBARGO) -> pd.Series:
    """
    Production-ready. Devuelve pd.Series con el Sharpe de cada path CPCV.
    Sin prints, sin plots. Misma lógica que cpcv_debug internamente.

    Formula: SR_path = sqrt(252) * mean(path_pnl) / std(path_pnl)
    """
    _, oos_by_split, _, _ = _build_cpcv_splits_table(
        clf, X, y, t1, fwd_ret, n_groups, k_test, pct_embargo)

    paths = get_paths(n_groups, k_test)
    sharpes = []
    for split_ids in paths:
        valid = [sid for sid in split_ids if sid in oos_by_split]
        if not valid:
            continue
        path_pnl = pd.concat([oos_by_split[sid] for sid in valid]).sort_index()
        sharpes.append(_fold_sharpe(path_pnl))

    return pd.Series(sharpes, name="cpcv_path_sharpes")


def wf_sharpe_dist(clf, X, y, t1, fwd_ret,
                   purged=False,
                   n_splits=6,
                   pct_embargo=PCT_EMBARGO) -> pd.Series:
    """
    Distribución de Sharpes OOS de WalkForward.
    purged=False → WalkForwardCV sin purge (solo expanding window)
    purged=True  → WalkForwardCV con purge+embargo (De Prado style)

    Formula por fold: SR = sqrt(252) * mean(oos_pnl) / std(oos_pnl)
    """
    if purged:
        splitter = WalkForwardCV(n_splits=n_splits, t1=t1, pctEmbargo=pct_embargo)
    else:
        splitter = WalkForwardCV(n_splits=n_splits, t1=None, pctEmbargo=0.0)

    sharpes = []
    for train_idx, test_idx in splitter.split(X):
        if len(train_idx) < 5 or len(test_idx) < 2:
            continue
        # Skip folds where train lacks both classes
        if len(np.unique(y.iloc[train_idx])) < 2:
            continue
        _, oos_pnl, _, _ = _pnl_from_split(
            clf, X, y, t1, fwd_ret, train_idx, test_idx)
        sharpes.append(_fold_sharpe(oos_pnl))

    label = "purged_wf" if purged else "walkforward"
    return pd.Series(sharpes, name=label)


def kfold_sharpe_dist(clf, X, y, t1, fwd_ret,
                      purged=False,
                      n_splits=6,
                      pct_embargo=PCT_EMBARGO) -> pd.Series:
    """
    Distribución de Sharpes OOS de KFold.
    purged=False → sklearn KFold sin purge
    purged=True  → PurgedKFold de De Prado con purge+embargo

    Formula por fold: SR = sqrt(252) * mean(oos_pnl) / std(oos_pnl)
    """
    if purged:
        splitter = PurgedKFold(n_splits=n_splits, t1=t1, pctEmbargo=pct_embargo)
        split_iter = splitter.split(X)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=False)
        split_iter = splitter.split(X)

    sharpes = []
    for train_idx, test_idx in split_iter:
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        if len(train_idx) < 5 or len(test_idx) < 2:
            continue
        # Skip folds where train lacks both classes (XGBoost requires 0-indexed classes)
        if len(np.unique(y.iloc[train_idx])) < 2:
            continue
        _, oos_pnl, _, _ = _pnl_from_split(
            clf, X, y, t1, fwd_ret, train_idx, test_idx)
        sharpes.append(_fold_sharpe(oos_pnl))

    label = "purged_kfold" if purged else "kfold"
    return pd.Series(sharpes, name=label)


def compare_methods(clf, X, y, t1, fwd_ret,
                    n_splits=6,
                    n_groups=N_GROUPS, k_test=K_TEST,
                    pct_embargo=PCT_EMBARGO) -> dict:
    """
    Corre los 5 métodos sobre los mismos datos y modelo.
    Devuelve dict con pd.Series de Sharpes por método.
    Produce un plot con las 5 distribuciones y una tabla resumen.

    Métodos:
      CPCV              — sharpe por path
      WalkForward       — sharpe por fold (sin purge)
      PurgedWalkForward — sharpe por fold (con purge+embargo)
      KFold             — sharpe por fold (sklearn, sin purge)
      PurgedKFold       — sharpe por fold (De Prado)
    """
    results = {
        "CPCV": cpcv_sharpe_dist(
            clf, X, y, t1, fwd_ret, n_groups, k_test, pct_embargo),
        "WalkForward": wf_sharpe_dist(
            clf, X, y, t1, fwd_ret, purged=False, n_splits=n_splits),
        "PurgedWalkForward": wf_sharpe_dist(
            clf, X, y, t1, fwd_ret, purged=True, n_splits=n_splits,
            pct_embargo=pct_embargo),
        "KFold": kfold_sharpe_dist(
            clf, X, y, t1, fwd_ret, purged=False, n_splits=n_splits),
        "PurgedKFold": kfold_sharpe_dist(
            clf, X, y, t1, fwd_ret, purged=True, n_splits=n_splits,
            pct_embargo=pct_embargo),
    }

    # ── Plot ─────────────────────────────────────────────────────────────────
    colors = {
        "CPCV":              "#2ecc71",
        "WalkForward":       "#3498db",
        "PurgedWalkForward": "#1abc9c",
        "KFold":             "#e74c3c",
        "PurgedKFold":       "#e67e22",
    }
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    for ax, (name, sharpes) in zip(axes, results.items()):
        ax.hist(sharpes.values, bins=max(3, len(sharpes) // 2),
                color=colors[name], edgecolor="white", alpha=0.85)
        ax.axvline(sharpes.mean(), color="black", linewidth=1.5,
                   linestyle="--", label=f"μ={sharpes.mean():.2f}")
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Sharpe", fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("Distribución de Sharpes OOS por método CV", fontsize=13)
    plt.tight_layout()
    plt.show()

    # Tabla resumen
    summary = pd.DataFrame({
        name: {
            "n_units":      len(s),
            "mean_SR":      round(s.mean(), 3),
            "std_SR":       round(s.std(), 3),
            "min_SR":       round(s.min(), 3),
            "max_SR":       round(s.max(), 3),
            "pct_positive": round((s > 0).mean(), 2),
        }
        for name, s in results.items()
    }).T
    display(summary)

    return results
