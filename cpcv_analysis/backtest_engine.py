# cpcv_analysis/backtest_engine.py
"""
backtest_engine.py
Funciones de debug y producción para backtesting CPCV + comparación de métodos.
Toda la lógica vive aquí. El notebook solo importa y llama.
"""
import numpy as np
import pandas as pd
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


def slice_by_dates(X, y, t1, fwd_ret, start: str = None, end: str = None):
    """
    Slice X/y/t1/fwd_ret por rango de fechas [start, end).
    start/end son strings 'YYYY-MM-DD'. None = sin límite.
    """
    mask = pd.Series(True, index=X.index)
    if start:
        mask = mask & (X.index >= pd.Timestamp(start))
    if end:
        mask = mask & (X.index < pd.Timestamp(end))
    idx = X.index[mask]
    return X.loc[idx], y.loc[idx], t1.loc[idx], fwd_ret.loc[idx]


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

    for split_id, (raw_tr, test_idx, final_tr, test_groups,
                    purged_only_idx, embargoed_idx_arr) in enumerate(cpcv.split(X)):
        is_pnl, oos_pnl, y_hat_tr, y_hat_te = _pnl_from_split(
            clf, X, y, t1, fwd_ret, final_tr, test_idx)

        # Usar directamente los índices calculados por el splitter:
        # purged_only_idx: removidos por solapamiento de label (antes/alrededor del test)
        # embargoed_idx_arr: removidos por embargo (inmediatamente después del test)
        purged_idx    = sorted(purged_only_idx.tolist())
        embargoed_idx = sorted(embargoed_idx_arr.tolist())

        splits_info.append({
            "split_id":    split_id,
            "test_groups": test_groups,
            "n_train":     len(final_tr),
            "n_test":      len(test_idx),
            "n_purged":    len(purged_idx),
            "n_embargoed": len(embargoed_idx),
            "_final_tr":     final_tr,
            "_test_idx":     test_idx,
            "_purged_idx":   purged_idx,
            "_embargoed_idx": embargoed_idx,
        })
        oos_by_split[split_id] = oos_pnl
        is_by_split[split_id] = is_pnl
        preds_by_split[split_id] = (y_hat_tr, y_hat_te)

    return splits_info, oos_by_split, is_by_split, preds_by_split


def _date_ranges(idx_array, X):
    """Convierte array de índices enteros a string de rangos de fechas (puede haber gaps).
    Ej: [0,1,2,5,6] → '2024-01-02→2024-01-04, 2024-01-09→2024-01-10'
    """
    if len(idx_array) == 0:
        return "-"
    dates = X.index[np.array(sorted(idx_array))]
    ranges = []
    start = dates[0]
    prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days > 5:  # gap > 1 semana de trading → nuevo rango
            ranges.append(f"{start.date()}→{prev.date()}")
            start = d
        prev = d
    ranges.append(f"{start.date()}→{prev.date()}")
    return ", ".join(ranges)


def _get_split_timeline(split_info, X, y, fwd_ret, is_pnl, oos_pnl,
                        y_hat_tr, y_hat_te):
    """
    Devuelve una sola serie temporal con TODAS las observaciones del split.
    Columnas: set | y_pred | y_real | fwd_ret | pnl
    set values: 'train' | 'test' | 'purged' | 'embargo'

    purged = estaba en raw_train pero fue eliminado por purging (overlaps con test)
    embargo = estaba en raw_train pero fue eliminado por embargo (después del test)
    """
    final_tr     = split_info["_final_tr"]
    test_idx     = split_info["_test_idx"]
    purged_set   = set(split_info["_purged_idx"])
    embargoed    = set(split_info["_embargoed_idx"])
    final_tr_set = set(final_tr.tolist())
    test_set     = set(test_idx.tolist())

    # Todos los índices que aparecen en este split
    all_idx = sorted(final_tr_set | test_set | purged_set | embargoed)

    rows = []
    tr_iter  = iter(range(len(final_tr)))
    te_iter  = iter(range(len(test_idx)))
    tr_pos   = {int(i): p for p, i in enumerate(final_tr)}
    te_pos   = {int(i): p for p, i in enumerate(test_idx)}

    for i in all_idx:
        date = X.index[i]
        if i in test_set:
            pos = te_pos[i]
            rows.append({
                "set":     "test",
                "y_pred":  int(y_hat_te[pos]),
                "y_real":  int(y.iloc[i]),
                "fwd_ret": fwd_ret.iloc[i],
                "pnl":     oos_pnl.iloc[pos],
            })
        elif i in final_tr_set:
            pos = tr_pos[i]
            rows.append({
                "set":     "train",
                "y_pred":  int(y_hat_tr[pos]),
                "y_real":  int(y.iloc[i]),
                "fwd_ret": fwd_ret.iloc[i],
                "pnl":     is_pnl.iloc[pos],
            })
        elif i in embargoed:  # embargo antes que purged: tiene prioridad visual
            rows.append({
                "set":     "embargo",
                "y_pred":  None,
                "y_real":  int(y.iloc[i]),
                "fwd_ret": fwd_ret.iloc[i],
                "pnl":     None,
            })
        elif i in purged_set:
            rows.append({
                "set":     "purged",
                "y_pred":  None,
                "y_real":  int(y.iloc[i]),
                "fwd_ret": fwd_ret.iloc[i],
                "pnl":     None,
            })
        else:  # no debería llegar aquí
            rows.append({
                "set":     "embargo",
                "y_pred":  None,
                "y_real":  int(y.iloc[i]),
                "fwd_ret": fwd_ret.iloc[i],
                "pnl":     None,
            })

    df = pd.DataFrame(rows, index=pd.Index([X.index[i] for i in all_idx], name="date"))
    return df


def _plot_split_timeline(split_info, X):
    """Bar horizontal: train=azul, test=naranja, purged=amarillo, embargo=rojo."""
    n = len(X)
    colors = ["#3498db"] * n  # default train
    for idx in split_info["_test_idx"]:
        colors[idx] = "#e67e22"
    for idx in split_info["_purged_idx"]:
        colors[idx] = "#f1c40f"
    for idx in split_info["_embargoed_idx"]:
        colors[idx] = "#e74c3c"

    fig, ax = plt.subplots(figsize=(14, 1.5))
    ax.bar(range(n), [1] * n, color=colors, width=1.0, linewidth=0)
    ax.set_yticks([])
    ax.set_xlabel("Índice de observación")
    ax.set_title(
        f"Split {split_info['split_id']}  test_groups={split_info['test_groups']}  "
        f"n_train={split_info['n_train']}  n_test={split_info['n_test']}  "
        f"n_purged={split_info['n_purged']}  n_embargoed={split_info['n_embargoed']}"
    )
    legend = [
        mpatches.Patch(color="#3498db", label="Train"),
        mpatches.Patch(color="#e67e22", label="Test"),
        mpatches.Patch(color="#f1c40f", label="Purged (label overlap)"),
        mpatches.Patch(color="#e74c3c", label="Embargo"),
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
               n_groups=N_GROUPS, k_test=K_TEST,
               pct_embargo=PCT_EMBARGO):
    """
    Debug completo de CPCV. Muestra:
      1. Tabla de splits (train/test/purged/embargo separados)
      2. Serie temporal completa por split
      3. Fórmulas IS vs OOS Sharpe verificables con calculadora
      4. Returns del path 0
      5. Sharpe del path 0 con fórmula
      6. Distribución de Sharpes de paths
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
            "split_id":      s["split_id"],
            "test_groups":   str(s["test_groups"]),
            "train_dates":   _date_ranges(s["_final_tr"], X),
            "test_dates":    _date_ranges(s["_test_idx"], X),
            "purged_dates":  _date_ranges(s["_purged_idx"], X),
            "embargo_dates": _date_ranges(s["_embargoed_idx"], X),
            "n_train":       s["n_train"],
            "n_test":        s["n_test"],
            "n_purged":      s["n_purged"],
            "n_embargoed":   s["n_embargoed"],
        })
    display(pd.DataFrame(rows).set_index("split_id"))

    # ── Sección 3: Serie temporal completa por split ──────────────────────────
    print(f"\n{'=' * 70}")
    print("SECCIÓN 3 — SERIE TEMPORAL COMPLETA POR SPLIT")
    print("  Columnas: set | y_pred | y_real | fwd_ret | pnl")
    print("  set: train / test / embargo/purged")
    print("=" * 70)
    for s in splits_info:
        sid = s["split_id"]
        y_hat_tr, y_hat_te = preds_by_split[sid]
        timeline_df = _get_split_timeline(
            s, X, y, fwd_ret, is_by_split[sid], oos_by_split[sid],
            y_hat_tr, y_hat_te)
        print(f"\n--- Split {sid}  test_groups={s['test_groups']}  "
              f"(n_train={s['n_train']}, n_test={s['n_test']}, n_embargo={s['n_embargoed']}) ---")
        display(timeline_df.style.format({
            "fwd_ret": lambda v: f"{v:.5f}" if v is not None else "-",
            "pnl":     lambda v: f"{v:+.5f}" if v is not None else "-",
        }))

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
                     pct_embargo=PCT_EMBARGO,
                     variant="purge_embargo") -> pd.Series:
    """
    Production-ready. Devuelve pd.Series con el Sharpe de cada path CPCV.
    Sin prints, sin plots.

    variant: "no_purge"      → train = raw_train (sin purge ni embargo)
             "purge"         → train = purged, pctEmbargo=0
             "purge_embargo" → train = purged + embargo (default)

    Formula: SR_path = sqrt(252) * mean(path_pnl) / std(path_pnl)
    """
    if variant == "no_purge":
        # Usar raw_train sin purge: iterar splitter y usar raw_tr directamente
        from itertools import combinations as _comb
        cpcv = CombinatorialPurgedKFold(n_groups, k_test, t1, 0.0)
        oos_by_split = {}
        for split_id, (raw_tr, test_idx, final_tr, test_groups, _, _e) in enumerate(cpcv.split(X)):
            if len(np.unique(y.iloc[raw_tr])) < 2:
                continue
            _, oos_pnl, _, _ = _pnl_from_split(clf, X, y, t1, fwd_ret, raw_tr, test_idx)
            oos_by_split[split_id] = oos_pnl
    elif variant == "purge":
        _, oos_by_split, _, _ = _build_cpcv_splits_table(
            clf, X, y, t1, fwd_ret, n_groups, k_test, 0.0)
    else:  # purge_embargo
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

    return pd.Series(sharpes, name=f"cpcv_{variant}")


def wf_sharpe_dist(clf, X, y, t1, fwd_ret,
                   variant="no_purge",
                   n_splits=6,
                   pct_embargo=PCT_EMBARGO) -> pd.Series:
    """
    Distribución de Sharpes OOS de WalkForward.
    variant: "no_purge" | "purge" | "purge_embargo"
    """
    if variant == "no_purge":
        splitter = WalkForwardCV(n_splits=n_splits, t1=None, pctEmbargo=0.0)
    elif variant == "purge":
        splitter = WalkForwardCV(n_splits=n_splits, t1=t1, pctEmbargo=0.0)
    else:  # purge_embargo
        splitter = WalkForwardCV(n_splits=n_splits, t1=t1, pctEmbargo=pct_embargo)

    sharpes = []
    for train_idx, test_idx in splitter.split(X):
        if len(train_idx) < 5 or len(test_idx) < 2:
            continue
        if len(np.unique(y.iloc[train_idx])) < 2:
            continue
        _, oos_pnl, _, _ = _pnl_from_split(
            clf, X, y, t1, fwd_ret, train_idx, test_idx)
        sharpes.append(_fold_sharpe(oos_pnl))

    return pd.Series(sharpes, name=f"wf_{variant}")


def kfold_sharpe_dist(clf, X, y, t1, fwd_ret,
                      variant="no_purge",
                      n_splits=6,
                      pct_embargo=PCT_EMBARGO) -> pd.Series:
    """
    Distribución de Sharpes OOS de KFold.
    variant: "no_purge" | "purge" | "purge_embargo"
    """
    if variant == "no_purge":
        splitter = KFold(n_splits=n_splits, shuffle=False)
        split_iter = splitter.split(X)
    elif variant == "purge":
        splitter = PurgedKFold(n_splits=n_splits, t1=t1, pctEmbargo=0.0)
        split_iter = splitter.split(X)
    else:  # purge_embargo
        splitter = PurgedKFold(n_splits=n_splits, t1=t1, pctEmbargo=pct_embargo)
        split_iter = splitter.split(X)

    sharpes = []
    for train_idx, test_idx in split_iter:
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        if len(train_idx) < 5 or len(test_idx) < 2:
            continue
        if len(np.unique(y.iloc[train_idx])) < 2:
            continue
        _, oos_pnl, _, _ = _pnl_from_split(
            clf, X, y, t1, fwd_ret, train_idx, test_idx)
        sharpes.append(_fold_sharpe(oos_pnl))

    return pd.Series(sharpes, name=f"kfold_{variant}")


def compare_methods(clf, X, y, t1, fwd_ret,
                    n_splits=6,
                    n_groups=N_GROUPS, k_test=K_TEST,
                    pct_embargo=PCT_EMBARGO) -> dict:
    """
    Corre los 3 métodos × 3 variantes (no purge / purge / purge+embargo) + CPCV.
    Plot: 3 paneles (KFold family | WalkForward family | CPCV family),
          cada uno con violin+boxplot por variante.
    Devuelve dict con todas las pd.Series de Sharpes.
    """
    results = {
        "kfold_no_purge":      kfold_sharpe_dist(clf, X, y, t1, fwd_ret, variant="no_purge",      n_splits=n_splits),
        "kfold_purge":         kfold_sharpe_dist(clf, X, y, t1, fwd_ret, variant="purge",         n_splits=n_splits, pct_embargo=pct_embargo),
        "kfold_purge_embargo": kfold_sharpe_dist(clf, X, y, t1, fwd_ret, variant="purge_embargo", n_splits=n_splits, pct_embargo=pct_embargo),
        "wf_no_purge":         wf_sharpe_dist(clf, X, y, t1, fwd_ret, variant="no_purge",      n_splits=n_splits),
        "wf_purge":            wf_sharpe_dist(clf, X, y, t1, fwd_ret, variant="purge",         n_splits=n_splits, pct_embargo=pct_embargo),
        "wf_purge_embargo":    wf_sharpe_dist(clf, X, y, t1, fwd_ret, variant="purge_embargo", n_splits=n_splits, pct_embargo=pct_embargo),
        "cpcv_no_purge":       cpcv_sharpe_dist(clf, X, y, t1, fwd_ret, n_groups, k_test, pct_embargo, variant="no_purge"),
        "cpcv_purge":          cpcv_sharpe_dist(clf, X, y, t1, fwd_ret, n_groups, k_test, pct_embargo, variant="purge"),
        "cpcv_purge_embargo":  cpcv_sharpe_dist(clf, X, y, t1, fwd_ret, n_groups, k_test, pct_embargo, variant="purge_embargo"),
    }

    families = [
        ("KFold family",       ["kfold_no_purge", "kfold_purge", "kfold_purge_embargo"],  "#5b9bd5"),
        ("WalkForward family", ["wf_no_purge",    "wf_purge",    "wf_purge_embargo"],     "#e06c75"),
        ("CCV / CPCV family",  ["cpcv_no_purge",  "cpcv_purge",  "cpcv_purge_embargo"],   "#6dbf8b"),
    ]
    variant_labels = ["No purge", "Purge", "Purge+Embargo"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    fig.suptitle(
        "OOS Sharpe Distributions by Validation Family\n"
        "(fold-level for CV/WalkForward; path-level for CPCV)",
        fontsize=13, fontweight="bold"
    )

    for ax, (family_name, keys, color) in zip(axes, families):
        data = [results[k].dropna().values for k in keys]
        positions = [1, 2, 3]

        # Violin
        parts = ax.violinplot(data, positions=positions, showmedians=False,
                              showextrema=False, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor(color)

        # Boxplot
        bp = ax.boxplot(data, positions=positions, widths=0.25,
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color="white", linewidth=2),
                        boxprops=dict(facecolor="white", color=color, linewidth=1.5),
                        whiskerprops=dict(color="gray", linewidth=1),
                        capprops=dict(color="gray", linewidth=1),
                        flierprops=dict(marker="o", markerfacecolor=color,
                                        markeredgewidth=0, markersize=4, alpha=0.7))

        # Median labels
        for pos, d in zip(positions, data):
            if len(d):
                med = float(np.median(d))
                ax.text(pos, med, f"med={med:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color="#333333")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(family_name, fontsize=11, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(variant_labels, fontsize=9)
        ax.set_xlabel("Validation variant", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_facecolor("#f9f9f9")

    axes[0].set_ylabel("OOS Sharpe Ratio per test unit", fontsize=9)
    plt.tight_layout()
    plt.show()

    # Tabla resumen
    summary_rows = {}
    for k, s in results.items():
        s = s.dropna()
        summary_rows[k] = {
            "n":            len(s),
            "mean":         round(float(s.mean()), 3) if len(s) else None,
            "median":       round(float(np.median(s)), 3) if len(s) else None,
            "std":          round(float(s.std()), 3) if len(s) else None,
            "pct_positive": round(float((s > 0).mean()), 2) if len(s) else None,
        }
    display(pd.DataFrame(summary_rows).T)

    return results
