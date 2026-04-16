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

from sklearn.model_selection import KFold

from cpcv_analysis.splitters import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    WalkForwardCV,
)
from cpcv_analysis.cv_runner import get_paths, run_cpcv
from cpcv_analysis.config import N_GROUPS, K_TEST, PCT_EMBARGO


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
