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
