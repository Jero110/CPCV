import numpy as np
from sklearn.ensemble import RandomForestClassifier


def annualized_sr(ret: np.ndarray, periods: int = 252) -> float:
    if len(ret) < 2 or ret.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * ret.mean() / ret.std())


def equity_curve(pnl: np.ndarray) -> np.ndarray:
    return np.cumsum(pnl)


def grid_search_rf(X_tr, y_tr, fwd_tr, param_grid: list, seed: int = 42) -> tuple:
    """Returns (best_params, best_sharpe, best_equity_curve)."""
    best_sr = -np.inf
    best_pnl = None
    best_params = None

    for params in param_grid:
        clf = RandomForestClassifier(**params, random_state=seed, n_jobs=1)
        clf.fit(X_tr, y_tr)
        y_hat = clf.predict(X_tr)
        pnl = (2 * y_hat - 1) * fwd_tr
        sr = annualized_sr(pnl)
        if sr > best_sr:
            best_sr = sr
            best_pnl = pnl.copy()
            best_params = params

    return best_params, best_sr, equity_curve(best_pnl)
