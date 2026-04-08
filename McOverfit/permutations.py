import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from .core import grid_search_rf, annualized_sr, equity_curve
from . import cache

DEFAULT_N_JOBS = max(1, (os.cpu_count() or 2) // 2)


def _is_worker(i, X_is, y_is, fwd_is, param_grid, seed):
    rng = np.random.default_rng(seed + i)
    fwd_perm = fwd_is[rng.permutation(len(fwd_is))]
    _, sr, eq = grid_search_rf(X_is, y_is, fwd_perm, param_grid, seed=i)
    return sr, eq


def _wf_worker(i, X_is, y_is, fwd_is, X_oos, fwd_oos, param_grid, seed):
    rng = np.random.default_rng(seed + 1 + i)
    fwd_perm = fwd_is[rng.permutation(len(fwd_is))]
    p_params, _, _ = grid_search_rf(X_is, y_is, fwd_perm, param_grid, seed=i)
    clf = RandomForestClassifier(**p_params, random_state=i, n_jobs=1)
    clf.fit(X_is, y_is)
    oos_pnl = (2 * clf.predict(X_oos) - 1) * fwd_oos
    return annualized_sr(oos_pnl), equity_curve(oos_pnl)


def run_is(X_is, y_is, fwd_is, param_grid, n_perm, is_frac, seed=42, n_jobs=DEFAULT_N_JOBS):
    """Run IS permutations. Returns cached result if available."""
    cached = cache.load("is", n_perm, is_frac, seed)
    if cached is not None:
        print(f"[cache] loaded IS  N={n_perm} IS_FRAC={is_frac} seed={seed}")
        return cached

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_is_worker)(i, X_is, y_is, fwd_is, param_grid, seed)
        for i in tqdm(range(n_perm), desc="IS permutations")
    )
    data = {
        "sharpes": np.array([r[0] for r in results]),
        "curves":  [r[1] for r in results],
    }
    cache.save("is", n_perm, is_frac, seed, data)
    return data


def run_wf(X_is, y_is, fwd_is, X_oos, fwd_oos, param_grid, n_perm, is_frac, seed=42, n_jobs=DEFAULT_N_JOBS):
    """Run WalkForward permutations. Returns cached result if available."""
    cached = cache.load("wf", n_perm, is_frac, seed)
    if cached is not None:
        print(f"[cache] loaded WF  N={n_perm} IS_FRAC={is_frac} seed={seed}")
        return cached

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_wf_worker)(i, X_is, y_is, fwd_is, X_oos, fwd_oos, param_grid, seed)
        for i in tqdm(range(n_perm), desc="WF permutations")
    )
    data = {
        "sharpes": np.array([r[0] for r in results]),
        "curves":  [r[1] for r in results],
    }
    cache.save("wf", n_perm, is_frac, seed, data)
    return data
