# tests/test_splitters.py
"""
Property tests for splitters and paths.
Fixtures are tiny synthetic datasets with hardcoded expected values
discovered by the audit script (audit_theory.py).

Run: conda run -n rappi python3 -m pytest tests/test_splitters.py -v
"""
import numpy as np
import pandas as pd
import pytest
from collections import Counter
from itertools import combinations

from cpcv_analysis.splitters import (
    getTrainTimes, getEmbargoTimes,
    CombinatorialPurgedKFold, PurgedKFold,
)
from cpcv_analysis.cv_runner import get_paths


# ── helpers ───────────────────────────────────────────────────────────────────

def _idx(n, start="2020-01-02"):
    return pd.bdate_range(start=start, periods=n)


def _t1(idx, horizon=5):
    t1 = pd.Series(idx.shift(horizon, freq="B"), index=idx)
    return t1.clip(upper=idx[-1])


# ── A1: purge leaves zero overlap ─────────────────────────────────────────────

def test_purge_no_overlap():
    """After getTrainTimes, no train obs has t1 overlapping the test window."""
    idx = _idx(60)
    t1 = _t1(idx)
    # Test window: last 15 obs
    test_slice = idx[45:]
    testTimes = pd.Series(t1.loc[test_slice].values, index=test_slice)
    purged = getTrainTimes(t1, testTimes)

    test_start = test_slice[0]
    test_end = testTimes.max()

    # No purged train obs should have index or t1 inside [test_start, test_end]
    index_in_test = purged[(purged.index >= test_start) & (purged.index <= test_end)]
    t1_in_test = purged[(purged >= test_start) & (purged <= test_end)]
    assert len(index_in_test) == 0, f"Train indices overlap test: {index_in_test.index.tolist()}"
    assert len(t1_in_test) == 0, f"Train t1 values overlap test: {t1_in_test.index.tolist()}"


# ── A2: embargo step size ─────────────────────────────────────────────────────

def test_embargo_exact_step():
    """getEmbargoTimes with pctEmbargo=0.05 on 100 obs: first obs maps to idx[5]."""
    idx = _idx(100)
    pct = 0.05
    step = int(len(idx) * pct)  # = 5
    mbrg = getEmbargoTimes(idx, pct)
    # The first obs should map forward by exactly `step` positions
    assert mbrg.iloc[0] == idx[step], (
        f"Expected first embargo target = {idx[step].date()}, got {mbrg.iloc[0].date()}"
    )


# ── A3: CPCV produces C(N,k) splits ──────────────────────────────────────────

def test_cpcv_n_splits():
    """CombinatorialPurgedKFold(N=6, k=2) yields exactly C(6,2)=15 splits."""
    idx = _idx(60)
    t1 = _t1(idx)
    X = pd.DataFrame(np.zeros((60, 1)), index=idx, columns=["f"])
    splitter = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, t1=t1, pctEmbargo=0.0)
    splits = list(splitter.split(X))
    assert len(splits) == 15, f"Expected 15 splits, got {len(splits)}"


# ── A4: test coverage = C(N-1, k-1) per observation ──────────────────────────

def test_cpcv_test_coverage():
    """Each of the 60 obs appears in test exactly C(5,1)=5 times."""
    idx = _idx(60)
    t1 = _t1(idx)
    X = pd.DataFrame(np.zeros((60, 1)), index=idx, columns=["f"])
    splitter = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, t1=t1, pctEmbargo=0.0)
    counter = Counter()
    for _, test_idx, _, _ in splitter.split(X):
        for i in test_idx:
            counter[i] += 1
    counts = set(counter.values())
    assert counts == {5}, f"Expected every obs to appear 5 times, got counts: {dict(Counter(counter.values()))}"


# ── B1: get_paths returns phi paths ──────────────────────────────────────────

def test_paths_count():
    """get_paths(6, 2) returns exactly phi = C(6,2)*2/6 = 5 paths."""
    from math import comb
    N, k = 6, 2
    phi = comb(N, k) * k // N  # = 5
    paths = get_paths(N, k)
    assert len(paths) == phi, f"Expected {phi} paths, got {len(paths)}"


# ── B2: each path covers all groups exactly once ──────────────────────────────

def test_paths_cover_all_groups():
    """Each path's split_ids cover groups {0,1,2,3,4,5} exactly once."""
    N, k = 6, 2
    all_combos = list(combinations(range(N), k))
    paths = get_paths(N, k)
    for pid, split_ids in enumerate(paths):
        groups_in_path = []
        for sid in split_ids:
            groups_in_path.extend(all_combos[sid])
        assert sorted(groups_in_path) == list(range(N)), (
            f"Path {pid} groups {groups_in_path} != {list(range(N))}"
        )


# ── B3: paths are disjoint (no shared split_id) ───────────────────────────────

def test_paths_disjoint():
    """No split_id appears in more than one path."""
    paths = get_paths(6, 2)
    all_ids = [sid for path in paths for sid in path]
    assert len(all_ids) == len(set(all_ids)), (
        f"Duplicate split_ids across paths: {[x for x in all_ids if all_ids.count(x) > 1]}"
    )


# ── PurgedKFold: train ends before test starts ────────────────────────────────

def test_purged_kfold_n_splits():
    """PurgedKFold(n_splits=6) yields exactly 6 folds."""
    idx = _idx(60)
    t1 = _t1(idx)
    X = pd.DataFrame(np.zeros((60, 1)), index=idx, columns=["f"])
    splitter = PurgedKFold(n_splits=6, t1=t1, pctEmbargo=0.0)
    folds = list(splitter.split(X))
    assert len(folds) == 6, f"Expected 6 folds, got {len(folds)}"
