# tests/audit_theory.py
"""
Iterative audit: verifies CPCV implementation against De Prado (2018) theory.
Run standalone: conda run -n rappi python3 tests/audit_theory.py

Prints PASS / FAIL / WARN per property with observed vs expected values.
Re-run freely while iterating on the code.
"""
import numpy as np
import pandas as pd
from itertools import combinations

from cpcv_analysis.splitters import (
    getTrainTimes, getEmbargoTimes,
    CombinatorialPurgedKFold, PurgedKFold,
)
from cpcv_analysis.cv_runner import get_paths


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2020-01-02", periods=n)


def _result(label, passed, observed, expected, warn=False):
    status = "PASS" if passed else ("WARN" if warn else "FAIL")
    symbol = "✓" if passed else ("⚠" if warn else "✗")
    print(f"  [{symbol} {status}] {label}")
    print(f"         observed={observed}  expected={expected}")
    return passed


results = []


def check(label, passed, observed, expected, warn=False):
    ok = _result(label, passed, observed, expected, warn=warn)
    results.append((label, ok, warn))
    return ok


# ── Block A — Splitters ───────────────────────────────────────────────────────

def audit_block_a():
    print("\n══ Block A — Splitters ══════════════════════════════════════════")
    N = 100
    idx = _make_index(N)
    t1 = pd.Series(idx.shift(5, freq="B"), index=idx).clip(upper=idx[-1])
    X = pd.DataFrame(np.zeros((N, 1)), index=idx, columns=["f"])

    # A1: getTrainTimes leaves 0 overlapping observations
    test_slice = idx[40:60]
    testTimes = pd.Series(t1.loc[test_slice].values, index=test_slice)
    purged = getTrainTimes(t1, testTimes)
    test_start = test_slice[0]
    test_end = testTimes.max()
    overlap = purged[
        (purged.index >= test_start) & (purged.index <= test_end) |
        (purged >= test_start) & (purged <= test_end)
    ]
    check("A1: getTrainTimes leaves 0 overlapping obs",
          len(overlap) == 0, observed=len(overlap), expected=0)

    # A2: getEmbargoTimes embargoes exactly step obs
    pct = 0.05
    step = int(N * pct)
    mbrg = getEmbargoTimes(idx, pct)
    first_embargo_target = mbrg.iloc[0]
    check("A2: getEmbargoTimes — first obs maps to idx[step]",
          first_embargo_target == idx[step],
          observed=first_embargo_target.date(),
          expected=idx[step].date())

    # A3: CombinatorialPurgedKFold(N=6, k=2) yields C(6,2)=15 splits
    N6 = 60
    idx6 = _make_index(N6)
    t1_6 = pd.Series(idx6.shift(5, freq="B"), index=idx6).clip(upper=idx6[-1])
    X6 = pd.DataFrame(np.zeros((N6, 1)), index=idx6, columns=["f"])
    splitter = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, t1=t1_6, pctEmbargo=0.0)
    splits = list(splitter.split(X6))
    check("A3: C(6,2)=15 splits generated",
          len(splits) == 15, observed=len(splits), expected=15)

    # A4: each obs appears in test exactly C(5,1)=5 times
    from collections import Counter
    test_counts = Counter()
    for _, test_idx, _, _ in splits:
        for i in test_idx:
            test_counts[i] += 1
    unique_counts = set(test_counts.values())
    all_five = unique_counts == {5}
    check("A4: each obs in test exactly C(5,1)=5 times",
          all_five, observed=dict(Counter(test_counts.values())), expected={5: N6})


# ── Block B — Paths ───────────────────────────────────────────────────────────

def audit_block_b():
    print("\n══ Block B — Paths ══════════════════════════════════════════════")
    from cpcv_analysis.config import N_GROUPS, K_TEST

    paths = get_paths(N_GROUPS, K_TEST)  # N=6, k=2

    # B1: phi = C(6,2)*2/6 = 5 paths
    from math import comb
    phi = comb(N_GROUPS, K_TEST) * K_TEST // N_GROUPS
    check("B1: get_paths returns phi=5 paths",
          len(paths) == phi, observed=len(paths), expected=phi)

    # B2: each path covers all N groups exactly once
    all_combos = list(combinations(range(N_GROUPS), K_TEST))
    bad_paths = []
    for pid, split_ids in enumerate(paths):
        groups_in_path = set()
        for sid in split_ids:
            for g in all_combos[sid]:
                groups_in_path.add(g)
        if groups_in_path != set(range(N_GROUPS)):
            bad_paths.append(pid)
    check("B2: each path covers all N groups exactly once",
          len(bad_paths) == 0, observed=f"{len(bad_paths)} bad paths", expected="0 bad paths")

    # B3: no split_id shared between paths
    all_split_ids = [sid for path in paths for sid in path]
    n_unique = len(set(all_split_ids))
    n_total = len(all_split_ids)
    check("B3: all split_ids are disjoint across paths",
          n_unique == n_total, observed=f"{n_total} total, {n_unique} unique", expected="all unique")

    # B4: Path SR from concat series vs mean of fold SRs — report difference
    import numpy as np
    from cpcv_analysis.cv_runner import _annualized_sr
    rng = np.random.default_rng(42)
    N6 = 60
    idx6 = _make_index(N6)
    t1_6 = pd.Series(idx6.shift(5, freq="B"), index=idx6).clip(upper=idx6[-1])
    X6 = pd.DataFrame(rng.standard_normal((N6, 2)), index=idx6, columns=["f1", "f2"])
    splitter = CombinatorialPurgedKFold(6, 2, t1_6, 0.0)

    fake_oos = {}
    for split_id, (_, test_idx, _, _) in enumerate(splitter.split(X6)):
        fake_oos[split_id] = pd.Series(
            rng.normal(0, 0.01, size=len(test_idx)), index=X6.index[test_idx]
        )

    path0 = paths[0]
    concat_pnl = pd.concat([fake_oos[s] for s in path0]).sort_index()
    sr_concat = _annualized_sr(concat_pnl)
    sr_mean   = float(np.mean([_annualized_sr(fake_oos[s]) for s in path0]))
    diff = abs(sr_concat - sr_mean)
    warn = diff > 0.5
    check("B4: path SR (concat) vs mean of fold SRs — diff reported",
          True,
          observed=f"concat SR={sr_concat:.3f}, mean SR={sr_mean:.3f}, diff={diff:.3f}",
          expected="diff reported (WARN if >0.5)",
          warn=warn)


# ── Block C — OOS Degradation ─────────────────────────────────────────────────

def audit_block_c():
    print("\n══ Block C — OOS Degradation ════════════════════════════════════")
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    from cpcv_analysis.cv_runner import run_cpcv, _annualized_sr
    from cpcv_analysis.config import XGB_PARAMS
    import numpy as np

    rng = np.random.default_rng(0)
    N = 120
    idx = _make_index(N)
    t1 = pd.Series(idx.shift(5, freq="B"), index=idx).clip(upper=idx[-1])
    X = pd.DataFrame(rng.standard_normal((N, 3)), index=idx, columns=["f1", "f2", "f3"])
    y = pd.Series(rng.integers(0, 2, size=N), index=idx)
    fwd_ret = pd.Series(rng.normal(0, 0.01, size=N), index=idx)

    clf = XGBClassifier(**XGB_PARAMS)
    splitter = CombinatorialPurgedKFold(6, 2, t1, pctEmbargo=0.01)
    fold_results, path_results, _ = run_cpcv(clf, X, y, t1, splitter, verbose=False, fwd_ret=fwd_ret)

    is_srs  = [f["is_sharpe"] for f in fold_results]
    oos_srs = [f["sharpe"]    for f in fold_results]
    pct_is_gt_oos = np.mean([i > o for i, o in zip(is_srs, oos_srs)])

    check("C1: IS_SR > OOS_SR in >70% of folds",
          pct_is_gt_oos > 0.7,
          observed=f"{pct_is_gt_oos:.1%}",
          expected=">70%")

    from scipy import stats as sp_stats
    slope, intercept, r, p, se = sp_stats.linregress(is_srs, oos_srs)
    check("C2: regression slope IS→OOS < 1",
          slope < 1.0,
          observed=f"slope={slope:.3f}",
          expected="< 1.0")

    mean_is = np.mean(is_srs)
    readable = mean_is <= 15
    check("C3: mean IS_SR ≤ 15 (readable scatter plot)",
          readable,
          observed=f"mean IS_SR={mean_is:.2f}",
          expected="≤ 15",
          warn=not readable)


# ── Block D — Rank Logits & PBO ───────────────────────────────────────────────

def audit_block_d():
    print("\n══ Block D — Rank Logits & PBO ══════════════════════════════════")
    from xgboost import XGBClassifier
    from cpcv_analysis.cv_runner import run_cpcv
    from cpcv_analysis.comparison import run_all_methods
    from cpcv_analysis.advanced_analysis import rank_logits
    from cpcv_analysis.config import XGB_PARAMS
    import numpy as np

    rng = np.random.default_rng(0)
    N = 120
    idx = _make_index(N)
    t1 = pd.Series(idx.shift(5, freq="B"), index=idx).clip(upper=idx[-1])
    X = pd.DataFrame(rng.standard_normal((N, 3)), index=idx, columns=["f1", "f2", "f3"])
    y = pd.Series(rng.integers(0, 2, size=N), index=idx)
    fwd_ret = pd.Series(rng.normal(0, 0.01, size=N), index=idx)

    print("  [running all methods for rank logit audit — ~30s]")
    _, all_folds_df = run_all_methods(X, y, t1, fwd_ret=fwd_ret)

    logits, prob_overfit, methods = rank_logits(all_folds_df, n_trials=500, random_state=0)

    std_logits = float(np.std(logits))
    check("D1: logit std > 0.1 (non-trivial variance)",
          std_logits > 0.1,
          observed=f"std={std_logits:.3f}",
          expected="> 0.1")

    check("D2: Prob[Overfit] in (0.05, 0.95)",
          0.05 < prob_overfit < 0.95,
          observed=f"PBO={prob_overfit:.3f}",
          expected="(0.05, 0.95)")

    unit_used = "fold-level SR (all_folds_df rows)"
    de_prado_unit = "path-level SR (one SR per strategy path)"
    is_path_level = False
    check("D3: rank logit unit is path-level SR (De Prado ch.11)",
          is_path_level,
          observed=unit_used,
          expected=de_prado_unit,
          warn=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  CPCV Audit — De Prado (2018) Fidelity Report")
    print("═" * 60)

    audit_block_a()
    audit_block_b()
    audit_block_c()
    audit_block_d()

    print("\n══ Summary ══════════════════════════════════════════════════════")
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    for label, ok, warn in results:
        if ok:
            symbol = "✓"
        elif warn:
            symbol = "⚠"
        else:
            symbol = "✗"
        print(f"  [{symbol}] {label}")
    print(f"\n  {passed}/{total} checks passed")
    if passed == total:
        print("  All checks PASS — implementation faithful to De Prado.")
    else:
        print("  Fix FAIL items before writing tests.")
