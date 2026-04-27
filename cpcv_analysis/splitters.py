# cpcv_analysis/splitters.py
"""
Core splitters faithful to De Prado (2018).

  getTrainTimes          — verbatim De Prado
  getEmbargoTimes        — verbatim De Prado
  PurgedKFold            — verbatim De Prado
  CombinatorialPurgedKFold — near-original adaptation (structurally identical)
  WalkForwardCV          — new, same interface
"""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold

try:
    from sklearn.model_selection import _BaseKFold
except ImportError:
    from sklearn.model_selection._split import _BaseKFold


# ── De Prado verbatim ──────────────────────────────────────────────────────

def getTrainTimes(t1: pd.Series, testTimes: pd.Series) -> pd.Series:
    """
    Remove from t1 those entries whose label overlaps with testTimes.
    De Prado (2018), Advances in Financial Machine Learning, Snippet 7.1.
    """
    trn = t1.copy()
    for i, j in testTimes.items():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index  # train start in test
        df1 = trn[(i <= trn) & (trn <= j)].index              # train end in test
        df2 = trn[(trn.index <= i) & (j <= trn)].index        # test in train
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def getEmbargoTimes(times: pd.Index, pctEmbargo: float) -> pd.Series:
    """
    Map each timestamp to the first timestamp post-embargo.
    De Prado (2018), Snippet 7.2.
    """
    step = int(len(times) * pctEmbargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = mbrg.reindex(times).ffill()
    mbrg.iloc[-step:] = times[-1]
    return mbrg


class PurgedKFold(_BaseKFold):
    """
    KFold with purging and embargo. Verbatim De Prado (2018), Snippet 7.3.
    """
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.0):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1         = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if self.t1 is None:
            raise ValueError("t1 must be provided")
        indices = np.arange(X.shape[0])
        mbrg    = getEmbargoTimes(X.index, self.pctEmbargo)
        test_starts = [
            (i[0], i[-1] + 1)
            for i in np.array_split(indices, self.n_splits)
        ]
        for i, j in test_starts:
            t0       = X.index[i]
            test_idx = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[X.index[test_idx]].max())
            trainTimes = getTrainTimes(
                self.t1,
                pd.Series(index=X.index[test_idx],
                          data=self.t1[X.index[test_idx]].values)
            )
            if self.pctEmbargo > 0:
                embargoTime = mbrg[X.index[test_idx[-1]]]
                trainTimes  = trainTimes[trainTimes.index < embargoTime]
            train_idx = indices[X.index.isin(trainTimes.index)]
            yield train_idx, test_idx


# ── Near-original adaptation ───────────────────────────────────────────────

class CombinatorialPurgedKFold:
    """
    Combinatorial Purged KFold — near-original adaptation of De Prado (2018).
    Generates C(N, k) splits. Each split: k groups as test, rest as train
    after purge + embargo.
    """
    def __init__(self, n_groups: int, n_test_groups: int,
                 t1: pd.Series, pctEmbargo: float = 0.0):
        self.N          = n_groups
        self.k          = n_test_groups
        self.t1         = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X):
        indices = np.arange(len(X))
        groups  = np.array_split(indices, self.N)
        mbrg    = getEmbargoTimes(X.index, self.pctEmbargo)

        for test_groups in combinations(range(self.N), self.k):
            test_idx      = np.sort(np.concatenate([groups[g] for g in test_groups]))
            raw_train_idx = np.sort(np.concatenate([groups[g] for g in range(self.N)
                                            if g not in test_groups]))

            # Step 1: purge by label overlap
            testTimes = pd.Series(self.t1.iloc[test_idx].values, index=X.index[test_idx])
            train_t1   = self.t1.iloc[raw_train_idx]
            trainTimes = getTrainTimes(train_t1, testTimes)
            purged_only_times = set(train_t1.index) - set(trainTimes.index)

            # Step 2-3: embargo — Corregido para ser aditivo
            embargoed_times = set()
            if self.pctEmbargo > 0:
                block_ends = np.append(test_idx[np.where(np.diff(test_idx) > 1)[0]], test_idx[-1])
                for end in block_ends:
                    # CAMBIO: Usamos t1 (fin del trade) en lugar del index (inicio del trade)
                    t1_end = self.t1.iloc[end]
                    # Buscamos el punto de embargo correspondiente a ese final
                    embargoTime = mbrg.iloc[X.index.searchsorted(t1_end, side='left') - 1]
                    in_embargo = train_t1[(train_t1.index > t1_end) & (train_t1.index <= embargoTime)].index
                    embargoed_times.update(in_embargo)
                
                trainTimes = trainTimes[~trainTimes.index.isin(embargoed_times)]

            final_train_idx = indices[X.index.isin(trainTimes.index)]
            purged_only_idx = indices[X.index.isin(purged_only_times - embargoed_times)]
            embargoed_idx   = indices[X.index.isin(embargoed_times)]
            yield raw_train_idx, test_idx, final_train_idx, test_groups, purged_only_idx, embargoed_idx

# ── Rolling Walk-Forward (fixed train window, slides by test_size) ────────

class RollingWalkForwardCV:
    """
    Rolling-window walk-forward CV: train window size is fixed, slides forward
    by test_size each fold.

    Folds are defined by explicit date boundaries:
        fold_dates = [(train_start, train_end, test_start, test_end), ...]

    If fold_dates is None, splits are computed from train_months/test_months
    relative to X.index using calendar arithmetic.
    """
    def __init__(self, fold_dates: list, t1: pd.Series = None,
                 pctEmbargo: float = 0.0):
        """
        fold_dates: list of (train_start, train_end, test_start, test_end) strings.
        """
        self.fold_dates  = fold_dates
        self.t1          = t1
        self.pctEmbargo  = pctEmbargo

    def split(self, X):
        mbrg = getEmbargoTimes(X.index, self.pctEmbargo) if self.pctEmbargo > 0 else None
        indices = np.arange(len(X))

        for tr_s, tr_e, te_s, te_e in self.fold_dates:
            tr_mask = (X.index >= pd.Timestamp(tr_s)) & (X.index < pd.Timestamp(tr_e))
            te_mask = (X.index >= pd.Timestamp(te_s)) & (X.index < pd.Timestamp(te_e))
            train_idx = indices[tr_mask]
            test_idx  = indices[te_mask]

            if len(train_idx) < 2 or len(test_idx) < 1:
                continue

            if self.t1 is not None:
                train_t1  = self.t1.iloc[train_idx]
                testTimes = pd.Series(
                    self.t1.iloc[test_idx].values,
                    index=X.index[test_idx]
                )
                trainTimes = getTrainTimes(train_t1, testTimes)
                if self.pctEmbargo > 0 and mbrg is not None:
                    embargoTime = mbrg[X.index[test_idx[-1]]]
                    trainTimes  = trainTimes[trainTimes.index < embargoTime]
                train_idx = indices[X.index.isin(trainTimes.index)]

            yield train_idx, test_idx


# ── Walk-Forward (new, same interface convention) ─────────────────────────

class WalkForwardCV:
    """
    Expanding-window walk-forward CV with optional purge and embargo.
    min_train_size: minimum number of observations in first training window.
    """
    def __init__(self, n_splits: int, t1: pd.Series = None,
                 pctEmbargo: float = 0.0, min_train_frac: float = 0.5):
        self.n_splits       = n_splits
        self.t1             = t1
        self.pctEmbargo     = pctEmbargo
        self.min_train_frac = min_train_frac

    def split(self, X):
        n      = len(X)
        min_tr = int(n * self.min_train_frac)
        step   = (n - min_tr) // self.n_splits
        mbrg   = getEmbargoTimes(X.index, self.pctEmbargo) if self.pctEmbargo > 0 else None

        for i in range(self.n_splits):
            test_start = min_tr + i * step
            test_end   = min_tr + (i + 1) * step if i < self.n_splits - 1 else n
            test_idx   = np.arange(test_start, test_end)
            train_idx  = np.arange(0, test_start)

            if self.t1 is not None:
                train_t1  = self.t1.iloc[train_idx]
                testTimes = pd.Series(
                    self.t1.iloc[test_idx].values,
                    index=X.index[test_idx]
                )
                trainTimes = getTrainTimes(train_t1, testTimes)
                if self.pctEmbargo > 0 and mbrg is not None:
                    embargoTime = mbrg[X.index[test_idx[-1]]]
                    trainTimes  = trainTimes[trainTimes.index < embargoTime]
                train_idx = np.where(X.index.isin(trainTimes.index))[0]

            yield train_idx, test_idx
