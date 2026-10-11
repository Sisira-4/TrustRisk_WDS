"""Shared code for the V3 residual detector.

This module deliberately contains NO reference to the 2017 test file. Only
run_test_once.py may read it (enforced by tests/test_v3_guard.py).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]            # repo root (…/experiments/v3_residual/common.py -> repo)
TARGET, STAMP = "ATT_FLAG", "DATETIME"


def load_series(path):
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = d.columns.str.strip()
    return d


def sensor_columns(df):
    return [c for c in df.columns if c not in (TARGET, STAMP)]


def runs(mask):
    """Contiguous True runs as (start, end) index pairs (inclusive)."""
    idx = np.flatnonzero(np.asarray(mask))
    if idx.size == 0:
        return []
    return [(int(g[0]), int(g[-1])) for g in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)]


def margin_free_pool(labels, margin=48):
    """Rows labelled -999 (unlabelled, assumed normal) that lie farther than
    `margin` hours from every labelled attack. -999 is NOT a confirmed-normal label."""
    labels = np.asarray(labels)
    near = np.zeros(len(labels), bool)
    for s, e in runs(labels == 1):
        near[max(0, s - margin): e + margin + 1] = True
    pool = np.flatnonzero((labels == -999) & ~near)
    return pool[pool > 0]


def persist(alarm, k=3, n=6):
    """CAUSAL k-of-n persistence: output at t depends only on alarms at t-n+1..t."""
    a = pd.Series(np.asarray(alarm, int))
    return (a.rolling(n, min_periods=1).sum().values >= k).astype(int)


class ResidualDetector:
    """One ridge regressor per sensor (predict it from the other sensors plus all
    sensors at t-1), trained on normal-ish rows only. Anomaly score at time t is the
    mean of the `n_top` largest per-sensor residual z-scores."""

    def __init__(self, alpha=1.0, n_top=3):
        self.alpha, self.n_top = alpha, n_top

    # -- helpers
    def _z(self, X):
        return ((X - self.mu) / self.sd)[:, self.keep]

    @staticmethod
    def _lagged(Z):
        return np.vstack([Z[:1], Z[:-1]])

    def _design(self, Z, Zl, j):
        others = [k for k in range(Z.shape[1]) if k != j]
        return np.hstack([Z[:, others], Zl])

    def _abs_residuals(self, Z):
        Zl = self._lagged(Z)
        R = np.empty_like(Z)
        for j, m in enumerate(self.models):
            R[:, j] = np.abs(Z[:, j] - m.predict(self._design(Z, Zl, j)))
        return R

    def _top(self, A):
        return np.sort(A, axis=1)[:, -self.n_top:].mean(axis=1)

    # -- API
    def fit(self, X, train_idx, names=None):
        X = np.asarray(X, float)
        tr = np.asarray(train_idx)
        self.mu = X[tr].mean(0)
        sd = X[tr].std(0)
        self.keep = sd > 1e-6
        self.sd = np.where(self.keep, sd, 1.0)
        self.names = [n for n, k in zip(names, self.keep) if k] if names is not None else None
        Z = self._z(X)
        Zl = self._lagged(Z)
        self.models = []
        for j in range(Z.shape[1]):
            F = self._design(Z, Zl, j)
            self.models.append(Ridge(alpha=self.alpha).fit(F[tr], Z[tr, j]))
        R = self._abs_residuals(Z)
        self.r_mu = R[tr].mean(0)
        self.r_sd = np.maximum(R[tr].std(0), 1e-3)
        self.train_scores = self._top((R - self.r_mu) / self.r_sd)[tr]
        return self

    def score(self, X):
        """Returns (A, s): per-sensor residual z-scores (T x F) and top-n mean score (T,)."""
        Z = self._z(np.asarray(X, float))
        A = (self._abs_residuals(Z) - self.r_mu) / self.r_sd
        return A, self._top(A)

    def threshold(self, q):
        """Threshold = q-th percentile of the TRAINING scores (never of validation/test)."""
        return float(np.percentile(self.train_scores, q))


# ---------- metrics (pointwise; no point-adjustment anywhere) ----------
def confusion(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    return int(((y == 1) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum()), \
           int(((y == 0) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())   # TP FN FP TN


def prf(tp, fn, fp, tn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def projected_f1(tpr, fpr, prevalence=0.195):
    """F1 expected at a different attack prevalence, from TPR and FPR."""
    denom = tpr * prevalence + fpr * (1 - prevalence)
    P = tpr * prevalence / denom if denom > 0 else 0.0
    return 2 * P * tpr / (P + tpr) if P + tpr else 0.0
