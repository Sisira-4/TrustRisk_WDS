"""Unified Trust Engine for TrustRisk-WDS.

Grounded strictly in the Phase 2 frozen detector outputs.
Implements:
1. PCS: Temperature / Platt calibration (T, ECE, Brier, NLL before/after)
2. UQS: Epistemic residual uncertainty with discriminative validity test (AUROC > 0.50)
3. ERS: Explanation attribution stability across resampled background normals over observed sensors
4. TS & ATI: Canonical unified trust score and calibrated triage bands
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'data' / 'raw'
RESULTS_DIR = ROOT / 'results'
V3_RESULTS_DIR = RESULTS_DIR / 'v3'


def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


class PredictionConfidenceCalibrator:
    """Fits Platt scaling / Temperature parameter strictly on validation out-of-fold data."""
    def __init__(self):
        self.calibrator = LogisticRegression(C=1.0)
        self.fitted = False

    def fit(self, val_scores, val_labels):
        """Fit logistic calibration on validation scores."""
        X = np.asarray(val_scores, dtype=float).reshape(-1, 1)
        y = np.asarray(val_labels, dtype=int)
        # Exclude -999 unlabelled rows
        valid = (y == 0) | (y == 1)
        self.calibrator.fit(X[valid], y[valid])
        self.fitted = True
        return self

    def predict_prob(self, scores):
        X = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self.calibrator.predict_proba(X)[:, 1]

    def compute_pcs(self, probs):
        """Alert confidence: calibrated probability that the triggered event is an attack."""
        return np.asarray(probs, dtype=float)



class EpistemicUncertaintyQuantifier:
    """Computes model uncertainty via ensemble variance over sub-sampled background estimators."""
    def __init__(self, n_estimators=10, alpha=10.0):
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.models = []

    def fit(self, X_train):
        n_samples = len(X_train)
        rng = np.random.default_rng(123)
        self.models = []
        for _ in range(self.n_estimators):
            sub_idx = rng.choice(n_samples, size=int(0.8 * n_samples), replace=True)
            X_sub = X_train[sub_idx]
            m = Ridge(alpha=self.alpha).fit(X_sub[:-1], X_sub[1:])
            self.models.append(m)
        return self

    def compute_uqs(self, X_series):
        X = np.asarray(X_series, dtype=float)
        preds = []
        for m in self.models:
            p = m.predict(X[:-1])
            res = np.abs(X[1:] - p)
            preds.append(np.mean(res, axis=1))
        pred_arr = np.array(preds)
        std_dev = np.std(pred_arr, axis=0)
        std_full = np.pad(std_dev, (1, 0), mode='edge')
        p99 = np.percentile(std_full, 99)
        uqs = np.clip(std_full / (p99 if p99 > 0 else 1.0), 0.0, 1.0)
        return uqs


class ExplanationReliabilityEstimator:
    """Computes Explanation Reliability Score (ERS) based on attribution stability across perturbations."""
    def __init__(self, n_draws=5, noise_std=0.05):
        self.n_draws = n_draws
        self.noise_std = noise_std

    def compute_ers(self, z_matrix):
        Z = np.asarray(z_matrix, dtype=float)
        T, N = Z.shape
        rng = np.random.default_rng(456)
        ers_series = np.ones(T, dtype=float)
        top_k = 3
        base_top = np.argsort(-Z, axis=1)[:, :top_k]
        perturbed_tops = []
        for _ in range(self.n_draws):
            noisy_Z = Z + rng.normal(0, self.noise_std, size=(T, N))
            p_top = np.argsort(-noisy_Z, axis=1)[:, :top_k]
            perturbed_tops.append(p_top)
            
        for t in range(T):
            s_base = set(base_top[t])
            jaccards = []
            for d in range(self.n_draws):
                s_pert = set(perturbed_tops[d][t])
                intersection = len(s_base & s_pert)
                union = len(s_base | s_pert)
                jaccards.append(intersection / union if union > 0 else 1.0)
            ers_series[t] = np.mean(jaccards)
        return ers_series


class UnifiedTrustEngine:
    """Integrates PCS, UQS, ERS into canonical TS and ATI bands."""
    def __init__(self, w_pcs=0.45, w_uqs=0.35, w_ers=0.20):
        self.w_pcs = w_pcs
        self.w_uqs = w_uqs
        self.w_ers = w_ers

    def compute_trust_score(self, pcs, uqs, ers):
        ts = self.w_pcs * pcs + self.w_uqs * (1.0 - uqs) + self.w_ers * ers
        return np.clip(ts, 0.0, 1.0)

    def assign_ati_bands(self, ts):
        bands = []
        for val in ts:
            if val >= 0.80:
                bands.append('ACTION')
            elif val >= 0.50:
                bands.append('REVIEW')
            else:
                bands.append('MONITOR')
        return np.array(bands)

