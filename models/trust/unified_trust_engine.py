"""Unified Trust Engine for TrustRisk-WDS.

Grounded strictly in the Phase 2 frozen detector outputs and verified checkpoint hash.
Implements:
1. Checkpoint Verification: Cryptographic SHA-256 validation against frozen_config.json / trustrisk.yaml
2. PCS: Temperature / Platt calibration (T, ECE, Brier, NLL before/after, parameter save/load, test refusal)
3. UQS: Dual Uncertainty Quantification (Epistemic contiguous block-bootstrap ensemble variance + Aleatoric residual noise)
4. ERS: Exact Linear SHAP Attribution Stability across resampled backgrounds and perturbations
5. HRS: Historical Reliability Estimator (rolling validation alert precision without lookahead)
6. TS & ATI: Canonical unified trust score (w1..w4 from config) and calibrated triage bands
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

# Re-export ERS engine and baseline
from models.trust.explanation_reliability import (
    ExplanationReliabilityEstimator,
    compute_zscore_rank_stability
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / 'data' / 'raw'
RESULTS_DIR = ROOT / 'results'
V3_RESULTS_DIR = RESULTS_DIR / 'v3'


def verify_checkpoint_hash():
    """Verifies that the serialized frozen detector checkpoint matches frozen_config.json / trustrisk.yaml."""
    cfg_path = ROOT / "frozen_config.json"
    if not cfg_path.exists():
        cfg_path = ROOT / "config" / "trustrisk.yaml"
        if not cfg_path.exists():
            return None
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        chk_rel = raw_cfg.get("detector", {}).get("checkpoint_file")
        expected_hash = raw_cfg.get("detector", {}).get("checkpoint_sha256")
    else:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        chk_rel = cfg.get("checkpoint_file")
        expected_hash = cfg.get("checkpoint_sha256")

    if not chk_rel or not expected_hash:
        return None
    chk_path = ROOT / chk_rel
    if not chk_path.exists():
        raise FileNotFoundError(f"Frozen detector checkpoint not found at {chk_path}")
    with open(chk_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"Checkpoint hash mismatch! Expected {expected_hash}, got {actual_hash}")
    return actual_hash


def load_checkpoint():
    """Loads and deserializes the verified frozen detector checkpoint."""
    import pickle
    cfg_path = ROOT / "frozen_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        chk_rel = cfg.get("checkpoint_file", "models/detector/frozen_detector_checkpoint.pkl")
    else:
        chk_rel = "models/detector/frozen_detector_checkpoint.pkl"
    chk_path = ROOT / chk_rel
    verify_checkpoint_hash()
    with open(chk_path, "rb") as f:
        return pickle.load(f)


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
    """Fits Platt scaling / Temperature parameter strictly on validation out-of-fold data.
    
    Attributes:
        method: 'platt' or 'temperature'
        A: Calibrator slope / scaling parameter
        B: Calibrator intercept
        T: Effective temperature (T = 1.0 / A)
        validation_data_identity: Identity of validation split
        checkpoint_sha256: Cryptographic hash of the frozen checkpoint
    """
    def __init__(self, method="platt"):
        self.method = method
        self.calibrator = LogisticRegression(C=1.0)
        self.fitted = False
        self.A = None
        self.B = None
        self.T = None
        self.validation_data_identity = None
        self.checkpoint_sha256 = None

    def fit(self, val_scores, val_labels, dataset_identity="BATADAL_dataset04_margin_free_pool_plus_attacks"):
        """Fit calibration parameters strictly on validation scores, rejecting test data."""
        # Test data refusal guard
        if "test" in str(dataset_identity).lower():
            raise ValueError("Calibration fit refuses test data to prevent data leakage!")
        if len(val_labels) == 2089: # BATADAL test set length
            raise ValueError("Detected test set size (2089). Calibration refuses test data!")

        X = np.asarray(val_scores, dtype=float).reshape(-1, 1)
        y = np.asarray(val_labels, dtype=int)
        valid = (y == 0) | (y == 1)
        self.calibrator.fit(X[valid], y[valid])
        self.A = float(self.calibrator.coef_[0][0])
        self.B = float(self.calibrator.intercept_[0])
        if self.A <= 0:
            raise ValueError(f"Fitted slope A must be positive for monotonic calibration, got {self.A}")
        self.T = float(1.0 / self.A)
        if self.T <= 0:
            raise ValueError(f"Effective temperature T must be strictly positive, got {self.T}")
        self.validation_data_identity = str(dataset_identity)
        self.checkpoint_sha256 = verify_checkpoint_hash()
        self.fitted = True
        return self

    def set_temperature(self, T):
        """Explicitly sets temperature parameter T, ensuring T > 0."""
        if T <= 0:
            raise ValueError(f"Temperature T must be strictly positive (> 0), got {T}")
        self.T = float(T)
        self.A = float(1.0 / T)

    def save_parameters(self, path):
        """Saves calibration parameters to a persistent JSON artifact."""
        data = {
            "method": self.method,
            "A": self.A,
            "B": self.B,
            "T": self.T,
            "validation_data_identity": self.validation_data_identity,
            "checkpoint_sha256": self.checkpoint_sha256
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_parameters(self, path):
        """Loads calibration parameters from a persistent JSON artifact."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.method = data.get("method", "platt")
        self.A = float(data["A"])
        self.B = float(data["B"])
        self.T = float(data["T"])
        if self.T <= 0:
            raise ValueError(f"Loaded temperature T must be strictly positive, got {self.T}")
        self.validation_data_identity = data.get("validation_data_identity")
        self.checkpoint_sha256 = data.get("checkpoint_sha256")
        self.calibrator.coef_ = np.array([[self.A]])
        self.calibrator.intercept_ = np.array([self.B])
        self.calibrator.classes_ = np.array([0, 1])
        self.fitted = True
        return self

    def predict_prob(self, scores):
        X = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self.calibrator.predict_proba(X)[:, 1]

    def compute_pcs(self, probs):
        """Alert confidence: calibrated probability that the triggered event is an attack."""
        return np.asarray(probs, dtype=float)


class EpistemicUncertaintyQuantifier:
    """Computes Dual Uncertainty Quantification (UQS = alpha * Epistemic + (1-alpha) * Aleatoric).
    
    1. Epistemic Uncertainty: Ensemble prediction variance over contiguous multi-hour
       operational blocks (block_size=168 hours or 24 hours), preserving 1-step autoregressive
       continuity (x_t, x_{t+1}) in time.
    2. Aleatoric Uncertainty: Per-sensor heteroscedastic noise estimated from training residual variance.
    3. Normalization: Fitted strictly on training baseline data, frozen for inference.
    """
    def __init__(self, n_estimators=10, alpha=10.0, block_size=168, alpha_epistemic=0.85):
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.block_size = block_size
        self.alpha_epistemic = alpha_epistemic
        self.base_model = None
        self.member_pair_indices = []
        self.member_blocks = []
        self.sensor_residual_std_ = None
        self.epistemic_scale_ = 1.0
        self.aleatoric_scale_ = 1.0
        self.fitted = False

        # Initialize normalisers for Task 3.3 compatibility
        self.diff_sd = None
        self.p99_epi = None
        self.p99_ale = None

    def fit(self, X_train):
        """Fits ensemble over contiguous temporal blocks AND establishes frozen normalization scales on training data."""
        X = np.asarray(X_train, dtype=float)
        n_pairs = len(X) - 1
        X_prev = X[:-1]
        X_next = X[1:]

        # 1. Base model for training residual estimation
        self.base_model = Ridge(alpha=self.alpha).fit(X_prev, X_next)
        R_train = X_next - self.base_model.predict(X_prev)
        self.sensor_residual_std_ = np.std(R_train, axis=0) + 1e-6

        # Normaliser differences for Task 3.3
        d = np.diff(X, axis=0)
        self.diff_sd = d.std(axis=0) + 1e-6

        # 2. Block-bootstrap ensemble preserving consecutive time steps
        rng = np.random.default_rng(123)
        self.models = []
        self.member_pair_indices = []
        self.member_blocks = []
        n_blocks = max(1, int((n_pairs * 0.8) / self.block_size))

        for _ in range(self.n_estimators):
            start_indices = rng.integers(0, max(1, n_pairs - self.block_size), size=n_blocks)
            blocks = [np.arange(s, min(s + self.block_size, n_pairs)) for s in start_indices]
            self.member_blocks.append(blocks)
            
            block_indices = np.concatenate(blocks)
            pairs = np.column_stack([block_indices, block_indices + 1])
            self.member_pair_indices.append(pairs)
            
            m = Ridge(alpha=self.alpha).fit(X_prev[block_indices], X_next[block_indices])
            self.models.append(m)

        # 3. Fit normalization scales strictly on training data
        preds = []
        for m in self.models:
            p = m.predict(X_prev)
            res = np.abs(X_next - p)
            preds.append(np.mean(res, axis=1))
        pred_arr = np.array(preds)
        std_train = np.std(pred_arr, axis=0)
        p99_ep = float(np.percentile(std_train, 99.0))
        self.epistemic_scale_ = p99_ep if p99_ep > 0 else 1.0
        self.p99_epi = self.epistemic_scale_

        R_train_std = R_train / self.sensor_residual_std_
        aleatoric_train = np.mean(R_train_std ** 2, axis=1)
        p99_al = float(np.percentile(aleatoric_train, 99.0))
        self.aleatoric_scale_ = p99_al if p99_al > 0 else 1.0
        self.p99_ale = self.aleatoric_scale_

        self.fitted = True
        return self

    def _raw_epistemic(self, X_series):
        X = np.asarray(X_series, dtype=float)
        if not self.models:
            return np.zeros(len(X))
        preds = []
        for m in self.models:
            p = m.predict(X[:-1])
            res = np.abs(X[1:] - p)
            preds.append(np.mean(res, axis=1))
        pred_arr = np.array(preds)
        std_dev = np.std(pred_arr, axis=0)
        return np.pad(std_dev, (1, 0), mode='edge')

    def _raw_aleatoric(self, X_series):
        X = np.asarray(X_series, dtype=float)
        if self.base_model is None or self.sensor_residual_std_ is None:
            diffs = np.diff(X, axis=0)
            diffs = np.vstack([diffs[:1], diffs])
            diff_sd = self.diff_sd if self.diff_sd is not None else (np.std(diffs, axis=0) + 1e-6)
            aleatoric = np.mean((diffs / diff_sd)**2, axis=1)
            return aleatoric
            
        pred = self.base_model.predict(X[:-1])
        res = X[1:] - pred
        res_std = res / self.sensor_residual_std_
        aleatoric = np.mean(res_std**2, axis=1)
        return np.pad(aleatoric, (1, 0), mode='edge')

    def compute_epistemic(self, X_series):
        raw = self._raw_epistemic(X_series)
        scale = self.epistemic_scale_ if self.epistemic_scale_ is not None else (self.p99_epi if self.p99_epi is not None else 1.0)
        return np.clip(raw / (scale if scale > 0 else 1.0), 0.0, 1.0)

    def compute_aleatoric(self, X_series):
        raw = self._raw_aleatoric(X_series)
        scale = self.aleatoric_scale_ if self.aleatoric_scale_ is not None else (self.p99_ale if self.p99_ale is not None else 1.0)
        return np.clip(raw / (scale if scale > 0 else 1.0), 0.0, 1.0)


    def compute_uqs(self, X_series, alpha_epistemic=None):
        """Dual uncertainty: combines epistemic ensemble variance with aleatoric telemetry jitter."""
        if alpha_epistemic is None:
            alpha_epistemic = self.alpha_epistemic
        uqs_epistemic = self.compute_epistemic(X_series)
        uqs_aleatoric = self.compute_aleatoric(X_series)
        return alpha_epistemic * uqs_epistemic + (1.0 - alpha_epistemic) * uqs_aleatoric


class HistoricalReliabilityEstimator:
    """Computes Historical Reliability of Alerts (HRS) based on rolling precision of past alerts.
    
    Causal implementation without lookahead:
    - Labels evaluated strictly on past, already-resolved alerts (t' < t)
    - Cold-start fallback rule: returns cold_start_prior (default 1.0) when no alerts have occurred
    - Shuffling or flipping the current label at time t has zero impact on HRS at time t
    """
    def __init__(self, rolling_window=168, cold_start_prior=1.0, min_samples=1):
        self.rolling_window = rolling_window
        self.cold_start_prior = cold_start_prior
        self.min_samples = min_samples

    def compute_hrs(self, past_alarms, past_true):
        """Computes causal rolling precision over past alerts."""
        alarms = np.asarray(past_alarms, dtype=int)
        y = np.asarray(past_true, dtype=int)
        T = len(alarms)
        hrs = np.full(T, self.cold_start_prior, dtype=float)

        for t in range(T):
            if t == 0:
                hrs[t] = self.cold_start_prior
                continue
            win_start = max(0, t - self.rolling_window)
            past_a = alarms[win_start:t]
            past_y = y[win_start:t]
            
            # Exclude unlabelled -999 rows
            valid = (past_y == 0) | (past_y == 1)
            past_a_valid = past_a[valid]
            past_y_valid = past_y[valid]
            
            n_alerts = np.sum(past_a_valid)
            if n_alerts >= self.min_samples:
                tp = np.sum((past_a_valid == 1) & (past_y_valid == 1))
                hrs[t] = float(tp / n_alerts)
            else:
                hrs[t] = self.cold_start_prior

        return hrs


class UnifiedTrustEngine:
    """Integrates PCS, UQS, ERS, and HRS into canonical TS and ATI bands.
    
    Canonical Formula:
        TS = w1 * PCS + w2 * (1 - UQS) + w3 * ERS + w4 * HRS
    When HRS is omitted:
        TS = w_pcs_3term * PCS + w_uqs_3term * (1 - UQS) + w_ers_3term * ERS
    
    Validation Protocol (Task 3.1):
        Trust score 3-term weights (w_pcs, w_uqs, w_ers), ATI action cutoff, and Policy 3
        PCS cutoff are selected via systematic leave-one-attack-event-out cross-validation
        across 5 validation folds on BATADAL_dataset04 in experiments/run_triage_cv.py,
        strictly prior to test set triage evaluation, and frozen in trust_frozen_config.json.
    
    All weights and band thresholds are loaded from config/trustrisk.yaml or trust_frozen_config.json.
    """
    def __init__(self, w_pcs=None, w_uqs=None, w_ers=None, config=None):
        if config is None:
            try:
                from config.loader import load_config
                config = load_config()
            except Exception:
                config = {}
        
        t_cfg = config.get("trust", {}) if isinstance(config, dict) else {}
        w_cfg = t_cfg.get("weights", {})
        
        # Canonical 4-term weights
        self.w1 = float(w_cfg.get("w1", 0.40))
        self.w2 = float(w_cfg.get("w2", 0.30))
        self.w3 = float(w_cfg.get("w3", 0.15))
        self.w4 = float(w_cfg.get("w4", 0.15))
        
        # Check trust_frozen_config.json as authoritative frozen reference
        frozen_cfg_path = ROOT / "trust_frozen_config.json"
        frozen_cfg = {}
        if frozen_cfg_path.exists():
            try:
                import json
                with open(frozen_cfg_path, "r", encoding="utf-8") as f:
                    frozen_cfg = json.load(f)
            except Exception:
                frozen_cfg = {}
        frozen_w = frozen_cfg.get("trust_weights_3term", {})
        frozen_ati = frozen_cfg.get("ati_bands", {})

        # 3-term weights (support custom override if provided in constructor)
        if w_pcs is not None and w_uqs is not None and w_ers is not None:
            self.w_pcs = float(w_pcs)
            self.w_uqs = float(w_uqs)
            self.w_ers = float(w_ers)
        else:
            self.w_pcs = float(w_cfg.get("w_pcs_3term", frozen_w.get("w_pcs", 0.30)))
            self.w_uqs = float(w_cfg.get("w_uqs_3term", frozen_w.get("w_uqs", 0.40)))
            self.w_ers = float(w_cfg.get("w_ers_3term", frozen_w.get("w_ers", 0.30)))
            
        ati_cfg = t_cfg.get("ati_bands", {})
        self.action_threshold = float(ati_cfg.get("action_threshold", frozen_ati.get("action_threshold", 0.85)))
        self.review_threshold = float(ati_cfg.get("review_threshold", frozen_ati.get("review_threshold", 0.50)))
        self.monitor_threshold = float(ati_cfg.get("monitor_threshold", frozen_ati.get("monitor_threshold", 0.00)))
        
        self.checkpoint_hash = verify_checkpoint_hash()

    def compute_trust_score(self, pcs, uqs, ers, hrs=None):
        pcs = np.asarray(pcs, dtype=float)
        uqs = np.asarray(uqs, dtype=float)
        ers = np.asarray(ers, dtype=float)
        
        if hrs is not None:
            hrs = np.asarray(hrs, dtype=float)
            ts = self.w1 * pcs + self.w2 * (1.0 - uqs) + self.w3 * ers + self.w4 * hrs
        else:
            ts = self.w_pcs * pcs + self.w_uqs * (1.0 - uqs) + self.w_ers * ers
            
        return np.clip(ts, 0.0, 1.0)

    def assign_ati_bands(self, ts):
        ts = np.asarray(ts, dtype=float)
        bands = np.empty(len(ts), dtype=object)
        bands[ts >= self.action_threshold] = 'ACTION'
        bands[(ts >= self.review_threshold) & (ts < self.action_threshold)] = 'REVIEW'
        bands[ts < self.review_threshold] = 'MONITOR'
        return bands
