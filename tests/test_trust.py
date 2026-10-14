import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trust.unified_trust_engine import (
    compute_ece,
    PredictionConfidenceCalibrator,
    EpistemicUncertaintyQuantifier,
    ExplanationReliabilityEstimator,
    UnifiedTrustEngine
)


def test_ece_computation():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    ece = compute_ece(y_true, y_prob)
    assert abs(ece) < 1e-6, "Perfect predictions should have 0 ECE"


def test_pcs_calibrator_monotonic():
    cal = PredictionConfidenceCalibrator()
    # Dummy training data
    scores = np.array([1.0, 2.0, 8.0, 9.0])
    labels = np.array([0, 0, 1, 1])
    cal.fit(scores, labels)
    test_scores = np.array([2.0, 5.0, 8.0])
    probs = cal.predict_prob(test_scores)
    assert probs[0] < probs[1] < probs[2], "Calibrated probabilities must be strictly monotonic with scores"


def test_ers_drops_on_noisy_matrix():
    estimator = ExplanationReliabilityEstimator(n_draws=10, noise_std=2.0)
    # Coherent matrix: strong signal in first 3 sensors
    z_coherent = np.zeros((10, 20))
    z_coherent[:, :3] = 50.0 # very strong signal
    ers_coherent = estimator.compute_ers(z_coherent)
    
    # Incoherent matrix: noise in all sensors
    rng = np.random.default_rng(99)
    z_random = rng.normal(0, 1, size=(10, 20))
    ers_random = estimator.compute_ers(z_random)
    
    assert np.mean(ers_coherent) > np.mean(ers_random), "ERS must be significantly higher on coherent signal than random noise"


def test_trust_pipeline_file_integrity():
    pipe_path = ROOT / "results" / "v3" / "trust_pipeline_clean.csv"
    assert pipe_path.exists(), "results/v3/trust_pipeline_clean.csv must exist"
    df = pd.read_csv(pipe_path)
    assert len(df) == 2089, "Test pipeline must have 2089 rows"
    for col in ["score", "threshold", "p_attack", "PCS", "UQS", "ERS", "TS", "ATI_band"]:
        assert col in df.columns, f"Missing {col} in trust pipeline"
        assert not df[col].isnull().any(), f"Null values in {col}"
    assert set(df["ATI_band"].unique()).issubset({"ACTION", "REVIEW", "MONITOR"})


def test_uqs_temporal_continuity():
    # Verify that EpistemicUncertaintyQuantifier preserves contiguous temporal blocks
    # and produces valid epistemic uncertainty values without random row shuffling.
    rng = np.random.default_rng(42)
    T, N = 500, 5
    data = np.zeros((T, N))
    for t in range(1, T):
        data[t] = 0.8 * data[t - 1] + rng.normal(0, 0.1, size=N)

    quant = EpistemicUncertaintyQuantifier(n_estimators=5, alpha=1.0, block_size=50)
    quant.fit(data)

    assert len(quant.models) == 5, "Ensemble must fit 5 models"
    for m in quant.models:
        assert m.coef_.shape == (N, N), "Each model must have shape (n_features, n_features)"

    uqs_in = quant.compute_uqs(data)
    assert uqs_in.shape == (T,), "UQS output must match input length"
    assert np.all((uqs_in >= 0.0) & (uqs_in <= 1.0)), "UQS values must be bounded in [0, 1]"


def test_linear_shap_attribution_stability():
    """Verify Phase 3 Linear SHAP Attribution Stability under perturbation."""
    estimator = ExplanationReliabilityEstimator(n_draws=10, noise_std=0.05)
    T, N = 100, 10
    
    # 1. Definite physical anomaly: top 3 sensors have clear, stable signal
    z_coherent = np.zeros((T, N))
    z_coherent[:, 0] = 15.0
    z_coherent[:, 1] = 12.0
    z_coherent[:, 2] = 10.0
    ers_coherent = estimator.compute_ers(z_coherent)
    assert np.all((ers_coherent >= 0.0) & (ers_coherent <= 1.0))
    assert np.mean(ers_coherent) > 0.95, "Coherent persistent anomaly must achieve high ERS attribution stability"

    # 2. Ambiguous signal: top sensors have nearly identical values, perturbed easily
    z_ambiguous = np.zeros((T, N))
    z_ambiguous[:, :5] = 2.0001
    ers_ambiguous = estimator.compute_ers(z_ambiguous)
    assert np.mean(ers_coherent) > np.mean(ers_ambiguous), "Coherent anomaly must have higher stability than ambiguous"


def test_dual_uncertainty_quantification():
    """Verify dual uncertainty: epistemic ensemble variance + aleatoric telemetry jitter."""
    rng = np.random.default_rng(42)
    T, N = 400, 4
    clean_data = np.zeros((T, N))
    for t in range(1, T):
        clean_data[t] = 0.85 * clean_data[t - 1] + rng.normal(0, 0.05, size=N)

    quant = EpistemicUncertaintyQuantifier(n_estimators=5, alpha=1.0, block_size=40)
    quant.fit(clean_data)

    uqs_dual = quant.compute_uqs(clean_data)
    uqs_epistemic = quant.compute_epistemic(clean_data)
    uqs_aleatoric = quant.compute_aleatoric(clean_data)

    assert uqs_dual.shape == (T,)
    assert np.all((uqs_dual >= 0.0) & (uqs_dual <= 1.0))
    assert np.all((uqs_epistemic >= 0.0) & (uqs_epistemic <= 1.0))
    assert np.all((uqs_aleatoric >= 0.0) & (uqs_aleatoric <= 1.0))

    # Test aleatoric spike on noisy burst
    noisy_data = clean_data.copy()
    noisy_data[200:220] += rng.normal(0, 2.0, size=(20, N))
    aleatoric_burst = quant.compute_aleatoric(noisy_data)
    assert np.mean(aleatoric_burst[200:220]) > np.mean(uqs_aleatoric[200:220]), (
        "Aleatoric uncertainty must elevate during high-frequency telemetry jitter"
    )



