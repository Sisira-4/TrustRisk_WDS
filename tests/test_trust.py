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

