"""Tests Phase C3: Prediction Confidence Score (PCS) calibrator validation and safeguards."""
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trust.unified_trust_engine import PredictionConfidenceCalibrator, compute_ece


def test_calibrator_refuses_test_data():
    """Verify C3: Calibrator strictly refuses test data to prevent evaluation leakage."""
    cal = PredictionConfidenceCalibrator()
    dummy_scores = np.linspace(1, 10, 2089)
    dummy_labels = np.random.randint(0, 2, size=2089)

    # 1. Refuses by dataset identity string
    with pytest.raises(ValueError, match="Calibration fit refuses test data"):
        cal.fit(dummy_scores[:50], dummy_labels[:50], dataset_identity="BATADAL_test_dataset")

    # 2. Refuses by test set length
    with pytest.raises(ValueError, match="Detected test set size"):
        cal.fit(dummy_scores, dummy_labels, dataset_identity="unknown_dataset")


def test_calibrator_rejects_non_positive_temperature():
    """Verify C3: Temperature parameter T <= 0 is strictly rejected."""
    cal = PredictionConfidenceCalibrator()
    with pytest.raises(ValueError, match="strictly positive"):
        cal.set_temperature(0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        cal.set_temperature(-1.5)


def test_calibrator_parameter_persistence_roundtrip():
    """Verify C3: Calibrator parameters persist to JSON and restore exact identical probabilities."""
    scores_val = np.array([1.2, 2.5, 3.8, 6.5, 7.8, 9.2])
    labels_val = np.array([0, 0, 0, 1, 1, 1])

    cal = PredictionConfidenceCalibrator()
    cal.fit(scores_val, labels_val, dataset_identity="validation_synthetic")

    eval_scores = np.array([2.0, 5.0, 8.0])
    probs_orig = cal.predict_prob(eval_scores)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cal.save_parameters(tmp_path)
        
        # Load into fresh instance
        cal_restored = PredictionConfidenceCalibrator()
        cal_restored.load_parameters(tmp_path)
        
        assert cal_restored.A == pytest.approx(cal.A, rel=1e-6)
        assert cal_restored.B == pytest.approx(cal.B, rel=1e-6)
        assert cal_restored.T == pytest.approx(cal.T, rel=1e-6)
        
        probs_restored = cal_restored.predict_prob(eval_scores)
        np.testing.assert_allclose(probs_orig, probs_restored, rtol=1e-7)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_ece_improvement_on_miscalibrated_signal():
    """Verify C3: Calibration significantly reduces ECE on miscalibrated overconfident scores."""
    # Create uncalibrated scores that are overconfident
    scores = np.array([1.0, 1.5, 2.0, 7.0, 8.0, 9.0] * 10)
    labels = np.array([0, 0, 0, 1, 1, 1] * 10)
    
    # Raw sigmoid
    raw_probs = 1.0 / (1.0 + np.exp(-1.0 * (scores - 4.5)))
    ece_before = compute_ece(labels, raw_probs)
    
    cal = PredictionConfidenceCalibrator()
    cal.fit(scores, labels, dataset_identity="validation_batch")
    cal_probs = cal.predict_prob(scores)
    ece_after = compute_ece(labels, cal_probs)

    assert ece_after <= ece_before, f"Calibrated ECE ({ece_after:.4f}) must be <= Raw ECE ({ece_before:.4f})"
