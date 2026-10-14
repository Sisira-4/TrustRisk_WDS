"""Tests Phase C4: Explanation Reliability Score (ERS) with SHAP and stability dynamics."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trust.explanation_reliability import (
    ExplanationReliabilityEstimator,
    compute_zscore_rank_stability
)


def test_ers_discriminates_stable_physical_anomaly_vs_noise():
    """Verify C4: ERS is significantly higher on coherent physical anomalies than random noise."""
    estimator = ExplanationReliabilityEstimator(n_draws=10, noise_std=0.05, top_k=3, seed=123)
    T, N = 50, 10
    
    # 1. Coherent, dominant physical anomaly on top sensors
    z_coherent = np.zeros((T, N))
    z_coherent[:, 0] = 20.0
    z_coherent[:, 1] = 18.0
    z_coherent[:, 2] = 15.0
    ers_coherent = estimator.compute_ers(z_coherent, z_scores=z_coherent)
    
    # 2. Ambiguous, unstable noise across all sensors
    rng = np.random.default_rng(999)
    z_noise = rng.normal(1.0, 0.001, size=(T, N))
    ers_noise = estimator.compute_ers(z_noise, z_scores=z_noise)

    assert np.all((ers_coherent >= 0.0) & (ers_coherent <= 1.0))
    assert np.all((ers_noise >= 0.0) & (ers_noise <= 1.0))
    assert np.mean(ers_coherent) > 0.85, f"Coherent signal must achieve > 0.85 ERS, got {np.mean(ers_coherent)}"
    assert np.mean(ers_coherent) > np.mean(ers_noise) + 0.30, (
        f"ERS on coherent signal ({np.mean(ers_coherent):.4f}) must exceed noisy signal ({np.mean(ers_noise):.4f}) by > 0.30"
    )


def test_ers_edge_cases_zero_variance_and_negatives():
    """Verify C4: Robust handling of zero-variance attributions and negative values."""
    estimator = ExplanationReliabilityEstimator(n_draws=5, noise_std=0.05, top_k=3)
    
    # All zeros (zero variance explanation)
    z_zero = np.zeros((10, 8))
    ers_zero = estimator.compute_ers(z_zero, z_scores=z_zero)
    assert np.all(ers_zero == 0.50), "Zero-variance explanation must return neutral fallback stability 0.50"
    
    # Negative attributions handled via absolute magnitude
    z_neg = np.zeros((10, 8))
    z_neg[:, 0] = -25.0
    z_neg[:, 1] = -20.0
    z_neg[:, 2] = -15.0
    ers_neg = estimator.compute_ers(z_neg, z_scores=z_neg)
    assert np.mean(ers_neg) > 0.90, "Negative attributions must be ranked by absolute magnitude without crashing"


def test_zscore_rank_stability_distinct_from_shap():
    """Verify C4: The heuristic metric is preserved as 'sensor z-score rank stability', distinct from SHAP."""
    z_mat = np.random.randn(20, 10)
    stability = compute_zscore_rank_stability(z_mat, n_draws=5, noise_std=0.05, top_k=3)
    assert len(stability) == 20
    assert np.all((stability >= 0.0) & (stability <= 1.0))
