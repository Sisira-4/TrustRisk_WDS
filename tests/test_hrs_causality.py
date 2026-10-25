"""Tests Phase C5: Historical Reliability Score (HRS) causality and leakage prevention."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trust.unified_trust_engine import HistoricalReliabilityEstimator


def test_hrs_cold_start_prior():
    """Verify C5: HRS outputs cold-start prior when no prior alarms exist."""
    estimator = HistoricalReliabilityEstimator(rolling_window=168, cold_start_prior=1.0, min_samples=1)
    
    alarms = np.zeros(100, dtype=int)
    labels = np.zeros(100, dtype=int)
    
    hrs = estimator.compute_hrs(alarms, labels)
    assert np.all(hrs == 1.0), "With 0 past alarms, HRS must equal cold-start prior 1.0"


def test_hrs_rolling_precision_computation():
    """Verify C5: HRS accurately reflects empirical precision of preceding alerts within window."""
    estimator = HistoricalReliabilityEstimator(rolling_window=50, cold_start_prior=1.0, min_samples=1)
    
    T = 100
    alarms = np.zeros(T, dtype=int)
    labels = np.zeros(T, dtype=int)
    
    # 4 alarms in preceding hours [10, 15, 20, 25] with 3 true positives
    alarms[[10, 15, 20, 25]] = 1
    labels[[10, 15, 20]] = 1
    labels[25] = 0 # 1 false alarm
    
    hrs = estimator.compute_hrs(alarms, labels)
    
    # At time 30, past alarms are 4 with 3 TPs -> precision = 3/4 = 0.75
    assert hrs[30] == pytest.approx(0.75, abs=1e-5)


def test_hrs_zero_label_leakage_at_current_time():
    """Verify C5: Changing/permuting the label at time t has ZERO impact on HRS at time t."""
    estimator = HistoricalReliabilityEstimator(rolling_window=168, cold_start_prior=1.0)
    
    rng = np.random.default_rng(42)
    T = 200
    alarms = (rng.uniform(size=T) > 0.8).astype(int)
    labels = (rng.uniform(size=T) > 0.7).astype(int)
    
    hrs_original = estimator.compute_hrs(alarms, labels)
    
    # Test across multiple time steps t: invert the label at t
    for test_t in [10, 50, 100, 150, 199]:
        labels_perturbed = labels.copy()
        labels_perturbed[test_t] = 1 - labels[test_t] # flip label at t
        
        hrs_perturbed = estimator.compute_hrs(alarms, labels_perturbed)
        
        # HRS at test_t must be bitwise identical (no lookahead / no self-conditioning)
        assert hrs_original[test_t] == hrs_perturbed[test_t], (
            f"Label leakage detected! Flipping label at t={test_t} changed HRS at t={test_t}."
        )
