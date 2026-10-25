"""Tests Phase C1 and C2: Temporal continuity of UQS block-bootstrap pairs and normalization isolation."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trust.unified_trust_engine import EpistemicUncertaintyQuantifier


def test_ensemble_pairs_strictly_consecutive():
    """Verify C1: every training pair (x_t, x_{t+1}) used by each ensemble member is truly consecutive."""
    rng = np.random.default_rng(42)
    T, N = 1000, 6
    synthetic_series = np.cumsum(rng.normal(0, 1, size=(T, N)), axis=0)
    
    block_size = 48
    n_estimators = 5
    quant = EpistemicUncertaintyQuantifier(n_estimators=n_estimators, block_size=block_size)
    quant.fit(synthetic_series)

    assert len(quant.models) == n_estimators
    assert len(quant.member_pair_indices) == n_estimators

    for m_idx, pairs in enumerate(quant.member_pair_indices):
        assert len(pairs) > 0, f"Member {m_idx} must have trained on pairs"
        # Each row is [idx_t, idx_{t+1}]
        t_indices = pairs[:, 0]
        next_t_indices = pairs[:, 1]
        
        # Verify delta is strictly +1 hour (consecutive adjacent hours)
        deltas = next_t_indices - t_indices
        assert np.all(deltas == 1), f"Member {m_idx} contains non-consecutive pair transitions!"
        
        # Verify that within each block, indices are sorted and strictly continuous
        blocks = quant.member_blocks[m_idx]
        for blk in blocks:
            assert np.all(np.diff(blk) == 1), f"Block in member {m_idx} is not consecutive!"


def test_uqs_scales_frozen_on_training_data():
    """Verify C2: normalization scales are fitted strictly on training data and frozen."""
    rng = np.random.default_rng(123)
    X_train = rng.normal(10, 2, size=(500, 4))
    quant = EpistemicUncertaintyQuantifier(n_estimators=3, block_size=24)
    quant.fit(X_train)

    assert quant.fitted is True
    assert quant.epistemic_scale_ > 0.0
    assert quant.aleatoric_scale_ > 0.0
    
    saved_ep_scale = quant.epistemic_scale_
    saved_al_scale = quant.aleatoric_scale_

    # Predict on test data with huge variance
    X_test_wild = rng.normal(100, 50, size=(200, 4))
    uqs_wild = quant.compute_uqs(X_test_wild)

    # Assert internal scales remained bitwise identical (no adaptation/leakage on test data)
    assert quant.epistemic_scale_ == saved_ep_scale
    assert quant.aleatoric_scale_ == saved_al_scale
    assert np.all((uqs_wild >= 0.0) & (uqs_wild <= 1.0))


def test_uqs_scoring_first_half_invariant_to_future_data():
    """Verify Task 3.3: scoring the first half of a test set alone gives the exact same values
    for those hours as scoring the whole set (no future test-series statistics or lookahead)."""
    rng = np.random.default_rng(42)
    T_train, N = 500, 4
    X_train = rng.normal(10, 2, size=(T_train, N))

    quant = EpistemicUncertaintyQuantifier(n_estimators=5, block_size=24)
    quant.fit(X_train)

    T_test = 300
    X_test = rng.normal(12, 3, size=(T_test, N))

    # Score full test set
    uqs_full = quant.compute_uqs(X_test)
    epi_full = quant.compute_epistemic(X_test)
    ale_full = quant.compute_aleatoric(X_test)

    # Score first half alone
    half = T_test // 2
    X_half = X_test[:half]
    uqs_half = quant.compute_uqs(X_half)
    epi_half = quant.compute_epistemic(X_half)
    ale_half = quant.compute_aleatoric(X_half)

    # Verify bitwise invariance on first half hours
    np.testing.assert_allclose(epi_half, epi_full[:half], atol=1e-12, err_msg="Epistemic scores must not depend on future hours")
    np.testing.assert_allclose(ale_half, ale_full[:half], atol=1e-12, err_msg="Aleatoric scores must not depend on future hours")
    np.testing.assert_allclose(uqs_half, uqs_full[:half], atol=1e-12, err_msg="UQS dual uncertainty must not depend on future hours")


def test_uqs_batadal_test_set_first_half_invariant():
    """Verify Task 3.3 on real BATADAL test set: scoring first half produces identical values to full test set."""
    test_csv = ROOT / "data" / "processed" / "test_processed.csv"
    train_csv = ROOT / "data" / "processed" / "train_processed.csv"
    if not (test_csv.exists() and train_csv.exists()):
        pytest.skip("Processed BATADAL datasets not found.")

    import pandas as pd
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    sensor_cols = [c for c in train_df.columns if c not in ["DATETIME", "LABEL", "TRAIN_MASK", "EVAL_MASK"]]

    quant = EpistemicUncertaintyQuantifier(n_estimators=5, block_size=168)
    quant.fit(train_df[sensor_cols].values)

    X_test = test_df[sensor_cols].values
    uqs_full = quant.compute_uqs(X_test)

    half = len(X_test) // 2
    uqs_half = quant.compute_uqs(X_test[:half])

    np.testing.assert_allclose(uqs_half, uqs_full[:half], atol=1e-12, err_msg="BATADAL test UQS must not depend on future test hours")


def test_uqs_unfitted_scoring_raises_error():
    """Verify that calling compute functions before fit() raises RuntimeError to prevent un-normalized leakage."""
    quant = EpistemicUncertaintyQuantifier()
    X = np.ones((50, 4))
    with pytest.raises(RuntimeError, match="must be fitted"):
        quant.compute_uqs(X)
    with pytest.raises(RuntimeError, match="must be fitted"):
        quant.compute_epistemic(X)
    with pytest.raises(RuntimeError, match="must be fitted"):
        quant.compute_aleatoric(X)
