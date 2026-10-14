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
