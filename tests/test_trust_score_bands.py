"""Tests Phase C6: Unified Trust Score (TS) canonical formula and ATI triage band boundaries."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.trust.unified_trust_engine import UnifiedTrustEngine
from config.loader import load_config


def test_trust_score_canonical_formula():
    """Verify C6: TS = w1*PCS + w2*(1-UQS) + w3*ERS + w4*HRS with weights summing to 1.0."""
    cfg = load_config()
    engine = UnifiedTrustEngine(config=cfg)

    # Hand-calculated instance with 4 terms
    # w1=0.40, w2=0.30, w3=0.15, w4=0.15
    pcs = np.array([0.90])
    uqs = np.array([0.10])
    ers = np.array([0.80])
    hrs = np.array([0.70])

    ts = engine.compute_trust_score(pcs, uqs, ers, hrs=hrs)
    # Expected: 0.40*0.90 + 0.30*(1 - 0.10) + 0.15*0.80 + 0.15*0.70
    # = 0.36 + 0.27 + 0.12 + 0.105 = 0.855
    assert ts[0] == pytest.approx(0.855, abs=1e-5)


def test_trust_score_3term_fallback():
    """Verify C6: When HRS is omitted, 3-term weights sum to 1.0 (w_pcs=0.30, w_uqs=0.40, w_ers=0.30)."""
    cfg = load_config()
    engine = UnifiedTrustEngine(config=cfg)

    pcs = np.array([0.80])
    uqs = np.array([0.20])
    ers = np.array([0.90])

    ts = engine.compute_trust_score(pcs, uqs, ers)
    expected = engine.w_pcs * 0.80 + engine.w_uqs * (1.0 - 0.20) + engine.w_ers * 0.90
    assert ts[0] == pytest.approx(expected, abs=1e-5)


def test_ati_band_exact_boundaries():
    """Verify C6: ATI bands strictly obey boundary rules at TS = 0.50 and TS = action_threshold (0.85)."""
    cfg = load_config()
    engine = UnifiedTrustEngine(config=cfg)
    act = engine.action_threshold

    boundary_scores = np.array([
        0.00,
        0.499999,
        0.500000,
        act - 0.000001,
        act,
        1.000000
    ])

    bands = engine.assign_ati_bands(boundary_scores)
    assert bands[0] == "MONITOR", "TS = 0.00 must be MONITOR"
    assert bands[1] == "MONITOR", "TS = 0.499999 must be MONITOR"
    assert bands[2] == "REVIEW", "TS = 0.500000 must be REVIEW (exact boundary)"
    assert bands[3] == "REVIEW", f"TS = {act - 0.000001} must be REVIEW"
    assert bands[4] == "ACTION", f"TS = {act} must be ACTION (exact boundary)"
    assert bands[5] == "ACTION", "TS = 1.000000 must be ACTION"


def test_trust_score_strictly_bounded_and_monotonic():
    """Verify C6: TS is strictly bounded in [0, 1] and strictly monotonic with each component."""
    engine = UnifiedTrustEngine()
    
    # Boundary extremes
    ts_min = engine.compute_trust_score([0.0], [1.0], [0.0], hrs=[0.0])
    ts_max = engine.compute_trust_score([1.0], [0.0], [1.0], hrs=[1.0])
    assert ts_min[0] == pytest.approx(0.0, abs=1e-6)
    assert ts_max[0] == pytest.approx(1.0, abs=1e-6)
    
    # Monotonicity with PCS
    ts_low_pcs = engine.compute_trust_score([0.2], [0.3], [0.5])
    ts_high_pcs = engine.compute_trust_score([0.8], [0.3], [0.5])
    assert ts_high_pcs[0] > ts_low_pcs[0]
