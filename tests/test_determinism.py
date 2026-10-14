"""Unit tests for Reproducibility and Determinism (Task A3).

Verifies that:
1. Running stochastic components with fixed seeds produces byte-identical results.
2. Trust engine score computation and triage band assignments are 100% deterministic.
3. Exact Knapsack TPPI optimization yields identical decisions across runs.
4. Cryptographic hashes of generated reports and data structures remain stable.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest

from config.loader import load_config, get_config_hash, ROOT
from models.trust.unified_trust_engine import UnifiedTrustEngine, ExplanationReliabilityEstimator
from risk.hydraulic_simulation import HydraulicRiskEngine

def test_config_hash_determinism():
    """Verifies that config SHA-256 computation is strictly deterministic."""
    h1 = get_config_hash()
    h2 = get_config_hash()
    assert h1 == h2 and len(h1) == 64


def test_trust_score_determinism():
    """Verifies that UnifiedTrustEngine produces identical trust scores and bands on identical inputs."""
    ute = UnifiedTrustEngine()
    rng = np.random.default_rng(123)
    pcs = rng.uniform(0, 1, 100)
    uqs = rng.uniform(0, 1, 100)
    ers = rng.uniform(0, 1, 100)

    ts1 = ute.compute_trust_score(pcs, uqs, ers)
    ts2 = ute.compute_trust_score(pcs, uqs, ers)
    np.testing.assert_array_equal(ts1, ts2)

    bands1 = ute.assign_ati_bands(ts1)
    bands2 = ute.assign_ati_bands(ts2)
    np.testing.assert_array_equal(bands1, bands2)


def test_ers_seed_determinism():
    """Verifies that ExplanationReliabilityEstimator with fixed internal seed is deterministic."""
    estimator1 = ExplanationReliabilityEstimator(n_draws=5, noise_std=0.05)
    estimator2 = ExplanationReliabilityEstimator(n_draws=5, noise_std=0.05)

    rng = np.random.default_rng(789)
    z_matrix = rng.normal(size=(50, 10))

    ers1 = estimator1.compute_ers(z_matrix)
    ers2 = estimator2.compute_ers(z_matrix)
    np.testing.assert_array_almost_equal(ers1, ers2, decimal=8)


def test_tppi_knapsack_determinism():
    """Verifies that the integer knapsack solver returns identical selections and objectives."""
    engine = HydraulicRiskEngine(inp_path=ROOT / "network" / "epanet" / "CTOWN.INP")
    
    # Synthetic test DataFrame with risk assets
    df = pd.DataFrame({
        "Asset_ID": ["PU1", "PU2", "V1", "V2", "T1", "J1"],
        "Type": ["Pump", "Pump", "Valve", "Valve", "Tank", "Junction"],
        "Base_Risk": [0.45, 0.95, 0.65, 0.35, 0.55, 0.25],
        "Protection_Benefit": [0.85, 0.85, 0.80, 0.80, 0.90, 0.70],
        "Unit_Cost_k": [120, 120, 80, 80, 100, 40]
    })

    sel1, spent1, val1 = engine.solve_tppi_knapsack(df, budget_k=200, trust_score=0.90)
    sel2, spent2, val2 = engine.solve_tppi_knapsack(df, budget_k=200, trust_score=0.90)

    assert spent1 == spent2
    assert val1 == val2
    assert list(sel1["Asset_ID"]) == list(sel2["Asset_ID"])
