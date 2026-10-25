"""Tests for Threat-Driven Cyber-Physical Risk Index (TDCRI) Stability (Phase B.6).

Verifies:
1. Canonical TDCRI weights match config/trustrisk.yaml and sum to 1.0.
2. TDCRI scores fall strictly in [0, 1].
3. Ranking sensitivity: Under +/-20% perturbations across all 6 weights,
   the top-10 asset ranking achieves Kendall's tau >= 0.85 (ranking stability guarantee).
"""
import pytest
import pandas as pd
import numpy as np

from config.loader import load_config
from risk.tdcri import TDCRICalculator


@pytest.fixture
def calculator():
    config = load_config()
    return TDCRICalculator(config)


@pytest.fixture
def sample_risk_df():
    """Generates a representative asset risk dataframe."""
    assets = [
        {"Asset_ID": "PU2", "Type": "Pump", "MDCI": 0.88, "ASS": 0.92, "ISS": 1.00, "CRI": 0.95, "ARS": 0.45},
        {"Asset_ID": "v1", "Type": "Valve", "MDCI": 0.72, "ASS": 0.86, "ISS": 0.78, "CRI": 0.40, "ARS": 0.35},
        {"Asset_ID": "V47", "Type": "Valve", "MDCI": 0.65, "ASS": 0.85, "ISS": 0.50, "CRI": 0.30, "ARS": 0.35},
        {"Asset_ID": "T4", "Type": "Tank", "MDCI": 0.70, "ASS": 0.80, "ISS": 0.82, "CRI": 0.15, "ARS": 0.65},
        {"Asset_ID": "V45", "Type": "Valve", "MDCI": 0.62, "ASS": 0.85, "ISS": 0.42, "CRI": 0.25, "ARS": 0.35},
        {"Asset_ID": "T2", "Type": "Tank", "MDCI": 0.68, "ASS": 0.80, "ISS": 0.75, "CRI": 0.15, "ARS": 0.65},
        {"Asset_ID": "T1", "Type": "Tank", "MDCI": 0.64, "ASS": 0.80, "ISS": 0.55, "CRI": 0.15, "ARS": 0.65},
        {"Asset_ID": "PU1", "Type": "Pump", "MDCI": 0.55, "ASS": 0.90, "ISS": 0.05, "CRI": 0.05, "ARS": 0.60},
        {"Asset_ID": "PU3", "Type": "Pump", "MDCI": 0.55, "ASS": 0.90, "ISS": 0.05, "CRI": 0.05, "ARS": 0.60},
        {"Asset_ID": "PU4", "Type": "Pump", "MDCI": 0.55, "ASS": 0.90, "ISS": 0.05, "CRI": 0.05, "ARS": 0.60},
        {"Asset_ID": "J280", "Type": "Junction", "MDCI": 0.30, "ASS": 0.50, "ISS": 0.05, "CRI": 0.05, "ARS": 0.70},
        {"Asset_ID": "J269", "Type": "Junction", "MDCI": 0.30, "ASS": 0.50, "ISS": 0.05, "CRI": 0.05, "ARS": 0.70},
    ]
    return pd.DataFrame(assets)


def test_tdcri_weights_configuration(calculator):
    """Verifies canonical TDCRI weights sum to 1.0 and match config."""
    total_w = (
        calculator.w_mdci + calculator.w_ass + calculator.w_iss +
        calculator.w_cri + calculator.w_ars + calculator.w_ts
    )
    assert np.isclose(total_w, 1.0), f"TDCRI weights must sum to 1.0, got {total_w}"
    assert calculator.w_mdci == 0.25
    assert calculator.w_ass == 0.20
    assert calculator.w_iss == 0.20
    assert calculator.w_cri == 0.15
    assert calculator.w_ars == 0.10
    assert calculator.w_ts == 0.10


def test_tdcri_bounds(calculator, sample_risk_df):
    """Verifies all TDCRI scores fall strictly in [0, 1]."""
    scores = calculator.compute_tdcri(sample_risk_df)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_tdcri_ranking_sensitivity_stability(calculator, sample_risk_df):
    """Verifies ranking stability under +/-20% perturbations (Kendall tau >= 0.85)."""
    mean_tau, min_tau, _ = calculator.test_ranking_sensitivity_kendall(
        sample_risk_df, top_k=10, n_trials=100, perturbation_range=0.20
    )
    assert mean_tau >= 0.85, f"Mean Kendall tau ({mean_tau:.4f}) must be >= 0.85"
    assert min_tau >= 0.70, f"Min Kendall tau ({min_tau:.4f}) must be >= 0.70 under +/-20% shifts"
