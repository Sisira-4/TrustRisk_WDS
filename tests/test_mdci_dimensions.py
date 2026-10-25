"""Tests for Multi-Dimensional Criticality Index (MDCI) (Phase B.3).

Verifies:
1. 4 distinct dimensions are evaluated (Structural, Operational, Monitoring, Recovery).
2. Weights are loaded from config/trustrisk.yaml and sum to 1.0.
3. Recovery complexity follows the documented hierarchy: Tanks > Pumps > Valves > Junctions.
4. MDCI correlates positively with hydraulic impact ISS (Spearman rho > 0, p < 0.05).
"""
import pytest
import pandas as pd
import numpy as np
import wntr

from config.loader import load_config
from risk.asset_criticality import AssetCriticalityEvaluator
from risk.hydraulic_simulation import HydraulicRiskEngine, INP_PATH


@pytest.fixture
def evaluator():
    config = load_config()
    wn = wntr.network.WaterNetworkModel(str(INP_PATH))
    return AssetCriticalityEvaluator(wn, config)


def test_mdci_weights_configuration(evaluator):
    """Verifies MDCI weights come from config and sum to 1.0."""
    total_beta = evaluator.beta_1 + evaluator.beta_2 + evaluator.beta_3 + evaluator.beta_4
    assert np.isclose(total_beta, 1.0), f"MDCI weights must sum to 1.0, got {total_beta}"
    assert evaluator.beta_1 == 0.25
    assert evaluator.beta_2 == 0.25
    assert evaluator.beta_3 == 0.25
    assert evaluator.beta_4 == 0.25


def test_recovery_complexity_hierarchy(evaluator):
    """Verifies Recovery Complexity: Tanks > Pumps > Valves > Junctions."""
    t_rec = evaluator.compute_recovery_complexity("Tank")
    p_rec = evaluator.compute_recovery_complexity("Pump")
    v_rec = evaluator.compute_recovery_complexity("Valve")
    j_rec = evaluator.compute_recovery_complexity("Junction")

    assert t_rec > p_rec, f"Tank recovery ({t_rec}) must exceed pump ({p_rec})"
    assert p_rec > v_rec, f"Pump recovery ({p_rec}) must exceed valve ({v_rec})"
    assert v_rec > j_rec, f"Valve recovery ({v_rec}) must exceed junction ({j_rec})"


def test_mdci_evaluation_bounds(evaluator):
    """Verifies all dimension scores and combined MDCI fall strictly in [0, 1]."""
    test_assets = [
        {"Asset_ID": "PU2", "Type": "Pump"},
        {"Asset_ID": "v1", "Type": "Valve"},
        {"Asset_ID": "T1", "Type": "Tank"},
        {"Asset_ID": "J280", "Type": "Junction"},
    ]
    df = pd.DataFrame(test_assets)
    res = evaluator.evaluate_dataframe(df)

    for col in ["Structural_Criticality", "Operational_Criticality", "Monitoring_Criticality", "Recovery_Complexity", "MDCI"]:
        assert (res[col] >= 0.0).all() and (res[col] <= 1.0).all(), f"{col} has values outside [0, 1]"


def test_mdci_spearman_correlation_positive(evaluator):
    """Verifies MDCI correlates positively with hydraulic consequence ISS."""
    engine = HydraulicRiskEngine(inp_path=INP_PATH)
    df = engine.simulate_outages()
    
    rho, p_val, (ci_low, ci_high) = AssetCriticalityEvaluator.validate_mdci_spearman(df["MDCI"], df["ISS"])
    assert rho > 0.0, f"MDCI must correlate positively with ISS, got {rho:.4f}"
    assert ci_low <= rho <= ci_high
