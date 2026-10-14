"""Tests for Asset Resilience Score (ARS) (Phase B.5).

Verifies:
1. 4 distinct resilience factors are evaluated (Redundancy, Recovery Time, Alternative Paths, Operational Flexibility).
2. Weights are loaded from config/trustrisk.yaml and sum to 1.0.
3. Tanks provide significantly higher operational flexibility than pumps/valves.
4. ARS scores fall strictly in [0, 1].
"""
import pytest
import pandas as pd
import numpy as np
import wntr

from config.loader import load_config
from risk.asset_resilience import AssetResilienceEvaluator
from risk.hydraulic_simulation import INP_PATH


@pytest.fixture
def evaluator():
    config = load_config()
    wn = wntr.network.WaterNetworkModel(str(INP_PATH))
    return AssetResilienceEvaluator(wn, config)


def test_ars_weights_configuration(evaluator):
    """Verifies ARS weights sum to 1.0 and match authoritative config."""
    total_v = evaluator.v_1 + evaluator.v_2 + evaluator.v_3 + evaluator.v_4
    assert np.isclose(total_v, 1.0), f"ARS weights must sum to 1.0, got {total_v}"
    assert evaluator.v_1 == 0.30
    assert evaluator.v_2 == 0.25
    assert evaluator.v_3 == 0.25
    assert evaluator.v_4 == 0.20


def test_tank_operational_flexibility(evaluator):
    """Verifies Tanks provide highest hydraulic buffer storage flexibility."""
    t_flex = evaluator.compute_operational_flexibility("T1", "Tank")
    p_flex = evaluator.compute_operational_flexibility("PU2", "Pump")
    v_flex = evaluator.compute_operational_flexibility("v1", "Valve")

    assert t_flex > p_flex, f"Tank flexibility ({t_flex}) must exceed pump ({p_flex})"
    assert t_flex > v_flex, f"Tank flexibility ({t_flex}) must exceed valve ({v_flex})"


def test_ars_evaluation_bounds(evaluator):
    """Verifies all factor scores and composite ARS fall strictly in [0, 1]."""
    test_assets = [
        {"Asset_ID": "PU2", "Type": "Pump"},
        {"Asset_ID": "v1", "Type": "Valve"},
        {"Asset_ID": "T1", "Type": "Tank"},
        {"Asset_ID": "J280", "Type": "Junction"},
    ]
    df = pd.DataFrame(test_assets)
    res = evaluator.evaluate_dataframe(df)

    factor_cols = [
        "Redundancy", "Recovery_Time_Score", "Alternative_Paths",
        "Operational_Flexibility", "ARS"
    ]
    for col in factor_cols:
        assert (res[col] >= 0.0).all() and (res[col] <= 1.0).all(), f"{col} has values outside [0, 1]"
