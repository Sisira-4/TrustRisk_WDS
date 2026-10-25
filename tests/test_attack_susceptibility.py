"""Tests for Attack Susceptibility Score (ASS) (Phase B.4).

Verifies:
1. 5 distinct factors are evaluated (Accessibility, Control Capability, Protocol Vulnerability, Detection Difficulty, Physical Isolation).
2. Weights are loaded from config/trustrisk.yaml and sum to 1.0.
3. Actuators have strictly higher control capability than passive junctions.
4. ASS scores fall strictly in [0, 1].
"""
import pytest
import pandas as pd
import numpy as np

from config.loader import load_config
from risk.attack_susceptibility import AttackSusceptibilityEvaluator


@pytest.fixture
def evaluator():
    config = load_config()
    return AttackSusceptibilityEvaluator(config)


def test_ass_weights_configuration(evaluator):
    """Verifies ASS weights sum to 1.0 and match authoritative config."""
    total_gamma = (
        evaluator.gamma_1 + evaluator.gamma_2 + evaluator.gamma_3 +
        evaluator.gamma_4 + evaluator.gamma_5
    )
    assert np.isclose(total_gamma, 1.0), f"ASS weights must sum to 1.0, got {total_gamma}"
    assert evaluator.gamma_1 == 0.25
    assert evaluator.gamma_2 == 0.20
    assert evaluator.gamma_3 == 0.20
    assert evaluator.gamma_4 == 0.20
    assert evaluator.gamma_5 == 0.15


def test_control_capability_hierarchy(evaluator):
    """Verifies Actuators (Pumps, Valves) have higher control capability than passive Junctions."""
    p_ctrl = evaluator.compute_control_capability("PU2", "Pump")
    v_ctrl = evaluator.compute_control_capability("v1", "Valve")
    j_ctrl = evaluator.compute_control_capability("J280", "Junction")

    assert p_ctrl > j_ctrl, f"Pump control capability ({p_ctrl}) must exceed junction ({j_ctrl})"
    assert v_ctrl > j_ctrl, f"Valve control capability ({v_ctrl}) must exceed junction ({j_ctrl})"


def test_ass_evaluation_bounds(evaluator):
    """Verifies all factor scores and composite ASS fall strictly in [0, 1]."""
    test_assets = [
        {"Asset_ID": "PU2", "Type": "Pump"},
        {"Asset_ID": "v1", "Type": "Valve"},
        {"Asset_ID": "T1", "Type": "Tank"},
        {"Asset_ID": "J280", "Type": "Junction"},
    ]
    df = pd.DataFrame(test_assets)
    res = evaluator.evaluate_dataframe(df)

    factor_cols = [
        "Accessibility", "Control_Capability", "Protocol_Vulnerability",
        "Detection_Difficulty", "Physical_Isolation", "ASS"
    ]
    for col in factor_cols:
        assert (res[col] >= 0.0).all() and (res[col] <= 1.0).all(), f"{col} has values outside [0, 1]"

    # Pumps and valves must have higher susceptibility than passive junctions
    pump_ass = res.loc[res["Type"] == "Pump", "ASS"].iloc[0]
    junc_ass = res.loc[res["Type"] == "Junction", "ASS"].iloc[0]
    assert pump_ass > junc_ass, f"Pump ASS ({pump_ass:.4f}) must exceed Junction ASS ({junc_ass:.4f})"
