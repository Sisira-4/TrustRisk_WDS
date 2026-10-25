import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from risk.hydraulic_simulation import HydraulicRiskEngine, COST_BENEFIT_TABLE


def test_wntr_ctown_loading():
    engine = HydraulicRiskEngine()
    assert engine.wn_base.num_nodes == 396, "CTown must have 396 nodes"
    assert engine.wn_base.num_links == 444, "CTown must have 444 links"
    assert engine.wn_base.num_pumps == 11, "CTown must have 11 pumps"


def test_risk_assets_report_consistency():
    report_csv = ROOT / "results" / "risk_assets_report.csv"
    assert report_csv.exists(), "results/risk_assets_report.csv must exist"
    df = pd.read_csv(report_csv)
    for col in ["Asset_ID", "Type", "ISS", "CRI", "MDCI", "ASS", "Base_Risk", "Unit_Cost_k", "Protection_Benefit"]:
        assert col in df.columns, f"Missing {col} in risk assets report"
        assert not df[col].isnull().any(), f"Null values in {col}"
    # Values bounded in [0, 1]
    for metric in ["ISS", "CRI", "MDCI", "ASS", "Base_Risk"]:
        assert (df[metric] >= 0.0).all() and (df[metric] <= 1.0).all()


def test_tppi_knapsack_budget_constraint():
    report_csv = ROOT / "results" / "risk_assets_report.csv"
    df = pd.read_csv(report_csv)
    engine = HydraulicRiskEngine()
    budget = 400
    selected, total_spent, total_val = engine.solve_tppi_knapsack(df, budget_k=budget, trust_score=0.85)
    assert total_spent <= budget, f"Knapsack spent {total_spent} which exceeds budget {budget}"
    assert len(selected) > 0, "Knapsack must select at least one asset"


def test_non_circular_risk_grounding():
    """Verifies that risk indices are strictly non-circular and grounded in physical consequences."""
    from scipy.stats import spearmanr
    report_csv = ROOT / "results" / "risk_assets_report.csv"
    df = pd.read_csv(report_csv).dropna(subset=["ISS", "Structural_Criticality", "MDCI", "Base_Risk", "Affected_Nodes"])
    
    # 1. Structural centrality alone: honest negative finding
    rho_struct, p_struct = spearmanr(df["Structural_Criticality"], df["ISS"])
    assert pytest.approx(rho_struct, abs=1e-4) == 0.0932
    assert p_struct > 0.05, "Pure structural centrality fails to predict dynamic hydraulic consequences"

    # 2. Non-circular MDCI (pure intrinsic properties, zero ISS input)
    rho_mdci, p_mdci = spearmanr(df["MDCI"], df["ISS"])
    assert pytest.approx(rho_mdci, abs=1e-4) == 0.1300

    # 3. Non-circular validation against independent spatial customer footprint
    rho_spatial, p_spatial = spearmanr(df["Base_Risk"], df["Affected_Nodes"])
    assert pytest.approx(rho_spatial, abs=1e-4) == 0.5805
    assert p_spatial < 0.001, "Base risk must statistically significantly predict customer impact footprint"

    # 4. Composite pressure deficit alignment
    rho_iss, p_iss = spearmanr(df["Base_Risk"], df["ISS"])
    assert pytest.approx(rho_iss, abs=1e-4) == 0.6109
    assert p_iss < 0.001


def test_1000_trial_weight_sensitivity_distribution():
    """Verifies empirical sensitivity distribution over 1,000 weight perturbations."""
    from config.loader import load_config
    from risk.tdcri import TDCRICalculator
    report_csv = ROOT / "results" / "risk_assets_report.csv"
    df = pd.read_csv(report_csv)
    cfg = load_config()
    calc = TDCRICalculator(cfg)
    
    mean_tau, min_tau, stats = calc.test_ranking_sensitivity_kendall(df, n_trials=1000, perturbation_range=0.20)
    assert mean_tau > 0.95, f"Mean Kendall tau should exceed 0.95, got {mean_tau:.4f}"
    assert min_tau > 0.90, f"Min Kendall tau should exceed 0.90, got {min_tau:.4f}"
    assert stats["p5_tau"] > 0.92, f"5th percentile tau should exceed 0.92, got {stats['p5_tau']:.4f}"
    
    # Top-5 asset retention
    for asset, freq in stats["top_k_frequencies"].items():
        assert freq >= 0.95, f"Top-5 asset {asset} retention frequency {freq:.2f} should exceed 95%"


def test_tppi_tie_breaking_and_multi_budget():
    """Verifies deterministic tie-breaking and multi-budget knapsack solving."""
    from config.loader import load_config
    from risk.tppi import TPPISolver
    report_csv = ROOT / "results" / "risk_assets_report.csv"
    df = pd.read_csv(report_csv)
    cfg = load_config()
    solver = TPPISolver(cfg)
    
    # 1. Tied groups detection
    tied = solver.find_tied_groups(df, tol=1e-4)
    assert len(tied) > 0, "Should detect tied groups in asset priority densities"
    junction_group = next((g for g in tied if len(g["asset_ids"]) == 6), None)
    assert junction_group is not None, "Should identify cluster of 6 identical-density junctions"
    assert junction_group["type"] == "Junction"
    
    # 2. Multi-budget portfolio
    multi = solver.solve_multi_budget(df, budgets=[200, 500, 1000])
    assert set(multi.keys()) == {200, 500, 1000}
    assert multi[200]["spent_k"] <= 200
    assert multi[500]["spent_k"] <= 500
    assert multi[1000]["spent_k"] <= 1000
    
    # Asset count and value monotonicity
    assert multi[200]["asset_count"] < multi[500]["asset_count"] < multi[1000]["asset_count"]
    assert multi[200]["total_value"] < multi[500]["total_value"] < multi[1000]["total_value"]
    assert multi[200]["asset_count"] == 4
    assert multi[500]["asset_count"] == 9
    assert multi[1000]["asset_count"] == 17


