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

