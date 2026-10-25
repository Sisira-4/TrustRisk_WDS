"""Tests for Pyomo TPPI Knapsack Optimization with HiGHS Solver (Phase B.7).

Verifies:
1. Pyomo model builds and solves with the installed HiGHS solver.
2. The chosen portfolio strictly satisfies the municipal budget constraint.
3. The Pyomo global optimum matches or beats the Dynamic Programming solution within tolerance.
4. Operates properly under both B = $50k and B = $500k.
"""
import pytest
import pandas as pd
import numpy as np

from config.loader import load_config
from risk.tppi import TPPISolver


@pytest.fixture
def solver():
    config = load_config()
    return TPPISolver(config)


@pytest.fixture
def asset_pool():
    """Generates a representative pool of water network assets."""
    assets = [
        {"Asset_ID": "PU2", "Type": "Pump", "TDCRI": 0.95, "Unit_Cost_k": 120, "Protection_Benefit": 0.85},
        {"Asset_ID": "v1", "Type": "Valve", "TDCRI": 0.75, "Unit_Cost_k": 80, "Protection_Benefit": 0.80},
        {"Asset_ID": "V47", "Type": "Valve", "TDCRI": 0.65, "Unit_Cost_k": 80, "Protection_Benefit": 0.80},
        {"Asset_ID": "T4", "Type": "Tank", "TDCRI": 0.70, "Unit_Cost_k": 100, "Protection_Benefit": 0.90},
        {"Asset_ID": "V45", "Type": "Valve", "TDCRI": 0.60, "Unit_Cost_k": 80, "Protection_Benefit": 0.80},
        {"Asset_ID": "T2", "Type": "Tank", "TDCRI": 0.68, "Unit_Cost_k": 100, "Protection_Benefit": 0.90},
        {"Asset_ID": "T1", "Type": "Tank", "TDCRI": 0.62, "Unit_Cost_k": 100, "Protection_Benefit": 0.90},
        {"Asset_ID": "PU1", "Type": "Pump", "TDCRI": 0.45, "Unit_Cost_k": 120, "Protection_Benefit": 0.85},
        {"Asset_ID": "J280", "Type": "Junction", "TDCRI": 0.35, "Unit_Cost_k": 40, "Protection_Benefit": 0.70},
        {"Asset_ID": "J269", "Type": "Junction", "TDCRI": 0.35, "Unit_Cost_k": 40, "Protection_Benefit": 0.70},
        {"Asset_ID": "J300", "Type": "Junction", "TDCRI": 0.35, "Unit_Cost_k": 40, "Protection_Benefit": 0.70},
        {"Asset_ID": "J256", "Type": "Junction", "TDCRI": 0.35, "Unit_Cost_k": 40, "Protection_Benefit": 0.70},
        {"Asset_ID": "J289", "Type": "Junction", "TDCRI": 0.35, "Unit_Cost_k": 40, "Protection_Benefit": 0.70},
        {"Asset_ID": "J415", "Type": "Junction", "TDCRI": 0.35, "Unit_Cost_k": 40, "Protection_Benefit": 0.70},
    ]
    return pd.DataFrame(assets)


def test_pyomo_highs_solve_and_budget_constraint(solver, asset_pool):
    """Verifies Pyomo solves with HiGHS and respects budget constraints for B=$500k."""
    budget_k = 500.0
    selected_df, total_cost, total_val, status = solver.solve_pyomo(
        asset_pool, budget_k=budget_k, trust_score=0.90
    )

    assert status == "ok", f"Solver status must be ok, got {status}"
    assert total_cost <= budget_k, f"Total cost ({total_cost}) exceeds budget ({budget_k})"
    assert len(selected_df) > 0, "Selected assets should not be empty"


def test_pyomo_small_budget_feasibility(solver, asset_pool):
    """Verifies feasibility under restricted budget B = $50k."""
    budget_k = 50.0
    selected_df, total_cost, total_val, status = solver.solve_pyomo(
        asset_pool, budget_k=budget_k, trust_score=0.90
    )

    assert status == "ok"
    assert total_cost <= budget_k, f"Total cost ({total_cost}) exceeds budget ({budget_k})"
    # With B = $50k, only one $40k junction can be selected
    assert total_cost in (0.0, 40.0)


def test_pyomo_vs_dp_cross_validation(solver, asset_pool):
    """Verifies that Pyomo/HiGHS matches or beats the Dynamic Programming solution."""
    budget_k = 300.0
    selected_pyo, cost_pyo, val_pyo, _ = solver.solve_pyomo(asset_pool, budget_k=budget_k)
    selected_dp, cost_dp, val_dp = solver.solve_dynamic_programming(asset_pool, budget_k=budget_k)

    assert cost_pyo <= budget_k
    assert cost_dp <= budget_k

    # Pyomo ILP must match DP value within integer scaling precision
    val_diff = val_pyo - val_dp
    assert val_diff >= -1e-3, f"Pyomo value ({val_pyo:.4f}) should not be worse than DP ({val_dp:.4f})"
